"""Pure NumPy implementations of PAOR selection, scoring, and decisions."""
from __future__ import annotations

import json
import logging
import math
import re
from typing import Any

import numpy as np

LOG = logging.getLogger(__name__)


def select_posterior(
    V: np.ndarray,
    E_ext: np.ndarray,
    E_prior: np.ndarray,
    max_L: int = 4,
    tau_d: float = 0.15,
    tau_rho: float = 0.85,
    lam: float = 0.5,
) -> tuple[list[int], dict[str, Any]]:
    """Greedy discrimination/redundancy selection using probe responses only."""
    V = np.asarray(V, dtype=float)
    E_ext = np.asarray(E_ext, dtype=float)
    E_prior = np.asarray(E_prior, dtype=float)
    if V.ndim != 2:
        raise ValueError(f"V must be 2-D, got {V.shape}")
    n, p = V.shape
    if E_ext.shape[0] != p:
        raise ValueError("E_ext and V have incompatible criterion counts")
    if p == 0:
        return [], {"d": np.empty(0), "rho_prior": np.empty(0), "rho": np.empty(0)}
    if n == 0:
        d = np.zeros(p)
    else:
        d = np.nan_to_num(V.std(axis=0), nan=0.0)
    if E_prior.ndim == 1:
        E_prior = E_prior.reshape(0, E_ext.shape[1]) if E_prior.size == 0 else E_prior.reshape(1, -1)
    sim_prior = (E_ext @ E_prior.T).max(axis=1) if len(E_prior) else -np.ones(p)
    sim_ext = E_ext @ E_ext.T
    selected: list[int] = []
    rho_trace = np.array(sim_prior, copy=True)
    for _ in range(min(max_L, p)):
        best, best_val = -1, -np.inf
        for candidate in range(p):
            if candidate in selected or d[candidate] < tau_d:
                continue
            rho = max([sim_prior[candidate]] + [sim_ext[candidate, s] for s in selected])
            rho_trace[candidate] = rho
            if rho > tau_rho:
                continue
            value = d[candidate] - lam * rho
            if value > best_val:
                best, best_val = candidate, value
        if best < 0:
            break
        selected.append(best)
    return selected, {"d": d, "rho_prior": sim_prior, "rho": rho_trace}


def aggregate(
    F_prior: np.ndarray,
    F_post: np.ndarray,
    c: float = 0.8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute U=(S_prior+eta*S_post)/(1+eta), including degenerate cases."""
    F_prior = np.asarray(F_prior, dtype=float)
    F_post = np.asarray(F_post, dtype=float)
    if F_prior.ndim != 2 or F_post.ndim != 2:
        raise ValueError("F_prior and F_post must be 2-D")
    if F_prior.shape[0] != F_post.shape[0]:
        raise ValueError("F_prior and F_post must have the same response count")
    m = F_prior.shape[0]
    k = F_prior.shape[1]
    if k == 0:
        S_prior = np.zeros(m)
        eta = 0.0
        S_post = F_post.mean(axis=1) if F_post.shape[1] else np.zeros(m)
        return S_post, S_prior, S_post, eta
    eta = float(c) / (2.0 * k)
    S_prior = F_prior.mean(axis=1)
    S_post = F_post.mean(axis=1) if F_post.shape[1] else np.zeros(m)
    U = (S_prior + eta * S_post) / (1.0 + eta)
    return U, S_prior, S_post, eta


def pairwise_decide(U_a: float, U_b: float, eps: float = 1e-4) -> str:
    if U_a - U_b > eps:
        return "A"
    if U_b - U_a > eps:
        return "B"
    return "fallback"


def expected_score(logprobs: dict[int, float] | None, discrete: int | float = 1) -> float:
    """Normalize available 0/1/2 logprobs and return E[score]/2."""
    if not logprobs:
        return float(discrete) / 2.0
    values = np.array([logprobs.get(i, -np.inf) for i in range(3)], dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return float(discrete) / 2.0
    values = values[finite]
    values = np.exp(values - values.max())
    probs = values / values.sum()
    levels = np.arange(3, dtype=float)[finite]
    return float(np.dot(probs, levels) / 2.0)


def extract_json(text: str) -> Any:
    """Parse JSON from raw text or a fenced code block."""
    text = text.strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates = [fenced.group(1)] if fenced else []
    candidates.extend([text[text.find("{"): text.rfind("}") + 1] if "{" in text and "}" in text else "",
                       text[text.find("["): text.rfind("]") + 1] if "[" in text and "]" in text else ""])
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError("no valid JSON object or array found")


def parse_rubrics(text: str, key: str, min_count: int, max_count: int) -> list[dict[str, str]]:
    try:
        data = extract_json(text)
    except ValueError as error:
        recovered = []
        pattern = (r'\{\s*"id"\s*:\s*"([^"]+)"\s*,\s*"(?:criterion|description|title)"\s*:\s*'
                   r'"((?:\\.|[^"\\])*)"\s*\}')
        for match in re.finditer(pattern, text, flags=re.DOTALL | re.IGNORECASE):
            try:
                criterion = json.loads(f'"{match.group(2)}"').strip()
            except json.JSONDecodeError:
                continue
            if criterion:
                recovered.append({"id": match.group(1), "criterion": criterion})
            if len(recovered) >= max_count:
                break
        if len(recovered) < min_count:
            raise error
        LOG.warning("Recovered %d complete rubrics from malformed JSON key=%s", len(recovered), key)
        return recovered
    raw = data.get(key) if isinstance(data, dict) else data
    if not isinstance(raw, list):
        raise ValueError(f"missing rubric list {key}")
    rubrics = []
    for i, item in enumerate(raw[:max_count]):
        if isinstance(item, str):
            criterion = item.strip()
            rid = f"R{i + 1}"
        elif isinstance(item, dict):
            normalized = {str(key).lower(): value for key, value in item.items()}
            criterion = str(normalized.get("criterion", normalized.get("description", normalized.get("title", "")))).strip()
            rid = str(normalized.get("id", f"R{i + 1}"))
        else:
            continue
        if criterion:
            rubrics.append({"id": rid, "criterion": criterion})
    if len(rubrics) < min_count:
        raise ValueError(f"rubric count {len(rubrics)} below {min_count}")
    return rubrics


def _score_fields(text: str) -> dict[str, int]:
    """Extract score-bearing fields from one or several JSON objects."""
    fields: dict[str, int] = {}
    try:
        data = extract_json(text)
    except ValueError:
        data = None
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict):
                value = value.get("score")
            try:
                value = int(value)
            except (TypeError, ValueError):
                continue
            if value in (0, 1, 2):
                fields[str(key)] = value
    nested = r'"([^"]+)"\s*:\s*\{[^{}]*?"score"\s*:\s*"?([012])'
    for match in re.finditer(nested, text, flags=re.DOTALL):
        fields.setdefault(match.group(1), int(match.group(2)))
    direct = r'"([^"]+)"\s*:\s*"?([012])"?(?=\s*[,}\n])'
    for match in re.finditer(direct, text):
        fields.setdefault(match.group(1), int(match.group(2)))
    return fields


def _field_for_id(fields: dict[str, int], cid: str) -> int | None:
    if cid in fields:
        return fields[cid]
    suffix = re.search(r"(\d+)$", cid)
    if suffix:
        matches = [value for key, value in fields.items() if re.search(r"(\d+)$", key) and re.search(r"(\d+)$", key).group(1) == suffix.group(1)]
        if len(matches) == 1:
            return matches[0]
    return None


def parse_atomic_scores(text: str, criterion_ids: list[str]) -> tuple[dict[str, int], int]:
    """Return integer scores and the number of genuinely missing criteria."""
    fields = _score_fields(text)
    scores = {cid: value for cid in criterion_ids if (value := _field_for_id(fields, cid)) is not None}
    return scores, len(criterion_ids) - len(scores)


def _score_field_position(text: str, cid: str) -> int | None:
    patterns = [r'"([^"]+)"\s*:\s*\{[^{}]*?"score"\s*:\s*', r'"([^"]+)"\s*:\s*"?']
    suffix = re.search(r"(\d+)$", cid)
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.DOTALL):
            key = match.group(1)
            if key == cid or (suffix and re.search(r"(\d+)$", key) and re.search(r"(\d+)$", key).group(1) == suffix.group(1)):
                return match.end()
    return None


def parse_score_token_logprobs(text: str, token_entries: list[dict[str, Any]], criterion_ids: list[str]) -> dict[str, dict[int, float]]:
    """Map the first token after each score field to its top-logprob candidates."""
    if not token_entries:
        return {}
    pieces = [str(entry.get("token", "")) for entry in token_entries]
    offsets = []
    cursor = 0
    for piece in pieces:
        offsets.append((cursor, cursor + len(piece)))
        cursor += len(piece)
    result: dict[str, dict[int, float]] = {}
    for cid in criterion_ids:
        position = _score_field_position(text, cid)
        if position is None:
            continue
        token_index = next((i for i, (start, end) in enumerate(offsets) if start <= position < end), None)
        if token_index is None:
            token_index = next((i for i, (start, end) in enumerate(offsets) if start >= position), None)
        if token_index is None:
            continue
        entry = token_entries[token_index]
        candidates: dict[int, float] = {}
        for top in entry.get("top_logprobs", []) or []:
            token = str(top.get("token", "")).strip()
            number = re.match(r"^([012])(?:\D|$)", token)
            if number:
                candidates[int(number.group(1))] = float(top.get("logprob", -np.inf))
        own_token = pieces[token_index].strip()
        number = re.match(r"^([012])(?:\D|$)", own_token)
        if number and "logprob" in entry:
            candidates.setdefault(int(number.group(1)), float(entry["logprob"]))
        if candidates:
            result[cid] = candidates
    return result


def softmax_expected_from_entries(text: str, entries: list[dict[str, Any]], criterion_ids: list[str], discrete: dict[str, int]) -> tuple[dict[str, float], dict[str, dict[int, float]]]:
    logprobs = parse_score_token_logprobs(text, entries, criterion_ids)
    values = {cid: expected_score(logprobs.get(cid), discrete.get(cid, 1)) for cid in criterion_ids}
    return values, logprobs


def unit_defect_band(K: int, c: float = 0.8) -> tuple[float, float]:
    if K <= 0:
        raise ValueError("K must be positive")
    return c / (2 * K), 1 / (2 * K)
