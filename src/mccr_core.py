"""Semantic consensus and orthogonal residual selection for MCCR-Eval."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def semantic_consensus_clusters(
    embeddings: np.ndarray,
    views: np.ndarray,
    *,
    theta_sim: float = 0.8,
    beta: float = 0.4,
) -> list[dict[str, Any]]:
    embeddings = np.asarray(embeddings, dtype=float)
    views = np.asarray(views, dtype=int)
    n = len(embeddings)
    if n == 0 or views.shape != (n,):
        raise ValueError("Embeddings and view IDs are empty or incompatible")
    if n == 1:
        labels = np.ones(1, dtype=int)
    else:
        distances = np.clip(1.0 - embeddings @ embeddings.T, 0.0, 2.0)
        distances = (distances + distances.T) / 2
        np.fill_diagonal(distances, 0.0)
        hierarchy = linkage(squareform(distances, checks=False), method="average")
        labels = fcluster(hierarchy, t=1.0 - theta_sim, criterion="distance")
    response_views = sorted(set(views[views > 0].tolist()))
    clusters = []
    for label in sorted(set(labels.tolist()), key=lambda x: int(np.flatnonzero(labels == x)[0])):
        indices = np.flatnonzero(labels == label)
        local = embeddings[indices]
        similarities = local @ local.T
        medoid = int(indices[int(np.argmax(similarities.mean(axis=1)))])
        prior_present = bool(np.any(views[indices] == 0))
        response_support = (float(np.mean([np.any(views[indices] == m) for m in response_views]))
                            if response_views else 0.0)
        rho = beta * float(prior_present) + (1.0 - beta) * response_support
        clusters.append({"members": indices.tolist(), "medoid": medoid,
                         "prior_present": prior_present, "response_support": response_support,
                         "rho": float(rho)})
    return clusters


def orthogonal_residual_select(
    F_core: np.ndarray,
    F_ext: np.ndarray,
    supports: np.ndarray,
    *,
    max_L: int = 4,
    tau_var: float = 0.04,
    rho_min: float = 0.2,
) -> dict[str, Any]:
    core = np.asarray(F_core, dtype=float)
    ext = np.asarray(F_ext, dtype=float)
    supports = np.asarray(supports, dtype=float)
    if ext.ndim != 2 or len(supports) != ext.shape[1]:
        raise ValueError("Invalid extension score matrix or supports")
    m = ext.shape[0]
    if m == 0:
        raise ValueError("At least one probe is required")
    core_centered = core - core.mean(axis=0, keepdims=True) if core.size else np.empty((m, 0))
    ext_centered = ext - ext.mean(axis=0, keepdims=True)
    if core_centered.shape[1]:
        u, singular, _ = np.linalg.svd(core_centered, full_matrices=False)
        rank = int(np.sum(singular > 1e-8))
        basis = u[:, :rank]
    else:
        rank = 0
        basis = np.empty((m, 0))
    selected: list[int] = []
    sigmas: list[float] = []
    variances: list[float] = []
    for _ in range(min(max_L, ext.shape[1])):
        best = -1
        best_variance = -1.0
        best_score = -np.inf
        best_residual = None
        for p in range(ext.shape[1]):
            if p in selected:
                continue
            vector = ext_centered[:, p]
            residual = vector - basis @ (basis.T @ vector) if basis.shape[1] else vector.copy()
            variance = float(residual @ residual / m)
            score = max(float(supports[p]), rho_min) * variance
            if score > best_score:
                best, best_variance, best_score, best_residual = p, variance, score, residual
        if best < 0 or best_variance < tau_var:
            break
        norm = float(np.linalg.norm(best_residual))
        if norm <= 1e-12:
            break
        selected.append(best)
        variances.append(best_variance)
        sigmas.append(float(np.sqrt(best_variance)))
        basis = np.column_stack([basis, best_residual / norm])
    return {"selected": selected, "sigmas": sigmas, "variances": variances,
            "core_rank": rank, "final_rank": int(basis.shape[1])}


def gated_aggregate(
    F_core: np.ndarray,
    F_ext: np.ndarray,
    core_support: np.ndarray,
    residual_sigmas: np.ndarray,
    *,
    gamma: float = 2.0,
    eta: float = 0.5,
) -> dict[str, np.ndarray]:
    core = np.asarray(F_core, dtype=float)
    ext = np.asarray(F_ext, dtype=float)
    core_support = np.asarray(core_support, dtype=float)
    residual_sigmas = np.asarray(residual_sigmas, dtype=float)
    if core.ndim != 2 or core.shape[1] == 0 or core.shape[1] != len(core_support):
        raise ValueError("A non-empty core with matching support is required")
    if ext.ndim != 2 or ext.shape[0] != core.shape[0] or ext.shape[1] != len(residual_sigmas):
        raise ValueError("Invalid target extension matrix")
    core_weights = core_support / core_support.sum()
    s_core = core @ core_weights
    if ext.shape[1]:
        ext_weights = residual_sigmas / (residual_sigmas.sum() + 1e-12)
        s_ext = ext @ ext_weights
    else:
        ext_weights = np.empty(0)
        s_ext = np.zeros(core.shape[0])
    utility = s_core + eta * np.power(s_core, gamma) * s_ext
    return {"U": utility, "S_core": s_core, "S_ext": s_ext,
            "core_weights": core_weights, "ext_weights": ext_weights}


# ---------------------------------------------------------------------------
# MCCR v2 primitives: directional (weakly supervised) selection and whitening.
# ---------------------------------------------------------------------------

def directional_alignment(F: np.ndarray, known_pairs: list[tuple[int, int]]) -> np.ndarray:
    """Mean signed score gap F[better] - F[worse] per criterion over known-ordered probe pairs.

    ``known_pairs`` holds (better_row, worse_row) indices, e.g. (strong probe, weak probe) or
    (original probe, counterfactually degraded copy). Values lie in [-1, 1]; a negative value
    means the criterion prefers the presumably worse response (RRD-style misalignment).
    """
    F = np.asarray(F, dtype=float)
    if F.ndim != 2:
        raise ValueError("F must be 2-D")
    if not known_pairs:
        return np.zeros(F.shape[1])
    better = np.asarray([b for b, _ in known_pairs], dtype=int)
    worse = np.asarray([w for _, w in known_pairs], dtype=int)
    return (F[better] - F[worse]).mean(axis=0)


def ledoit_wolf_covariance(X: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf (2004) shrinkage towards a scaled identity; X is (samples, features)."""
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    if p == 0:
        return np.empty((0, 0)), 0.0
    Xc = X - X.mean(axis=0, keepdims=True)
    S = Xc.T @ Xc / max(n, 1)
    mu = float(np.trace(S) / p)
    delta = float(np.sum((S - mu * np.eye(p)) ** 2) / p)
    if n <= 1 or delta <= 1e-12:
        return mu * np.eye(p) if mu > 0 else np.eye(p), 1.0
    beta = 0.0
    for row in Xc:
        outer = np.outer(row, row)
        beta += float(np.sum((outer - S) ** 2))
    beta = min(beta / (n * n) / p, delta)
    shrink = beta / delta
    cov = shrink * mu * np.eye(p) + (1.0 - shrink) * S
    return cov, float(shrink)


def whitened_weights(F: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    """Correlation-aware weights w ∝ Σ^{-1/2} 1 (RRD_WU), with Ledoit-Wolf shrinkage.

    Small probe sets make the sample covariance singular, so shrinkage is required. Negative
    weights are clipped to zero; a degenerate matrix falls back to uniform weights.
    """
    F = np.asarray(F, dtype=float)
    p = F.shape[1]
    if p == 0:
        return np.empty(0)
    cov, _ = ledoit_wolf_covariance(F)
    values, vectors = np.linalg.eigh(cov + eps * np.eye(p))
    values = np.clip(values, eps, None)
    inv_sqrt = vectors @ np.diag(values ** -0.5) @ vectors.T
    weights = np.clip(inv_sqrt @ np.ones(p), 0.0, None)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 1e-12:
        return np.full(p, 1.0 / p)
    return weights / total


def contrastive_select(
    F_core: np.ndarray,
    F_ext: np.ndarray,
    alignment: np.ndarray,
    *,
    max_L: int = 4,
    tau_align: float = 0.05,
    max_collinearity: float = 0.9,
) -> dict[str, Any]:
    """Greedy supplementary-criterion selection driven by known-direction evidence.

    Candidates are ranked by directional alignment (how consistently they prefer the known
    better probe). A candidate is skipped when its centred score profile is almost entirely
    explained by the core and already-selected criteria (residual energy fraction below
    ``1 - max_collinearity``). Unlike residual-variance selection this does not require the
    probe matrix to have spare rank: a collinearity test is meaningful even for 6 rows.
    """
    core = np.asarray(F_core, dtype=float)
    ext = np.asarray(F_ext, dtype=float)
    alignment = np.asarray(alignment, dtype=float)
    if ext.ndim != 2 or len(alignment) != ext.shape[1]:
        raise ValueError("Invalid extension score matrix or alignment")
    m = ext.shape[0]
    centred_core = core - core.mean(axis=0, keepdims=True) if core.size else np.empty((m, 0))
    centred_ext = ext - ext.mean(axis=0, keepdims=True)
    if centred_core.shape[1]:
        u, singular, _ = np.linalg.svd(centred_core, full_matrices=False)
        basis = u[:, :int(np.sum(singular > 1e-8))]
    else:
        basis = np.empty((m, 0))
    core_rank = int(basis.shape[1])
    selected: list[int] = []
    skipped_collinear: list[int] = []
    order = sorted(range(ext.shape[1]), key=lambda p: (-alignment[p], p))
    for p in order:
        if len(selected) >= max_L or alignment[p] < tau_align:
            break
        vector = centred_ext[:, p]
        energy = float(vector @ vector)
        if energy > 1e-12:
            residual = vector - basis @ (basis.T @ vector) if basis.shape[1] else vector
            fraction = float(residual @ residual) / energy
            if fraction < 1.0 - max_collinearity:
                skipped_collinear.append(p)
                continue
            if float(np.linalg.norm(residual)) > 1e-8:
                basis = np.column_stack([basis, residual / np.linalg.norm(residual)])
        selected.append(p)
    return {"selected": selected, "alignment": [float(alignment[p]) for p in selected],
            "sigmas": [max(float(alignment[p]), 1e-6) for p in selected],
            "skipped_collinear": skipped_collinear, "core_rank": core_rank, "final_rank": int(basis.shape[1])}


def hierarchical_aggregate(
    F_core: np.ndarray,
    F_ext: np.ndarray,
    core_weights: np.ndarray,
    ext_weights: np.ndarray,
    *,
    mode: str = "lexicographic",
    c: float = 0.8,
    eta: float = 0.5,
    gamma: float = 2.0,
) -> dict[str, np.ndarray]:
    """Combine core and supplementary scores.

    ``lexicographic`` (PAOR band generalised to weighted cores): U = S_core + c·min_k(w_k)/2·S_ext.
    With 0/1/2 criterion scores the smallest possible core gap is min_k(w_k)/2 (half a point on
    the least-weighted core criterion); since c < 1 and S_ext ∈ [0, 1], the supplementary term can
    never overturn a strict core ordering. With uniform weights this is PAOR's c/(2K).
    ``gated``: MCCR v1 formula; ``linear``: (S_core + η S_ext) / (1 + η).
    """
    core = np.asarray(F_core, dtype=float)
    ext = np.asarray(F_ext, dtype=float)
    core_weights = np.asarray(core_weights, dtype=float)
    ext_weights = np.asarray(ext_weights, dtype=float)
    if core.ndim != 2 or core.shape[1] == 0 or core.shape[1] != len(core_weights):
        raise ValueError("A non-empty core with matching weights is required")
    core_weights = core_weights / core_weights.sum()
    s_core = core @ core_weights
    if ext.shape[1]:
        ext_weights = ext_weights / max(float(ext_weights.sum()), 1e-12)
        s_ext = ext @ ext_weights
    else:
        s_ext = np.zeros(core.shape[0])
    if mode == "lexicographic":
        utility = s_core + c * float(core_weights.min()) / 2.0 * s_ext
    elif mode == "gated":
        utility = s_core + eta * np.power(s_core, gamma) * s_ext
    elif mode == "linear":
        utility = (s_core + eta * s_ext) / (1.0 + eta)
    else:
        raise ValueError(f"Unknown aggregation mode {mode}")
    return {"U": utility, "S_core": s_core, "S_ext": s_ext, "core_weights": core_weights, "ext_weights": ext_weights}


# ---------------------------------------------------------------------------
# MCCR-ME primitives: grounding, view-origin classes, hitting sets, deletion minimisation.
# ---------------------------------------------------------------------------

ME_PRIORITY = {"explicit": 3, "both": 2, "query_implicit": 1, "response_only": 0, "sufficiency": 0}


def _normalise(text: str) -> str:
    return " ".join(str(text).lower().split())


def span_in_query(evidence: str | None, query: str, min_chars: int = 3) -> bool:
    """True when an "explicit" criterion's evidence is really a (whitespace/case-normalised) span of the query."""
    if not evidence:
        return False
    evidence = _normalise(evidence).strip(" \"'“”.,;:")
    return len(evidence) >= min_chars and evidence in _normalise(query)


def classify_clusters(clusters: list[dict[str, Any]], U: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    """Assign each semantic cluster to the query/response comparison classes.

    * ``explicit``        a query-view member is stated in the query (verified evidence span);
    * ``both``            query view and at least one response view proposed it (Q∩R);
    * ``query_implicit``  only the query view proposed it and it is not explicitly stated (Q-only);
    * ``response_only``   only response views proposed it (R-only).

    The representative keeps query-grounded wording: the verified explicit member for ``explicit``,
    the medoid when it is a query-view member, otherwise the first query-view member; response-only
    clusters use the medoid.
    """
    output = []
    for cluster_id, cluster in enumerate(clusters):
        members = [U[j] for j in cluster["members"]]
        prior = [x for x in members if int(x["view"]) == 0]
        responses = {int(x["view"]) for x in members if int(x["view"]) > 0}
        explicit = [x for x in prior if x.get("type") == "explicit" and span_in_query(x.get("evidence"), query)]
        if explicit:
            label, representative = "explicit", explicit[0]
        elif prior and responses:
            label, representative = "both", prior[0]
        elif prior:
            label, representative = "query_implicit", prior[0]
        else:
            label, representative = "response_only", U[cluster["medoid"]]
        if prior and U[cluster["medoid"]] in prior:
            representative = U[cluster["medoid"]] if label != "explicit" else representative
        output.append({"id": f"E{cluster_id + 1}", "criterion": representative["criterion"], "class": label,
                       "cluster_id": cluster_id, "evidence": representative.get("evidence") if label == "explicit" else None,
                       "response_support": float(cluster.get("response_support", 0.0)),
                       "n_response_views": len(responses), "rho": float(cluster.get("rho", 0.0))})
    return output


def greedy_hitting_set(F: np.ndarray, known_pairs: list[tuple[int, int]], *, tau_align: float = 0.05
                       ) -> tuple[list[int], dict[str, Any]]:
    """Smallest (greedy, ln-approximate) set of criteria explaining every known-ordered pair.

    Each pair (better, worse) defines a conflict set {c : F[better, c] > F[worse, c]}. Criteria whose
    mean signed gap is below ``-tau_align`` prefer the worse response and are excluded (RRD-style).
    Pairs no remaining criterion separates are reported as unexplained. Dual of Reiter's minimal
    diagnoses: a minimal explanation is a minimal hitting set of the conflict sets.
    """
    F = np.asarray(F, dtype=float)
    if F.ndim != 2 or F.shape[1] == 0:
        return [], {"unexplained_pairs": len(known_pairs), "misaligned": []}
    alignment = directional_alignment(F, known_pairs) if known_pairs else np.zeros(F.shape[1])
    allowed = {c for c in range(F.shape[1]) if alignment[c] >= -tau_align}
    conflict = [{c for c in allowed if F[b, c] > F[w, c]} for b, w in known_pairs]
    uncovered = {i for i, values in enumerate(conflict) if values}
    unexplained = sum(1 for values in conflict if not values)
    selected: list[int] = []
    while uncovered:
        best = max(sorted(allowed - set(selected)), key=lambda c: sum(c in conflict[i] for i in uncovered), default=None)
        if best is None or not any(best in conflict[i] for i in uncovered):
            break
        selected.append(best)
        uncovered = {i for i in uncovered if best not in conflict[i]}
    return selected, {"unexplained_pairs": unexplained, "misaligned": sorted(set(range(F.shape[1])) - allowed),
                      "alignment": alignment.tolist()}


def deletion_minimize(items: list[dict[str, Any]], implied_by: dict[str, list[str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Linear deletion-based minimisation (as for minimal unsatisfiable subsets).

    Items are visited from lowest to highest priority (``ME_PRIORITY`` then support). An item is deleted
    when it is implied by criteria that are all still kept; deleting in this order means a mutually
    implying pair keeps its higher-priority member and cycles cannot delete both.
    """
    order = sorted(items, key=lambda x: (ME_PRIORITY.get(x.get("class", ""), 0), x.get("response_support", 0.0), x["id"]))
    kept = {x["id"] for x in items}
    removed = []
    for item in order:
        sources = [s for s in implied_by.get(item["id"], []) if s != item["id"]]
        if sources and all(s in kept for s in sources):
            kept.discard(item["id"])
            removed.append({"id": item["id"], "criterion": item["criterion"], "implied_by": sources})
    return [x for x in items if x["id"] in kept], removed
