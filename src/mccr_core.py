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
