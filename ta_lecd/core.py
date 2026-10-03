"""TA objective, input validation, and relocation closure."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .representation import normalized_indicator
from .init import litekmeans_compat
from .relocation import (
    build_counts,
    tolerance_penalty_asym,
    update_y_tolerance_asym,
)
from .signature import build_projected_signatures, ProjectedSignatures


@dataclass
class TAProblem:
    members: np.ndarray
    F: object
    enc: np.ndarray
    kbs: np.ndarray
    offsets: np.ndarray
    feat_idx: np.ndarray
    inv_sizes: np.ndarray
    feature_base: np.ndarray
    projected: ProjectedSignatures
    enc_T: np.ndarray
    projected_T: np.ndarray
    n: int
    m: int
    K: int


def prepare_problem(members: np.ndarray, K: int) -> TAProblem:
    members = np.asarray(members)
    if members.ndim != 2:
        raise ValueError("members must have shape (n_samples, n_partitions)")
    n, m = members.shape
    if m == 0 or n == 0:
        raise ValueError("members must be nonempty")
    if members.dtype.kind not in "iuf" or not np.all(np.isfinite(members)):
        raise ValueError("members must contain finite integer labels")
    if np.any(members != np.floor(members)):
        raise ValueError("members must contain integer labels")
    if isinstance(K, (bool, np.bool_)) or not isinstance(K, (int, np.integer)):
        raise ValueError("K must be an integer")
    K = int(K)
    if not 1 <= K <= n:
        raise ValueError("K must satisfy 1 <= K <= n_samples")
    F, meta = normalized_indicator(members, None)
    enc, kbs, offsets, feat_idx, inv_sizes, feature_base = meta
    projected = build_projected_signatures(enc)
    return TAProblem(
        members=np.asarray(members), F=F, enc=enc, kbs=kbs,
        offsets=offsets, feat_idx=feat_idx, inv_sizes=inv_sizes,
        feature_base=feature_base, projected=projected,
        enc_T=np.ascontiguousarray(enc.T, dtype=np.int32),
        projected_T=np.ascontiguousarray(projected.group_ids.T, dtype=np.int32),
        n=n, m=m, K=K,
    )


def initialize_labels(prob: TAProblem, seed: int = 2026) -> np.ndarray:
    return litekmeans_compat(prob.F, prob.K, seed=int(seed), max_iter=100).astype(np.int32)


def rebuild_counts(y: np.ndarray, prob: TAProblem):
    y = np.ascontiguousarray(y, dtype=np.int32)
    return build_counts(y.copy(), prob.feat_idx, prob.K, int(prob.offsets[-1]))


def exact_alpha_update(W: np.ndarray, gamma: float, zero_tol: float = 1e-14) -> np.ndarray:
    """Minimize sum_r W_r * alpha_r**gamma on the simplex.

    If losses are numerically zero, distribute weight uniformly among them."""
    W = np.asarray(W, dtype=float)
    gamma = float(gamma)
    if not np.isfinite(gamma) or gamma <= 1.0:
        raise ValueError("gamma must be > 1")
    if W.ndim != 1 or W.size == 0 or not np.all(np.isfinite(W)):
        raise ValueError("W must be a nonempty finite vector")
    if np.any(W < -zero_tol):
        raise ValueError("W must be nonnegative")
    z = np.flatnonzero(W <= zero_tol)
    if z.size:
        alpha = np.zeros_like(W)
        alpha[z] = 1.0 / z.size
        return alpha
    exponent = 1.0 / (1.0 - gamma)
    # Log-domain normalization avoids overflow for gamma close to 1.
    logw = exponent * np.log(W)
    logw -= np.max(logw)
    tmp = np.exp(logw)
    return tmp / tmp.sum()


def compute_W_exact(counts: np.ndarray, prob: TAProblem) -> np.ndarray:
    W = np.empty(prob.m, dtype=float)
    n = prob.n
    for b in range(prob.m):
        s = 0.0
        lo, hi = int(prob.offsets[b]), int(prob.offsets[b + 1])
        block = counts[:, lo:hi].astype(float, copy=False)
        inv = prob.inv_sizes[lo:hi]
        s = float(np.sum((block * block) * inv[None, :]))
        W[b] = max(0.0, n - s)
    return W


def objective_from_counts(
    counts: np.ndarray,
    sizes: np.ndarray,
    alpha: np.ndarray,
    prob: TAProblem,
    lambda_: float,
    gamma: float,
    delta: float,
    eta: float,
):
    alpha = np.asarray(alpha, dtype=float)
    W = compute_W_exact(counts, prob)
    avg = prob.n / float(prob.K)
    pen = float(tolerance_penalty_asym(sizes, avg, float(delta), float(eta)))
    full = float(np.dot(alpha ** float(gamma), W) + float(lambda_) * pen)
    return full, W, pen


def cd1_closure_fixed_alpha(
    y0: np.ndarray,
    alpha: np.ndarray,
    prob: TAProblem,
    lambda_: float,
    gamma: float,
    delta: float,
    eta: float,
    max_sweeps: int = 1000,
):
    """Relocation phase; report whether a no-move sweep was reached."""
    y = np.ascontiguousarray(y0, dtype=np.int32).copy()
    counts, sizes = rebuild_counts(y, prob)
    before = y.copy()
    apow = np.asarray(alpha, dtype=float) ** float(gamma)
    last_moves = update_y_tolerance_asym(
        y, counts, sizes, prob.feat_idx, prob.inv_sizes,
        apow, float(lambda_), float(delta), float(eta), int(max_sweeps),
    )
    obj, W, pen = objective_from_counts(
        counts, sizes, alpha, prob, lambda_, gamma, delta, eta
    )
    return {
        "y": y,
        "counts": counts,
        "sizes": sizes,
        "objective": obj,
        "W": W,
        "penalty": pen,
        "changed": bool(np.any(y != before)),
        "converged": bool(last_moves == 0),
    }


def validate_hyperparameters(lambda_, gamma, delta, eta):
    if not np.all(np.isfinite([lambda_, gamma, delta, eta])):
        raise ValueError("hyperparameters must be finite")
    if float(lambda_) < 0:
        raise ValueError("lambda_ must be nonnegative")
    if float(gamma) <= 1:
        raise ValueError("gamma must be > 1")
    if not 0 <= float(delta) < 1:
        raise ValueError("delta must be in [0,1)")
    if float(eta) < 0:
        raise ValueError("eta must be nonnegative")
