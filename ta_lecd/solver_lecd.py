"""Relocation descent with size-preserving E1 exchanges.

E1 pairs have different consensus labels and differ in one base partition."""
from __future__ import annotations

import time
import numpy as np
from numba import njit, types
from numba.typed import Dict

from .core import (
    TAProblem,
    cd1_closure_fixed_alpha,
    objective_from_counts,
    validate_hyperparameters,
)


@njit(cache=True)
def _best_e1_exchange_one_partition(
    y,
    counts,
    enc_col,
    group_ids,
    n_groups,
    offset,
    inv_sizes,
    alpha_pow_r,
    K,
    kr,
    tol,
):
    """Find the best exchange differing only in partition r.

    Within each group, one sample per (consensus label, base label) cell
    suffices: all pairs from the same two cells have the same increment."""
    n = y.shape[0]
    # Composite-key -> cell-record index.
    cell_map = Dict.empty(key_type=types.int64, value_type=types.int64)
    heads = np.full(n_groups, -1, dtype=np.int64)
    next_cell = np.full(n, -1, dtype=np.int64)
    cell_p = np.empty(n, dtype=np.int32)
    cell_u = np.empty(n, dtype=np.int32)
    cell_rep = np.empty(n, dtype=np.int64)
    ncell = 0

    for i in range(n):
        g = int(group_ids[i])
        p = int(y[i])
        u = int(enc_col[i])
        key = (np.int64(g) * np.int64(K) + np.int64(p)) * np.int64(kr) + np.int64(u)
        if key not in cell_map:
            c = ncell
            ncell += 1
            cell_map[key] = c
            cell_p[c] = p
            cell_u[c] = u
            cell_rep[c] = i
            next_cell[c] = heads[g]
            heads[g] = c

    best_delta = 0.0
    best_i = -1
    best_j = -1

    for g in range(n_groups):
        a = heads[g]
        while a >= 0:
            p = int(cell_p[a])
            u = int(cell_u[a])
            b = next_cell[a]
            while b >= 0:
                q = int(cell_p[b])
                v = int(cell_u[b])
                if p != q and u != v:
                    gu = int(offset + u)
                    gv = int(offset + v)
                    # Change in Q_r = ||B_r^T Y||_F^2 under the swap.
                    dQ = 2.0 * (
                        (counts[q, gu] - counts[p, gu] + 1.0) * inv_sizes[gu]
                        + (counts[p, gv] - counts[q, gv] + 1.0) * inv_sizes[gv]
                    )
                    delta = -alpha_pow_r * dQ
                    if delta < best_delta - tol:
                        best_delta = delta
                        best_i = int(cell_rep[a])
                        best_j = int(cell_rep[b])
                b = next_cell[b]
            a = next_cell[a]

    return best_i, best_j, best_delta, ncell


@njit(cache=True)
def _apply_swap_inplace(y, counts, feat_idx, i, j):
    p = int(y[i])
    q = int(y[j])
    if p == q:
        return
    m = feat_idx.shape[1]
    for b in range(m):
        gi = int(feat_idx[i, b])
        gj = int(feat_idx[j, b])
        if gi == gj:
            continue
        counts[p, gi] -= 1
        counts[q, gi] += 1
        counts[q, gj] -= 1
        counts[p, gj] += 1
    y[i] = q
    y[j] = p


def best_e1_exchange(
    y: np.ndarray,
    counts: np.ndarray,
    alpha: np.ndarray,
    prob: TAProblem,
    gamma: float,
    tol: float = 1e-15,
):
    """Find the best decreasing exchange over E1."""
    y = np.ascontiguousarray(y, dtype=np.int32)
    alpha_pow = np.asarray(alpha, dtype=float) ** float(gamma)
    best_pair = None
    best_delta = 0.0
    best_r = -1
    total_cells = 0
    for r in range(prob.m):
        i, j, d, nc = _best_e1_exchange_one_partition(
            y,
            counts,
            prob.enc_T[r],
            prob.projected_T[r],
            int(prob.projected.n_groups[r]),
            int(prob.offsets[r]),
            prob.inv_sizes,
            float(alpha_pow[r]),
            int(prob.K),
            int(prob.kbs[r]),
            float(tol),
        )
        total_cells += int(nc)
        if i >= 0 and d < best_delta - tol:
            best_pair = (int(i), int(j))
            best_delta = float(d)
            best_r = r
    return best_pair, best_delta, best_r, total_cells


def lecd_closure_fixed_alpha(
    y0: np.ndarray,
    alpha: np.ndarray,
    prob: TAProblem,
    lambda_: float,
    gamma: float,
    delta: float,
    eta: float,
    max_exchange_rounds: int = 1000,
    cd_max_sweeps: int = 1000,
    tol: float = 1e-15,
):
    """Alternate relocation closure and E1 exchanges at fixed weights."""
    validate_hyperparameters(lambda_, gamma, delta, eta)
    start = time.perf_counter()
    cd = cd1_closure_fixed_alpha(
        y0, alpha, prob, lambda_, gamma, delta, eta, max_sweeps=cd_max_sweeps
    )
    y = cd["y"].copy()
    counts = cd["counts"].copy()
    sizes = cd["sizes"].copy()
    history = [{"stage": "cd1", "objective": cd["objective"]}]
    n_exchanges = 0
    scanned_cells = 0

    converged = False
    stop_reason = "max_exchange_rounds"
    for round_idx in range(int(max_exchange_rounds) + 1):
        if not cd["converged"]:
            stop_reason = "cd_max_sweeps"
            break
        pair, d, r, ncells = best_e1_exchange(y, counts, alpha, prob, gamma, tol=tol)
        scanned_cells += ncells
        if pair is None:
            converged = True
            stop_reason = "joint_closure"
            break
        if round_idx == max_exchange_rounds:
            break
        before_obj, _, _ = objective_from_counts(
            counts, sizes, alpha, prob, lambda_, gamma, delta, eta
        )
        i, j = pair
        _apply_swap_inplace(y, counts, prob.feat_idx, int(i), int(j))
        # Sizes are invariant under exchange.
        after_swap_obj, _, _ = objective_from_counts(
            counts, sizes, alpha, prob, lambda_, gamma, delta, eta
        )
        # Check the predicted increment against the updated objective.
        if abs((after_swap_obj - before_obj) - d) > 5e-9 * max(1.0, abs(before_obj)):
            raise RuntimeError("E1 exchange delta does not match the objective change")
        if after_swap_obj >= before_obj - tol:
            raise RuntimeError("accepted E1 exchange did not strictly decrease objective")

        n_exchanges += 1
        cd = cd1_closure_fixed_alpha(
            y, alpha, prob, lambda_, gamma, delta, eta, max_sweeps=cd_max_sweeps
        )
        y = cd["y"].copy()
        counts = cd["counts"].copy()
        sizes = cd["sizes"].copy()
        history.append({
            "stage": "exchange+cd1",
            "round": round_idx,
            "partition": int(r),
            "exchange_delta": float(d),
            "objective": float(cd["objective"]),
        })

    obj, W, pen = objective_from_counts(
        counts, sizes, alpha, prob, lambda_, gamma, delta, eta
    )
    return {
        "converged": converged,
        "stop_reason": stop_reason,
        "y": y,
        "counts": counts,
        "sizes": sizes,
        "objective": obj,
        "W": W,
        "penalty": pen,
        "history": history,
        "n_exchanges": n_exchanges,
        "scanned_cells": scanned_cells,
        "runtime_s": time.perf_counter() - start,
    }
