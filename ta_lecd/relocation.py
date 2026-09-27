"""Internal relocation step of LE-CD."""
import numpy as np
from numba import njit

@njit
def build_counts(y, feat_idx, K, L):
    counts = np.zeros((K, L), np.int32)
    sizes = np.zeros(K, np.int32)
    n, m = feat_idx.shape
    for i in range(n):
        c = y[i]
        sizes[c] += 1
        for b in range(m):
            counts[c, feat_idx[i, b]] += 1
    return counts, sizes

@njit
def tolerance_phi_asym(size, avg_size, delta, eta):
    """Weighted asymmetric tolerance penalty, upper-side weight normalized to 1."""
    lo = (1.0 - delta) * avg_size
    hi = (1.0 + delta) * avg_size
    if size < lo:
        d = lo - size
        return eta * d * d
    if size > hi:
        d = size - hi
        return d * d
    return 0.0

@njit
def tolerance_penalty_asym(sizes, avg_size, delta, eta):
    value = 0.0
    for c in range(sizes.shape[0]):
        value += tolerance_phi_asym(float(sizes[c]), avg_size, delta, eta)
    return value

@njit
def update_y_tolerance_asym(
    y, counts, sizes, feat_idx, inv_sizes,
    apow, lam, delta, eta, max_inner=1000
):
    n, m = feat_idx.shape
    K = sizes.shape[0]
    avg_size = n / float(K)
    for _ in range(max_inner):
        moved = 0
        for i in range(n):
            p = y[i]
            if sizes[p] <= 1:
                continue
            best_q = p
            best_delta = 0.0
            np0 = sizes[p]
            old_p = tolerance_phi_asym(float(np0), avg_size, delta, eta)
            new_p = tolerance_phi_asym(float(np0 - 1), avg_size, delta, eta)
            for q in range(K):
                if q == p:
                    continue
                dcons = 0.0
                for b in range(m):
                    g = feat_idx[i, b]
                    inv = inv_sizes[g]
                    xp = counts[p, g]
                    xq = counts[q, g]
                    sim_change = apow[b] * ((-2.0*xp + 1.0) + (2.0*xq + 1.0)) * inv
                    dcons -= sim_change
                nq0 = sizes[q]
                dpen = ((new_p - old_p)
                    + tolerance_phi_asym(float(nq0 + 1), avg_size, delta, eta)
                    - tolerance_phi_asym(float(nq0), avg_size, delta, eta))
                candidate_delta = dcons + lam * dpen
                if candidate_delta < best_delta - 1e-15:
                    best_q = q
                    best_delta = candidate_delta
            if best_q != p:
                q = best_q
                for b in range(m):
                    g = feat_idx[i, b]
                    counts[p, g] -= 1
                    counts[q, g] += 1
                sizes[p] -= 1
                sizes[q] += 1
                y[i] = q
                moved += 1
        if moved == 0:
            break
    return moved
