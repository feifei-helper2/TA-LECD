import numpy as np
from scipy import sparse
from .common import relabel0

def encode_members(members):
    M = np.asarray(members)
    n, m = M.shape
    enc = np.empty((n, m), dtype=np.int32)
    kbs = np.empty(m, dtype=np.int32)
    sizes_list = []
    offsets = np.zeros(m+1, dtype=np.int32)

    for b in range(m):
        z = relabel0(M[:, b])
        enc[:, b] = z
        kb = int(z.max()+1)
        kbs[b] = kb
        sizes = np.bincount(z, minlength=kb).astype(np.int32)
        sizes_list.append(sizes)
        offsets[b+1] = offsets[b] + kb

    L = int(offsets[-1])
    feat_idx = np.empty((n, m), dtype=np.int32)
    inv_sizes = np.empty(L, dtype=np.float64)
    feature_base = np.empty(L, dtype=np.int32)

    for b, (z, sizes) in enumerate(zip(enc.T, sizes_list)):
        off = offsets[b]
        feat_idx[:, b] = off + z
        inv_sizes[off:off+len(sizes)] = 1.0 / sizes
        feature_base[off:off+len(sizes)] = b

    return enc, kbs, offsets, feat_idx, inv_sizes, feature_base

def normalized_indicator(members, base_weights=None):
    enc, kbs, offsets, feat_idx, inv_sizes, feature_base = encode_members(members)
    n, m = enc.shape
    if base_weights is None:
        base_weights = np.ones(m, dtype=float)
    base_weights = np.asarray(base_weights, dtype=float)

    rows = np.repeat(np.arange(n), m)
    cols = feat_idx.ravel()
    vals = (np.sqrt(base_weights[None, :]) * np.sqrt(inv_sizes[feat_idx])).ravel()

    X = sparse.csr_matrix(
        (vals, (rows, cols)),
        shape=(n, int(offsets[-1]))
    )
    return X, (enc, kbs, offsets, feat_idx, inv_sizes, feature_base)
