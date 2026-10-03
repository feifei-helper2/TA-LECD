"""Group samples by their base labels with one partition omitted.

Prefix/suffix IDs avoid copying a full signature for each omitted column."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ProjectedSignatures:
    group_ids: np.ndarray      # shape (n, m), int32
    n_groups: np.ndarray       # shape (m,), int32


def _refine_pair_ids(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, int]:
    """Assign a shared ID to equal integer pairs."""
    n = a.shape[0]
    out = np.empty(n, dtype=np.int32)
    table: dict[tuple[int, int], int] = {}
    next_id = 0
    for i in range(n):
        key = (int(a[i]), int(b[i]))
        cid = table.get(key)
        if cid is None:
            cid = next_id
            table[key] = cid
            next_id += 1
        out[i] = cid
    return out, next_id


def build_projected_signatures(enc: np.ndarray) -> ProjectedSignatures:
    """Return group IDs for an (n, m) encoded partition matrix.

    Samples share group_ids[:, r] when all labels except column r agree."""
    enc = np.ascontiguousarray(enc, dtype=np.int32)
    if enc.ndim != 2:
        raise ValueError("enc must be a 2-D integer array")
    n, m = enc.shape
    if n == 0 or m == 0:
        raise ValueError("enc must have at least one sample and one partition")

    # prefix[:, b] identifies enc[:, :b]
    prefix = np.empty((n, m + 1), dtype=np.int32)
    prefix[:, 0] = 0
    for b in range(m):
        prefix[:, b + 1], _ = _refine_pair_ids(prefix[:, b], enc[:, b])

    # suffix[:, b] identifies enc[:, b:]
    suffix = np.empty((n, m + 1), dtype=np.int32)
    suffix[:, m] = 0
    for b in range(m - 1, -1, -1):
        suffix[:, b], _ = _refine_pair_ids(enc[:, b], suffix[:, b + 1])

    group_ids = np.empty((n, m), dtype=np.int32)
    n_groups = np.empty(m, dtype=np.int32)
    for r in range(m):
        group_ids[:, r], ng = _refine_pair_ids(prefix[:, r], suffix[:, r + 1])
        n_groups[r] = ng
    return ProjectedSignatures(group_ids=group_ids, n_groups=n_groups)
