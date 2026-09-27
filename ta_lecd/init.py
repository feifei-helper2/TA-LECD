import numpy as np
from scipy import sparse

def litekmeans_compat(X, k, seed=2026, max_iter=100):
    """
    Lloyd initialization for dense or sparse partition representations:
      - sample initialization
      - squared Euclidean Lloyd iterations
      - farthest-point repair from non-singleton donor clusters
    """
    n = X.shape[0]
    if not isinstance(k, (int, np.integer)) or not 1 <= k <= n:
        raise ValueError("k must be an integer satisfying 1 <= k <= n_samples.")
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer.")
    rng = np.random.RandomState(seed)
    idx = rng.choice(n, size=k, replace=False)
    centers = X[idx].toarray() if sparse.issparse(X) else np.asarray(X[idx]).copy()

    label = np.ones(n, dtype=np.int32)
    last = np.zeros(n, dtype=np.int32)

    for _ in range(max_iter):
        if np.array_equal(label, last):
            break
        last = label.copy()

        bb = np.sum(centers * centers, axis=1)
        ab = X.dot(centers.T) if sparse.issparse(X) else X @ centers.T
        D = bb[None, :] - 2.0 * np.asarray(ab)
        val = D.min(axis=1)
        label = D.argmin(axis=1).astype(np.int32)

        uniq = np.unique(label)
        if len(uniq) < k:
            missing = np.setdiff1d(np.arange(k), uniq)
            aa = (np.asarray(X.multiply(X).sum(axis=1)).ravel()
                  if sparse.issparse(X) else np.sum(X*X, axis=1))
            # Visit samples by decreasing distance without emptying a donor.
            # Update counts after each move; this also handles repeated rows.
            ranked = np.argsort(aa + val)[::-1]
            live_counts = np.bincount(label, minlength=k)
            cursor = 0
            for empty in missing:
                while live_counts[label[ranked[cursor]]] <= 1:
                    cursor += 1
                sample = ranked[cursor]
                donor = label[sample]
                label[sample] = empty
                live_counts[donor] -= 1
                live_counts[empty] += 1
                cursor += 1

        counts = np.bincount(label, minlength=k).astype(float)
        Y = sparse.csr_matrix(
            (np.ones(n), (label, np.arange(n))),
            shape=(k, n)
        )
        sums = Y.dot(X)
        centers = np.asarray(
            sums.toarray() if sparse.issparse(sums) else sums
        ) / counts[:, None]

    if np.any(np.bincount(label, minlength=k) == 0):
        raise RuntimeError("Initialization failed to produce nonempty clusters.")
    return label
