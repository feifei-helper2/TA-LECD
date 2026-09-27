# TA-LECD

Tolerance-aware ensemble clustering with linear exchange coordinate descent.

## Installation

Python 3.10 or newer is required. From the repository directory, run:

```bash
python -m pip install .
```

Dependencies: NumPy, SciPy, and Numba. Computation runs on CPU; the first call
includes Numba compilation time.

## Usage

```python
import numpy as np
from ta_lecd import TALECD

members = np.load("members.npy", allow_pickle=False)
model = TALECD(n_clusters=3, lambda_=1e-3, gamma=2.0,
               delta=0.25, eta=0.5, seed=2026)
labels = model.fit_predict(members)
print(model.cluster_sizes_)
print(model.converged_, model.stop_reason_)
```

`members` has shape `(n_samples, n_partitions)`. Each column contains the
integer cluster labels of one base partition, with samples in the same order
across columns. Labels must be finite; missing assignments are not supported.
Label IDs are encoded independently for each column. Generate these partitions
before calling the model; raw feature matrices are not accepted as features.

Set `n_clusters` to the desired consensus cluster count, between 1 and the
number of samples. Optional `init_labels` must use every label in `0,...,K-1`.
Otherwise, initialization uses the normalized partition representation.

| Parameter | Meaning | Default |
| --- | --- | --- |
| `lambda_ >= 0` | Size penalty strength | `0.001` |
| `gamma > 1` | Partition-weight exponent | `2.0` |
| `0 <= delta < 1` | Relative tolerance interval width | `0.25` |
| `eta >= 0` | Lower-side penalty weight | `0.5` |
| `seed` | Initialization seed | `2026` |
| `max_outer` | Maximum outer iterations | `50` |
| `cd_max_sweeps` | Maximum sweeps per relocation phase | `1000` |
| `max_exchange_rounds` | Maximum exchanges per inner solve | `1000` |

The solver alternates partition-weight updates with relocation and exchange
steps. Exchanges consider samples in different consensus clusters whose base
labels differ in exactly one partition. Clusters remain nonempty.

Results are available as `labels_`, `alpha_`, `cluster_sizes_`, and `objective_`.
`history_` contains iteration values. Check `converged_` before using the result:
if an iteration limit was reached, `stop_reason_` identifies the limit to increase.

## Command line

```bash
python -m ta_lecd members.npy --clusters 3 --output results/consensus.npz
```

Use `python -m ta_lecd --help` for options. The installed `ta-lecd` command is
equivalent. Output includes labels, weights, cluster sizes, objective values,
parameters, and convergence status. Read it with `np.load(path, allow_pickle=False)`.
Existing output files are not overwritten.
