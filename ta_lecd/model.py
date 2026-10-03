"""TA-LECD estimator."""
from __future__ import annotations

import time
import numpy as np

from .core import (prepare_problem, initialize_labels, rebuild_counts,
                   objective_from_counts, exact_alpha_update,
                   validate_hyperparameters)
from .solver_lecd import lecd_closure_fixed_alpha


def _integer(name, value, minimum=1):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or value < minimum):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


class TALECD:
    """Consensus clustering from an (n_samples, n_partitions) label matrix.

    converged_ requires joint relocation/exchange closure and stationary
    weights. Hitting an iteration limit returns a feasible, unconverged result."""

    def __init__(self, n_clusters, lambda_=1e-3, gamma=2.0, delta=0.25,
                 eta=0.5, seed=2026, max_outer=50, cd_max_sweeps=1000,
                 max_exchange_rounds=1000):
        self.n_clusters = _integer("n_clusters", n_clusters)
        self.lambda_ = float(lambda_)
        self.gamma = float(gamma)
        self.delta = float(delta)
        self.eta = float(eta)
        validate_hyperparameters(self.lambda_, self.gamma, self.delta, self.eta)
        self.seed = _integer("seed", seed, 0)
        if self.seed > 2**32 - 1:
            raise ValueError("seed must be at most 2**32 - 1")
        self.max_outer = _integer("max_outer", max_outer)
        self.cd_max_sweeps = _integer("cd_max_sweeps", cd_max_sweeps)
        self.max_exchange_rounds = _integer("max_exchange_rounds", max_exchange_rounds, 0)

    def fit(self, members, init_labels=None):
        """Fit consensus; optional initial labels must cover 0,...,K-1."""
        start = time.perf_counter()
        prob = prepare_problem(members, self.n_clusters)
        if init_labels is None:
            y = initialize_labels(prob, self.seed)
        else:
            initial = np.asarray(init_labels)
            if (initial.shape != (prob.n,) or initial.dtype.kind not in "iuf"
                    or not np.all(np.isfinite(initial))
                    or np.any(initial != np.floor(initial))
                    or np.any(initial < 0) or np.any(initial >= prob.K)):
                raise ValueError("init_labels must be integer labels in [0, K) with shape (n,)")
            y = initial.astype(np.int32, copy=True)
            if np.any(np.bincount(y, minlength=prob.K) == 0):
                raise ValueError("init_labels must have no empty clusters")

        counts, sizes = rebuild_counts(y, prob)
        alpha = np.full(prob.m, 1.0 / prob.m)
        history = []
        total_exchanges = 0
        converged = False
        stop_reason = "max_outer"
        for outer in range(self.max_outer):
            _, W, _ = objective_from_counts(counts, sizes, alpha, prob,
                self.lambda_, self.gamma, self.delta, self.eta)
            alpha = exact_alpha_update(W, self.gamma)
            previous_y = y.copy()
            result = lecd_closure_fixed_alpha(y, alpha, prob, self.lambda_,
                self.gamma, self.delta, self.eta,
                max_exchange_rounds=self.max_exchange_rounds,
                cd_max_sweeps=self.cd_max_sweeps)
            y, counts, sizes = result["y"], result["counts"], result["sizes"]
            total_exchanges += result["n_exchanges"]
            history.append({"outer": outer + 1, "objective": result["objective"],
                            "n_exchanges": result["n_exchanges"],
                            "inner_converged": result["converged"]})
            if not result["converged"]:
                stop_reason = result["stop_reason"]
                break
            # Alpha was optimized for previous_y. If assignments are unchanged,
            # both weights and the two assignment neighborhoods are stationary.
            if np.array_equal(y, previous_y):
                converged = True
                stop_reason = "joint_closure_and_weights"
                break

        self.labels_ = y.copy()
        self.alpha_ = alpha.copy()
        self.cluster_sizes_ = sizes.copy()
        self.objective_ = float(result["objective"])
        self.W_ = result["W"].copy()
        self.balance_penalty_ = float(result["penalty"])
        self.history_ = history
        self.converged_ = converged
        self.stop_reason_ = stop_reason
        self.n_iter_ = len(history)
        self.diagnostics_ = {"solver": "lecd", "converged": converged,
            "stop_reason": stop_reason, "total_exchanges": total_exchanges,
            "outer_iterations": len(history), "runtime_s": time.perf_counter() - start}
        return self

    def fit_predict(self, members, init_labels=None):
        """Fit the model and return consensus labels."""
        return self.fit(members, init_labels=init_labels).labels_.copy()
