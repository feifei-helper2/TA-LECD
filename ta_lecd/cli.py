"""Run TA-LECD on a NumPy base-partition matrix."""
import argparse
import json
from pathlib import Path
import numpy as np
from .model import TALECD


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("members", type=Path, help=".npy matrix (samples, partitions)")
    parser.add_argument("--clusters", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new .npz output")
    parser.add_argument("--lambda", dest="lambda_", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--delta", type=float, default=0.25)
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-outer", type=int, default=50)
    parser.add_argument("--cd-max-sweeps", type=int, default=1000)
    parser.add_argument("--max-exchange-rounds", type=int, default=1000)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new path")
    if args.members.suffix.lower() != ".npy" or args.output.suffix.lower() != ".npz":
        parser.error("input must be .npy and output must be .npz")
    members = np.load(args.members, allow_pickle=False)
    model = TALECD(args.clusters, lambda_=args.lambda_, gamma=args.gamma,
        delta=args.delta, eta=args.eta, seed=args.seed, max_outer=args.max_outer,
        cd_max_sweeps=args.cd_max_sweeps, max_exchange_rounds=args.max_exchange_rounds)
    model.fit(members)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    params = {k: v for k, v in vars(args).items() if k not in {"members", "output"}}
    with args.output.open("xb") as stream:
        np.savez_compressed(stream, labels=model.labels_, alpha=model.alpha_,
            cluster_sizes=model.cluster_sizes_, objective=model.objective_,
            evidence_losses=model.W_, balance_penalty=model.balance_penalty_,
            converged=model.converged_, history_json=json.dumps(model.history_),
            diagnostics_json=json.dumps(model.diagnostics_), parameters_json=json.dumps(params))
    print(json.dumps(model.diagnostics_, indent=2))


if __name__ == "__main__":
    main()
