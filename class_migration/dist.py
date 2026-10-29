"""
write_distributions.py — regenerate the per-block class proportions for a
completed Dirichlet run.

horizon.py --mode class needs distribution_client*.csv to work out when
each class stopped being trained. Runs made before the client wrote that
file can have it reconstructed here, because the proportions are a
deterministic function of seed, client index, block and alpha.

    python write_distributions.py --logs logs/pathmnist_dirichlet_path_diri \\
        --dataset pathmnist --alpha 0.5 --clients 3 --blocks 4

The arguments must match the run; they are echoed in the client log's
header if you need to check.
"""

import argparse
import csv
import os

from datasets import get_dataset
from scenarios import build_scenario


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logs", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--clients", type=int, default=3)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing files")
    args = ap.parse_args()

    spec = get_dataset(args.dataset)
    sc = build_scenario("dirichlet", spec, num_blocks=args.blocks,
                        alpha=args.alpha)
    if not hasattr(sc, "distribution"):
        raise SystemExit("not a Dirichlet scenario")

    names = [sc.unit_name(u) for u in range(sc.num_classes)]
    os.makedirs(args.logs, exist_ok=True)

    for cid in range(args.clients):
        path = os.path.join(args.logs, f"distribution_client{cid}.csv")
        if os.path.exists(path) and not args.force:
            print(f"  {os.path.basename(path)}: exists, skipping "
                  f"(use --force to overwrite)")
            continue
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["block"] + [f"p_{n}" for n in names])
            for b in range(args.blocks):
                p = sc.distribution(cid, b)
                w.writerow([b] + [f"{v:.6f}" for v in p])
        dom = [names[int(sc.distribution(cid, b).argmax())]
               for b in range(args.blocks)]
        print(f"  {os.path.basename(path)}: written "
              f"(dominant per block: {', '.join(dom)})")

    print(f"\nNow score with:\n  python horizon.py --logs {args.logs} "
          f"--mode class --sweep --tau-sweep "
          f"--out {args.logs}/horizon.csv")


if __name__ == "__main__":
    main()
