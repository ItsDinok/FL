#!/usr/bin/env python3
"""
analyse_steps.py — step-level prediction horizon.

Round-level Delta t is quantised to communication rounds, so when
forgetting completes inside a single round the metric sits at its floor
and cannot separate signals. This script recomputes the horizon in
optimiser steps using the intra-round probes written by the client
(steps_client*.csv), which makes the pre- versus post-training
distinction quantitative rather than categorical.

Definitions (step-level analogues of the round-level ones):
    t_forget_step(d) = first global step after domain d stops training at
                       which its retained accuracy falls tau below its
                       running maximum.
    t_signal_step(s) = the global step at which signal s fired, which for
                       a pre-training signal is step 0 of the round it
                       fired in, and for a post-training signal is the
                       final step of that round.
    Delta t_step     = t_forget_step - t_signal_step

Usage:
    python3 analyse_steps.py --logs logs_dil/ --rounds-per-block 5 \
                             --horizon logs_dil/horizon_dil.csv
"""

import argparse
import csv
import glob
import math
import os
from collections import defaultdict

TAU = 0.05

# which family each signal belongs to, and therefore when in the round it
# is actually computed
PRE_SIGNALS = {"new_loss", "new_entropy", "grad_norm_new", "grad_norm_old",
               "grad_cos", "grad_inner", "pred_dL_old"}


def read_csvs(pattern):
    out = {}
    for path in sorted(glob.glob(pattern)):
        cid = "".join(c for c in os.path.basename(path) if c.isdigit())
        rows = []
        with open(path) as f:
            for row in csv.DictReader(f):
                parsed = {}
                for k, v in row.items():
                    try:
                        parsed[k] = float(v)
                    except (TypeError, ValueError):
                        parsed[k] = math.nan
                rows.append(parsed)
        out[cid] = rows
    return out


def fleet_steps(store, col):
    """Average a column across clients, keyed by global step."""
    acc = defaultdict(list)
    for rows in store.values():
        for r in rows:
            v = r.get(col, math.nan)
            if not math.isnan(v):
                acc[int(r["global_step"])].append(v)
    return {s: sum(v) / len(v) for s, v in sorted(acc.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs_dil")
    ap.add_argument("--rounds-per-block", type=int, default=5)
    ap.add_argument("--num-domains", type=int, default=4)
    ap.add_argument("--tau", type=float, default=TAU)
    ap.add_argument("--horizon", default=None,
                    help="horizon_table.csv from horizon.py, to convert "
                         "round-level firings into step-level ones")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    steps = read_csvs(os.path.join(args.logs, "steps_client*.csv"))
    if not steps:
        raise SystemExit(f"no steps_client*.csv in {args.logs} — was "
                         "INTRA_ROUND_EVAL enabled for this run?")

    # steps per round, taken from the data itself
    per_round = defaultdict(set)
    for rows in steps.values():
        for r in rows:
            per_round[int(r["round"])].add(int(r["step"]))
    spr = max(max(v) for v in per_round.values())
    print(f"steps per round: {spr}")

    # ── step-level forgetting events ────────────────────────────────
    events = {}
    for d in range(args.num_domains):
        series = fleet_steps(steps, f"acc_task{d}")
        if not series:
            continue
        end_step = (d + 1) * args.rounds_per_block * spr
        run_max, t_forget = -1.0, None
        for s, acc in series.items():
            run_max = max(run_max, acc)
            if s > end_step and acc < run_max - args.tau:
                t_forget = s
                break
        if t_forget is not None:
            events[d] = (end_step, t_forget)

    print(f"\nstep-level forgetting events (tau = {args.tau}):")
    for d, (end, tf) in events.items():
        print(f"  domain {d}: trained to step {end}, "
              f"t_forget at step {tf} "
              f"(round {tf // spr + 1}, {tf - end} steps after)")
    if not events:
        raise SystemExit("no step-level events found")

    # ── convert round-level firings to steps ────────────────────────
    if not args.horizon or not os.path.exists(args.horizon):
        print("\nNo --horizon table supplied; step-level events reported "
              "above. Pass horizon_table.csv to score signals.")
        return

    fired = defaultdict(dict)       # signal -> event key -> round
    with open(args.horizon) as f:
        for row in csv.DictReader(f):
            if row["detected"] != "1":
                continue
            try:
                fired[row["signal"]][row["event"]] = int(float(row["t_signal"]))
            except (TypeError, ValueError):
                continue

    rows_out = []
    print(f"\n{'signal':<20}{'when':>6}{'mean dt (steps)':>18}"
          f"{'mean dt (rounds)':>19}")
    print("-" * 63)
    summary = []
    for sig, per_event in fired.items():
        when = "pre" if sig in PRE_SIGNALS else "post"
        deltas = []
        for d, (end, tf) in events.items():
            r = per_event.get(str(d))
            if r is None:
                continue
            # a pre-training signal is computed before any update of its
            # round; a post-training signal after the last one
            fire_step = (r - 1) * spr if when == "pre" else r * spr
            if fire_step <= tf:
                deltas.append(tf - fire_step)
                rows_out.append([sig, when, d, r, fire_step, tf,
                                 tf - fire_step])
        if deltas:
            m = sum(deltas) / len(deltas)
            summary.append((sig, when, m))
            print(f"{sig:<20}{when:>6}{m:>18.0f}{m / spr:>19.2f}")

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["signal", "when", "domain", "fire_round",
                        "fire_step", "t_forget_step", "delta_t_steps"])
            w.writerows(rows_out)
        print(f"\nwritten to {args.out}")

    pre = [m for _, w, m in summary if w == "pre"]
    post = [m for _, w, m in summary if w == "post"]
    if pre and post:
        print(f"\npre-training signals:  mean {sum(pre)/len(pre):.0f} steps "
              f"of warning")
        print(f"post-training signals: mean {sum(post)/len(post):.0f} steps")
        print("A positive gap here is the warning that round-level Delta t "
              "cannot express.")


if __name__ == "__main__":
    main()
