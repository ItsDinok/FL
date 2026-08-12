"""
horizon.py — prediction-horizon (delta-t) analysis for the intervention-
signal benchmark. Run OFFLINE on the CSVs produced by a benchmark run.

Definitions:
    t_signal : first round where a signal exceeds its detection threshold
    t_forget : first round where a retained task's accuracy falls more
               than FORGET_DROP below its running maximum
    delta_t  = t_forget - t_signal          (large positive = early warning)

Detection thresholds are z-scores against a per-signal baseline estimated
from the calibration rounds (rounds within the first task, where no
forgetting is possible). Signals have a direction:
    'up'   : detection when value > mean + Z_K * std      (e.g. new_loss)
    'down' : detection when value < mean - Z_K * std      (e.g. cka_final)
    'abs'  : detection when |z| > Z_K                     (e.g. grad_cos)

Outputs:
    horizon_table.csv  one row per (signal, forgetting event):
                       t_signal, t_forget, delta_t, detected, false alarms
    summary printed to stdout: per-signal mean delta_t, detection rate,
                       false-alarm count, mean per-round cost (ms)

Usage:
    python horizon.py --logs logs/ --rounds-per-task 5 --num-tasks 4
"""

import argparse
import csv
import glob
import math
import os
from collections import defaultdict

FORGET_DROP = 0.05     # t_forget: acc falls 5pp below its running max
Z_K = 3.0              # detection threshold in baseline std units

# signal -> detection direction
SIGNAL_DIRECTIONS = {
    "new_loss": "up", "new_entropy": "up",
    "grad_norm_new": "up", "grad_norm_old": "up",
    "grad_cos": "abs", "grad_inner": "abs", "pred_dL_old": "abs",
    "ref_loss": "up", "ref_acc": "down",
    "weight_div_round": "up", "weight_div_task": "up",
    "proto_drift": "up", "centroid_drift": "up",
    "cka_block3": "down", "cka_block7": "down",
    "cka_block11": "down", "cka_final": "down",
}
COST_COLUMNS = {"t_grad_ms": ["new_loss", "new_entropy", "grad_norm_new",
                              "grad_norm_old", "grad_cos", "grad_inner",
                              "pred_dL_old"],
                "t_perf_ms": ["ref_loss", "ref_acc"],
                "t_weight_ms": ["weight_div_round", "weight_div_task"],
                "t_repr_ms": ["proto_drift", "centroid_drift", "cka_block3",
                              "cka_block7", "cka_block11", "cka_final"]}


def read_csvs(pattern):
    """Read all matching CSVs; return {cid: [row dicts]} with float coercion."""
    out = {}
    for path in sorted(glob.glob(pattern)):
        cid = "".join(ch for ch in os.path.basename(path) if ch.isdigit())
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


def fleet_series(store, col):
    """Average a column across clients -> {round: value}."""
    acc = defaultdict(list)
    for rows in store.values():
        for r in rows:
            v = r.get(col, math.nan)
            if not math.isnan(v):
                acc[int(r["round"])].append(v)
    return {rd: sum(vs) / len(vs) for rd, vs in sorted(acc.items())}


def find_t_forget(metrics, num_tasks, rounds_per_task):
    """Per task: first round its retained accuracy drops FORGET_DROP below
    its running max, restricted to rounds AFTER the task finished training."""
    events = {}
    for t in range(num_tasks):
        series = fleet_series(metrics, f"acc_task{t}")
        task_end = (t + 1) * rounds_per_task
        run_max, t_forget = -1.0, None
        for rd, acc in series.items():
            run_max = max(run_max, acc)
            if rd > task_end and acc < run_max - FORGET_DROP:
                t_forget = rd
                break
        if t_forget is not None:
            events[t] = t_forget
    return events


def find_class_events(metrics, logs_dir, rounds_per_task):
    """Class-migration mode: per-CLASS forgetting events.

    A class's effective 'task end' is the last round of the last block in
    which ANY client allocated it more than 1/K of its Dirichlet
    distribution (read from distribution_client*.csv). t_forget is then
    the first later round where fleet per-class accuracy drops FORGET_DROP
    below its running max. Spontaneous recovery can produce multiple
    events per class; each is scored independently.
    Returns {(class_name, event_idx): (effective_end_round, t_forget)}.
    """
    # class columns from the metrics CSVs
    first = next(iter(metrics.values()))
    class_cols = [k for k in first[0] if k.startswith("acc_")
                  and k != "acc_mean" and not k.startswith("acc_task")]
    if not class_cols:
        raise SystemExit("--mode class needs per-class acc_* columns")
    K = len(class_cols)

    # distribution files: block -> {class: max proportion across clients}
    dist = defaultdict(lambda: defaultdict(float))
    for path in glob.glob(os.path.join(logs_dir, "distribution_client*.csv")):
        with open(path) as f:
            for row in csv.DictReader(f):
                b = int(float(row["block"]))
                for k, v in row.items():
                    if k.startswith("p_"):
                        cls = k[2:]
                        dist[b][cls] = max(dist[b][cls], float(v))
    if not dist:
        raise SystemExit("--mode class needs distribution_client*.csv files")
    num_blocks = max(dist) + 1

    events = {}
    for col in class_cols:
        cls = col[len("acc_"):]
        # last block where any client meaningfully trained this class
        active = [b for b in range(num_blocks)
                  if dist[b].get(cls, 0.0) > 1.0 / K]
        if not active:
            continue
        eff_end = (max(active) + 1) * rounds_per_task
        series = fleet_series(metrics, col)
        run_max, idx, in_event = -1.0, 0, False
        for rd, acc in series.items():
            run_max = max(run_max, acc)
            dropped = rd > eff_end and acc < run_max - FORGET_DROP
            if dropped and not in_event:
                events[(cls, idx)] = (eff_end, rd)
                idx += 1
                in_event = True
            elif not dropped:
                in_event = False
                if acc >= run_max - FORGET_DROP / 2:
                    run_max = acc   # reset baseline after recovery
    return events


def find_t_signal(series, baseline_rounds, direction, after_round,
                  z_k=None):
    """First round > after_round where the signal crosses its threshold.

    The baseline uses median/MAD rather than mean/std: calibration rounds
    can contain a monotone warm-up transient (e.g. new_loss decaying from
    its untrained value), and mean/std would inflate the threshold enough
    to mask genuine later spikes. z = (v - median) / (1.4826 * MAD)."""
    if z_k is None:
        z_k = Z_K
    base = sorted(v for rd, v in series.items()
                  if rd in baseline_rounds and not math.isnan(v))
    if len(base) < 2:
        return None, []
    n = len(base)
    med = (base[n // 2] if n % 2 else (base[n // 2 - 1] + base[n // 2]) / 2)
    devs = sorted(abs(v - med) for v in base)
    mad = (devs[n // 2] if n % 2 else (devs[n // 2 - 1] + devs[n // 2]) / 2)
    scale = 1.4826 * mad
    if scale == 0:
        scale = abs(med) * 0.05 + 1e-9

    def crossed(v):
        z = (v - med) / scale
        if direction == "up":
            return z > z_k
        if direction == "down":
            return z < -z_k
        return abs(z) > z_k

    crossings = [rd for rd, v in sorted(series.items())
                 if not math.isnan(v) and rd not in baseline_rounds
                 and crossed(v)]
    t_signal = next((rd for rd in crossings if rd > after_round), None)
    return t_signal, crossings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--rounds-per-task", type=int, default=5)
    ap.add_argument("--num-tasks", type=int, default=4)
    ap.add_argument("--mode", choices=["task", "class"], default="task",
                    help="task: Split CIFAR-100 / DIL (acc_task* columns); "
                         "class: class-migration (per-class acc_* columns "
                         "+ distribution_client*.csv)")
    ap.add_argument("--out", default="horizon_table.csv")
    ap.add_argument("--sweep", type=float, nargs="*", default=None,
                    help="z-score thresholds to sweep, e.g. "
                         "--sweep 1.5 2 2.5 3 4 5 (bare --sweep uses that "
                         "default list); writes <out>_sweep.csv and prints "
                         "a detection-vs-earliness frontier per signal")
    args = ap.parse_args()
    if args.sweep is not None and len(args.sweep) == 0:
        args.sweep = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

    metrics = read_csvs(os.path.join(args.logs, "metrics_client*.csv"))
    signals = read_csvs(os.path.join(args.logs, "signals_client*.csv"))
    if not metrics or not signals:
        raise SystemExit("No metrics/signals CSVs found in --logs directory")

    rpt = args.rounds_per_task
    # calibration window: first task minus round 1 (untrained model) and
    # round 2 (warm-up transient) — both would otherwise contaminate the
    # baseline of decaying signals like new_loss
    baseline_rounds = set(range(3, rpt + 1))

    # ground-truth forgetting events -> {event_key: (window_open, t_forget)}
    if args.mode == "task":
        raw = find_t_forget(metrics, args.num_tasks, rpt)
        events = {t: ((t + 1) * rpt, tf) for t, tf in raw.items()}
        print("Forgetting events (t_forget per task):")
        for t, (_, tf) in events.items():
            print(f"  task {t}: t_forget = round {tf}")
    else:
        events = find_class_events(metrics, args.logs, rpt)
        print("Forgetting events (per class, effective-end -> t_forget):")
        for (cls, i), (eff, tf) in sorted(events.items()):
            print(f"  {cls} [event {i}]: eff_end r{eff} -> t_forget r{tf}")
    if not events:
        raise SystemExit("No forgetting events found — nothing to score")

    # per-signal analysis at the default threshold
    table, summary = score_signals(signals, events, baseline_rounds, rpt,
                                   z_k=Z_K, collect_table=True)

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["signal", "event", "t_signal", "t_forget",
                    "delta_t", "detected"])
        w.writerows(table)

    print(f"\nPer-event table written to {args.out}\n")
    print(f"{'signal':<18}{'mean Δt':>9}{'med Δt':>8}{'Δt range':>10}"
          f"{'det rate':>10}{'FA':>5}{'cost (ms)':>11}")
    print("-" * 71)
    for s in sorted(summary,
                    key=lambda x: (-(x['mean_dt'] if x['mean_dt'] == x['mean_dt']
                                     else -99), )):
        dt = f"{s['mean_dt']:.1f}" if s["mean_dt"] == s["mean_dt"] else "—"
        md = f"{s['median_dt']:.0f}" if s["median_dt"] == s["median_dt"] else "—"
        rng = (f"{s['min_dt']:.0f}–{s['max_dt']:.0f}"
               if s["min_dt"] == s["min_dt"] else "—")
        cost = f"{s['cost_ms']:.0f}" if s["cost_ms"] == s["cost_ms"] else "—"
        print(f"{s['signal']:<18}{dt:>9}{md:>8}{rng:>10}"
              f"{s['det_rate']:>10.0%}{s['false_alarms']:>5}{cost:>11}")
    print("\nNote: cost (ms) is the wall-clock of the signal's GROUP "
          "(grad/perf/weight/repr), shared across signals in that group.")

    # threshold sweep: detection-rate vs earliness frontier per signal
    if args.sweep:
        sweep_rows = []
        for z_k in args.sweep:
            _, summ = score_signals(signals, events, baseline_rounds, rpt,
                                    z_k=z_k, collect_table=False)
            for s in summ:
                sweep_rows.append([f"{z_k:g}", s["signal"],
                                   f"{s['det_rate']:.3f}",
                                   (f"{s['mean_dt']:.2f}"
                                    if s["mean_dt"] == s["mean_dt"] else ""),
                                   (f"{s['median_dt']:.1f}"
                                    if s["median_dt"] == s["median_dt"] else ""),
                                   s["false_alarms"]])
        sweep_path = os.path.splitext(args.out)[0] + "_sweep.csv"
        with open(sweep_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["z_k", "signal", "det_rate", "mean_dt",
                        "median_dt", "false_alarms"])
            w.writerows(sweep_rows)
        print(f"\nThreshold sweep written to {sweep_path}")

        # compact frontier: det_rate @ each z_k per signal (mean Δt in parens)
        print(f"\n{'signal':<18}" + "".join(f"{('z=%g' % z):>16}"
                                            for z in args.sweep))
        print("-" * (18 + 16 * len(args.sweep)))
        by_sig = defaultdict(dict)
        for z, sig, dr, mdt, _, fa in sweep_rows:
            by_sig[sig][z] = (float(dr), mdt, fa)
        for sig in SIGNAL_DIRECTIONS:
            if sig not in by_sig:
                continue
            cells = []
            for z in args.sweep:
                dr, mdt, fa = by_sig[sig][f"{z:g}"]
                dt_s = mdt if mdt else "—"
                cells.append(f"{dr:>4.0%} Δ{dt_s:>4} F{fa:<2}")
            print(f"{sig:<18}" + "".join(f"{c:>16}" for c in cells))
        print("\nCells: detection rate, mean Δt, F=false alarms. "
              "Lower z = more sensitive.")


def score_signals(signals, events, baseline_rounds, rpt, z_k,
                  collect_table):
    """Score every signal against the event set at threshold z_k.
    Returns (per_event_table, summary_rows)."""
    table, summary = [], []
    for sig, direction in SIGNAL_DIRECTIONS.items():
        series = fleet_series(signals, sig)
        if not series:
            continue
        deltas, detected = [], 0
        all_crossings = set()
        for key, (window_open, t_forget) in events.items():
            # forgetting can only begin once the task/class stops being
            # trained, so the detection window opens at that round
            t_signal, crossings = find_t_signal(
                series, baseline_rounds, direction,
                after_round=window_open, z_k=z_k)
            all_crossings.update(crossings)
            if t_signal is not None and t_signal <= t_forget:
                deltas.append(t_forget - t_signal)
                detected += 1
                if collect_table:
                    table.append([sig, key, t_signal, t_forget,
                                  t_forget - t_signal, 1])
            elif collect_table:
                table.append([sig, key, t_signal or "", t_forget, "", 0])

        # false alarms: crossings in rounds with no event within the next
        # task-length window
        fa = sum(1 for rd in all_crossings
                 if not any(rd <= tf <= rd + rpt
                            for _, tf in events.values()))

        cost_col = next((c for c, sigs in COST_COLUMNS.items()
                         if sig in sigs), None)
        cost_series = fleet_series(signals, cost_col) if cost_col else {}
        mean_cost = (sum(cost_series.values()) / len(cost_series)
                     if cost_series else math.nan)

        if deltas:
            sd = sorted(deltas)
            n = len(sd)
            median_dt = (sd[n // 2] if n % 2
                         else (sd[n // 2 - 1] + sd[n // 2]) / 2)
            stats = dict(mean_dt=sum(deltas) / n, median_dt=median_dt,
                         min_dt=sd[0], max_dt=sd[-1])
        else:
            stats = dict(mean_dt=math.nan, median_dt=math.nan,
                         min_dt=math.nan, max_dt=math.nan)

        summary.append({"signal": sig, **stats,
                        "det_rate": detected / len(events),
                        "false_alarms": fa, "cost_ms": mean_cost})
    return table, summary


if __name__ == "__main__":
    main()
