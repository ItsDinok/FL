#!/bin/bash
#SBATCH --job-name=fl_analyse
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/analyse_%j.log
#SBATCH --error=logs/analyse_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --container-image=/home/users/mdevine/FL/itsdinok+fl-experiment+latest.sqsh
#SBATCH --container-mounts=/home/users/mdevine/FL/class_migration:/workspace:rw
#SBATCH --container-workdir=/workspace
#
# Scores one or more completed runs. Works out the right --mode from the
# CSV columns, regenerates the Dirichlet distribution file if it is
# missing, and runs the step-level analysis when intra-round probing was
# enabled.
#
#   sbatch analyse.sh                                  # every run in logs/
#   sbatch analyse.sh logs/pathmnist_split_path_cil    # just these
#
# Nothing here needs a GPU; drop --gres and switch partition if your
# cluster has a CPU queue.

set -u
echo "Analysing: ${@:-all runs in logs/}"

srun bash -s <<EOF
cd /workspace
export HOME=/workspace
export XDG_CACHE_HOME=/workspace/.cache
export PYTHONUNBUFFERED=1

TARGETS="${@:-}"
if [ -z "\$TARGETS" ]; then
    TARGETS=\$(find logs -maxdepth 1 -mindepth 1 -type d \
              -exec test -e '{}/metrics_client0.csv' ';' -print | sort)
fi

for RUN in \$TARGETS; do
    RUN=\${RUN%/}
    echo
    echo "============================================================"
    echo "  \$RUN"
    echo "============================================================"

    if [ ! -f "\$RUN/metrics_client0.csv" ]; then
        echo "  no metrics_client0.csv — skipping"
        continue
    fi

    # completeness: a client that died leaves a short CSV and skews the
    # fleet averages, so say so before scoring anything
    echo "--- completeness ---"
    wc -l "\$RUN"/metrics_client*.csv | head -n -1
    ERRS=\$(grep -lc "Traceback" "\$RUN"/client*.log 2>/dev/null | wc -l)
    [ "\$ERRS" -gt 0 ] && echo "  WARNING: \$ERRS client log(s) contain tracebacks"

    # mode comes from the column names: acc_task* means task mode
    HEADER=\$(head -1 "\$RUN/metrics_client0.csv")
    case "\$HEADER" in
        *acc_task0*) MODE=task ;;
        *)           MODE=class ;;
    esac
    echo "--- mode: \$MODE ---"

    # class mode needs the per-block proportions; regenerate if absent
    if [ "\$MODE" = "class" ] && [ ! -f "\$RUN/distribution_client0.csv" ]; then
        DS=\$(basename "\$RUN" | cut -d_ -f1)
        NCL=\$(ls "\$RUN"/metrics_client*.csv | wc -l)
        ALPHA=\$(grep -m1 -o 'alpha=[0-9.]*' "\$RUN"/client0_*.log 2>/dev/null \
                 | head -1 | cut -d= -f2)
        ALPHA=\${ALPHA:-0.5}
        echo "  regenerating distributions (dataset=\$DS alpha=\$ALPHA clients=\$NCL)"
        python write_distributions.py --logs "\$RUN" --dataset "\$DS" \
            --alpha "\$ALPHA" --clients "\$NCL" --blocks 4 || true
    fi

    echo "--- round-level scoring ---"
    python horizon.py --logs "\$RUN" --mode "\$MODE" --sweep --tau-sweep \
        --out "\$RUN/horizon.csv" || echo "  horizon.py failed"

    echo
    echo "--- gradient behaviour at transitions ---"
    python - "\$RUN" <<'PY'
import csv, sys, os
run = sys.argv[1]
path = os.path.join(run, "signals_client0.csv")
if not os.path.exists(path):
    print("  no signals_client0.csv"); raise SystemExit
rows = list(csv.DictReader(open(path)))
rpb = 5
marks = set()
for b in range(1, 6):
    marks |= {b * rpb, b * rpb + 1}
print(f"  {'round':>6} {'type':<12} {'cosine':>10} {'inner':>14} {'forecast':>14}")
for r in rows:
    rd = int(float(r["round"]))
    if rd not in marks:
        continue
    kind = "TRANSITION" if (rd - 1) % rpb == 0 and rd > 1 else "within"
    def g(k):
        try:    return float(r[k])
        except (KeyError, ValueError): return float("nan")
    print(f"  {rd:>6} {kind:<12} {g('grad_cos'):>+10.4f} "
          f"{g('grad_inner'):>+14.6f} {g('pred_dL_old'):>+14.6f}")
PY

    # step-level horizon, when intra-round probing was enabled
    if [ -f "\$RUN/steps_client0.csv" ]; then
        echo
        echo "--- step-level horizon ---"
        SCRIPT=analyse_steps.py
        [ -f "\$SCRIPT" ] || SCRIPT=analyse.py
        python "\$SCRIPT" --logs "\$RUN" --rounds-per-block 5 \
            --horizon "\$RUN/horizon.csv" \
            --out "\$RUN/steps_horizon.csv" || echo "  step analysis failed"
    fi
done

echo
echo "============================================================"
echo "Outputs written per run: horizon.csv, horizon_sweep.csv,"
echo "horizon_tau.csv, and steps_horizon.csv where applicable."
EOF

echo "Done."
