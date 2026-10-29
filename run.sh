#!/bin/bash
#SBATCH --job-name=fl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=25242748@students.lincoln.ac.uk
#SBATCH --time=16:00:00
#SBATCH --output=logs/slurm_%j.log
#SBATCH --error=logs/slurm_%j.err
#SBATCH --nodelist=hpc-novel-gpu[01-06]
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --container-image=/home/users/mdevine/FL/itsdinok+fl-experiment+latest.sqsh
#SBATCH --container-mounts=/home/users/mdevine/FL/class_migration:/workspace:rw
#SBATCH --container-workdir=/workspace

# One experiment per submission. Everything is passed through, so no file
# needs editing between ablations:
#
#   mkdir -p logs
#   sbatch run.sh --dataset cifar100 --scenario split
#   sbatch run.sh --dataset pacs     --scenario domain --port 8081
#   sbatch run.sh --dataset cifar100 --scenario dirichlet --alpha 0.05 \
#                 --tag a005 --port 8082
#
# Slurm opens the log files before this script runs, so logs/ must exist.

ARGS="$@"
echo "Experiment: $ARGS"

srun bash -s <<EOF
cd /workspace
mkdir -p logs
export HOME=/workspace
export XDG_CACHE_HOME=/workspace/.cache
export TORCH_HOME=/workspace/.cache
export GRPC_VERBOSITY=ERROR

python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# derive client count and log directory from the same config the
# processes will use, so nothing can drift apart
read NC LOGDIR <<< \$(python - <<'PY'
import argparse, config
ap = argparse.ArgumentParser(); config.add_arguments(ap)
a, _ = ap.parse_known_args("""$ARGS""".split())
print(a.clients, config.logs_dir(a))
PY
)
echo "clients=\$NC logs=\$LOGDIR"
mkdir -p "\$LOGDIR"

# fetch data once, so clients cannot race on the archive
python datasets.py --download \$(python - <<'PY'
import argparse, config
ap = argparse.ArgumentParser(); config.add_arguments(ap)
a, _ = ap.parse_known_args("""$ARGS""".split())
print(a.dataset)
PY
)

python server.py $ARGS > "\$LOGDIR/server_\${SLURM_JOB_ID}.log" 2>&1 &
sleep 30

NGPU=\$(nvidia-smi -L 2>/dev/null | wc -l); [ "\$NGPU" -lt 1 ] && NGPU=1
for i in \$(seq 0 \$((NC - 1))); do
    CUDA_VISIBLE_DEVICES=\$((i % NGPU)) \
        python client.py --cid \$i $ARGS \
        > "\$LOGDIR/client\${i}_\${SLURM_JOB_ID}.log" 2>&1 &
done
wait
EOF
