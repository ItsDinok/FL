#!/bin/bash
#SBATCH --job-name=fl_fetch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/fetch_%j.log
#SBATCH --error=logs/fetch_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --container-image=/home/users/mdevine/FL/itsdinok+fl-experiment+latest.sqsh
#SBATCH --container-mounts=/home/users/mdevine/FL/class_migration:/workspace:rw
#SBATCH --container-workdir=/workspace
#
# Downloads and prepares every reachable dataset, then reports which ones
# need fetching by other means.
#
#   mkdir -p logs && sbatch fetch_data.sh
#
# The container has no wget or curl, so everything goes through Python's
# urllib. Compute nodes also reach only some hosts; anything unreachable
# is reported rather than retried forever.

set -u
echo "Fetching datasets..."

srun bash -s <<'EOF'
cd /workspace
mkdir -p logs
export HOME=/workspace
export XDG_CACHE_HOME=/workspace/.cache
export TORCH_HOME=/workspace/.cache

cd ~/FL/class_migration
python3 -c "
import medmnist, os
print('default root:', medmnist.dataset.DEFAULT_ROOT)
print('expected file: pathmnist.npz')
"
ls -lh data/pathmnist.npz

python3 -c "
from medmnist import PathMNIST
ds = PathMNIST(split='train', root='data', download=False, as_rgb=True)
print(len(ds), 'images, labels shape', ds.labels.shape)
"

python fetch_data.py
EOF

echo "Done."
