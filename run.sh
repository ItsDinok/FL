#!/bin/bash
#SBATCH --job-name=fl_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=25242748@students.lincoln.ac.uk
#SBATCH --time=08:00:00
#SBATCH --output=logs/fl_train_%j.log
#SBATCH --error=logs/fl_train_%j.err
#SBATCH --nodelist=hpc-novel-gpu[01-06]
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Enroot container configurations
#SBATCH --container-image=/home/users/mdevine/itsdinok+fl-experiment+latest.sqsh
#SBATCH --container-mounts=/home/shared:/home/shared:ro,/home/users/mdevine/workspace:/workspace:rw
#SBATCH --container-workdir=/workspace

module purge
mkdir -p logs

echo "Starting FL job in $PWD..."

srun bash -s <<'EOF'
cd /workspace
mkdir -p logs

export HOME=/workspace
export XDG_CACHE_HOME=/workspace/.cache
export TORCH_HOME=/workspace/.cache

python server.py > logs/server_${SLURM_JOB_ID}.log 2>&1 &
sleep 5

python client.py --cid 0 > logs/client0_${SLURM_JOB_ID}.log 2>&1 &
python client.py --cid 1 > logs/client1_${SLURM_JOB_ID}.log 2>&1 &
python client.py --cid 2 > logs/client2_${SLURM_JOB_ID}.log 2>&1 &

wait
EOF

