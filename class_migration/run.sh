#!/bin/bash
#SBATCH --job-name=fl_dil
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=25242748@students.lincoln.ac.uk
#SBATCH --time=16:00:00
#SBATCH --output=logs_dil/fl_dil_%j.log
#SBATCH --error=logs_dil/fl_dil_%j.err
#SBATCH --nodelist=hpc-novel-gpu[01-06]
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --container-image=/home/users/mdevine/FL/itsdinok+fl-experiment+latest.sqsh
#SBATCH --container-mounts=/home/users/mdevine/FL:/workspace:rw
#SBATCH --container-workdir=/workspace

# NOTE: Slurm opens the --output/--error files BEFORE this script runs, so
# the logs_dil/ directory must already exist when you call sbatch:
#     mkdir -p logs_dil && sbatch run_dil.sh

echo "Starting domain-incremental CIFAR-100 signal benchmark..."
srun bash -s <<'EOF'
cd /workspace
mkdir -p logs_dil
export HOME=/workspace
export XDG_CACHE_HOME=/workspace/.cache
export TORCH_HOME=/workspace/.cache
export GRPC_VERBOSITY=ERROR

# client count comes from client_dil.py so the two never drift apart
NUM_CLIENTS=$(sed -n 's/^NUM_CLIENTS *= *\([0-9]*\).*/\1/p' client_dil.py)
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
[ "$NGPU" -lt 1 ] && NGPU=1
echo "clients=$NUM_CLIENTS gpus=$NGPU"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

python server_dil.py > logs_dil/server_${SLURM_JOB_ID}.log 2>&1 &
sleep 30
for i in $(seq 0 $((NUM_CLIENTS - 1))); do
    CUDA_VISIBLE_DEVICES=$((i % NGPU)) \
        python client_dil.py --cid $i \
        > logs_dil/client${i}_${SLURM_JOB_ID}.log 2>&1 &
done
wait
EOF
