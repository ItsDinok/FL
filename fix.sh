#!/usr/bin/env python3
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


python dist.py --logs logs/pathmnist_dirichlet_path_diri \
    --dataset pathmnist --alpha 0.5 --clients 3 --blocks 4

python horizon.py --logs logs/pathmnist_dirichlet_path_diri \
    --mode class --sweep --tau-sweep \
    --out logs/pathmnist_dirichlet_path_diri/horizon.csv

python horizon.py --logs logs/pathmnist_split_path_cil \
    --mode task --sweep --tau-sweep \
    --out logs/pathmnist_split_path_cil/horizon.csv
