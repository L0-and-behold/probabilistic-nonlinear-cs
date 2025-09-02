#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=165536
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=user@mail.com

#SBATCH --output=systematic_comparison_script_%j.out
#SBATCH --error=systematic_comparison_script_%j.err
#SBATCH --job-name=systematic_comparison

module load julia

cd ~/absolute/path/to/the/folder/containing/this/file
julia systematic-compressive-sensing-experiments.jl