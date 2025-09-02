#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=165536
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=user@mail.com

#SBATCH --output=time_benchmarks_julia_script_%j.out
#SBATCH --error=time_benchmarks_julia_script_%j.err
#SBATCH --job-name=time_benchmarks

module load julia

cd ~/absolute/path/to/the/folder/containing/this/file
julia runtime-benchmarks.jl