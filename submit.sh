#!/usr/bin/env bash
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --job-name=CAPSNET
#SBATCH --mail-type ALL
#SBATCH --mail-user jos.vandewolfshaar@gmail.com
#SBATCH --output CAPSNET-%j.log
#SBATCH --mem=16000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load tensorflow/1.3.1-foss-2016a-Python-3.5.2-CUDA-8.0.61
source envs/lws/bin/activate

python capsnet/capsule.py --logs /home/s2098407/capsnet/logs/$1.csv --datadir /data/s2098407/MNIST_data --no_pbar --pclayer $1