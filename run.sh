#!/bin/bash

#SBATCH -J ddoppala_AimaNet_WithBackground
#SBATCH -p general
#SBATCH -o aimanet_wb_out_%j.txt
#SBATCH -e aimanet_wb_error_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ddoppala@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=05:00:00
#SBATCH -A c00041

#Load any modules that your program needs
module load deeplearning/2.10.0

#Run your program
srun python3 mainSeperate78.py --data_path ./UBFC-wb-separate --epochs 8 --batch_size 6
