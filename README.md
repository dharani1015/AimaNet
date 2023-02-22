# AimaNet
Built an open source PyTorch reimplementation of the state-of-theart MTTS-CAN algorithm. Trained and tested using UBFC-Phys dataset. Explored the effect of including and excluding background information on the model. Optimized the model for mobile deployment.

## To run the model on Carbonate machine.

1. Clone repo

```
git clone https://github.iu.edu/ddoppala/AimaNet.git
```

2. Move into AimaNet folder. Copy input and output file. File should be named as input_numpy.npy and output_numpy.npy

3. On terminal, run the following commands to load deep learning libraries.

```
module load deeplearning/2.10.0

```

4. To get terminal to run our model training , execute below command. This creates interactive job

```
srun -p gpu -A c00041 --gpus-per-node v100:1 --pty bash
```

5. Training on single all video processed numpy input and output file-

a) To run training on a single Run below command to start training.

```
python3 main.py --epochs 2 --batch_size 3 --train_split_index 132720
```

b) To run training on seperate 78 files (63 training and 15 test files)-

```
python3 mainSeperate78.py --data_path ./UBFC-wb-separate --epochs 24 --batch_size 6
```

6. To submit SLURM job on the carbonate machine-

```
sbatch run.sh
```

## References

Apart from the references mentioned in the report, the following open source implementation was referred to for completing this project:
- https://github.com/xliucs/MTTS-CAN
