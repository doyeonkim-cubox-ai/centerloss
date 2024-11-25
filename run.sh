#!/bin/bash

#SBATCH --job-name=facedetection
#SBATCH --output="./logs/%A"
#SBATCH --nodelist=nv176
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G

##### Number of total processes
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Nodelist:= " $SLURM_JOB_NODELIST
echo "Number of nodes:= " $SLURM_JOB_NUM_NODES
echo "Ntasks per node:= "  $SLURM_NTASKS_PER_NODE
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "


echo "Run started at:- "
date

# ex) srun python -m mnist_resnet50.train

srun jupyter notebook --ip 0.0.0.0
#srun python -m centerloss.train
#srun python -m centerloss.test
#srun python -m centerloss.inference -img "./images/image4.png"
#srun python -m FaceDetection.inference
