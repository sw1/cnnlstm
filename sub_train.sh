#!/bin/bash
#$ -S bin/bash
#$ -j y
#$ -cwd
#$ -M sw424@drexel.edu
#$ -l h_rt=48:00:00
#$ -P rosenPrj
#$ -pe shm 16
#$ -l mem_free=3G
#$ -l h_vmem=4G
#$ -q all.q@@intelhosts

. /etc/profile.d/modules.sh
module load shared
module load proteus
module load sge/univa
module load gcc/4.8.1
module load python/intelpython/3.5.3

k=4
class=lake
USERNAME=sw424

SCRATCH=/scratch/$USERNAME/train_$class
SCRATCH_OUT=$SCRATCH/out
SCRATCH_DATA=$SCRATCH/data
SCRATCH_KMERS=$SCRATCH_DATA/kmers_$k

mkdir -p $SCRATCH_OUT
mkdir -p $SCRATCH_KMERS
mkdir -p $SCRATCH/logs_k$k

CODE=/home/$USERNAME/earth/code
KMERS=/home/$USERNAME/earth/data/kmers_$k
OUT=/home/$USERNAME/earth/out

py=/mnt/HA/opt/python/intel/2017/intelpython3/bin/python3.5
script=7_model.py
labels=/home/$USERNAME/earth/data/labels.csv

cp -r $KMERS/* $SCRATCH_KMERS/
cp $CODE/$script $SCRATCH/
cp $labels $SCRATCH_DATA/

cd $SCRATCH

$py $script $k $class

cp $SCRATCH_OUT/* $OUT/

exit 0
