#!/bin/bash
#$ -S bin/bash
#$ -j y
#$ -cwd
#$ -M sw424@drexel.edu
#$ -l h_rt=04:00:00
#$ -P rosenPrj
#$ -l ua=haswell
#$ -pe shm 8
#$ -l mem_free=6G
#$ -l h_vmem=8G
##$ -q long.q

. /etc/profile.d/modules.sh
module load shared
module load proteus
module load sge/univa
module load gcc/4.8.1
module load python/intelpython/3.5.3

k=4
USERNAME=sw424

SCRATCH=/scratch/$USERNAME/sweep_test
SCRATCH_OUT=$SCRATCH/out
SCRATCH_DATA=$SCRATCH/data
SCRATCH_KMERS=$SCRATCH_DATA/kmers_$k

mkdir -p $SCRATCH_OUT
mkdir -p $SCRATCH_KMERS

CODE=/home/$USERNAME/earth/code
KMERS=/home/$USERNAME/earth/data/kmers_$k
OUT=/home/$USERNAME/earth/out

py=/mnt/HA/opt/python/intel/2017/intelpython3/bin/python3.5
script=8_sweep.py
labels=/home/$USERNAME/earth/data/labels.csv

cp $CODE/$script $SCRATCH/
cp $labels $SCRATCH_DATA/
cp -r $KMERS/* $SCRATCH_KMERS/

cd $SCRATCH

$py $script

#cp $SCRATCH_OUT/* $OUT/

exit 0
