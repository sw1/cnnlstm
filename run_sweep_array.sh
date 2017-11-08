#!/bin/bash

k=4
USERNAME=sw424

SCRATCH=/scratch/$USERNAME/sweep
SCRATCH_OUT=$SCRATCH/out
SCRATCH_DATA=$SCRATCH/data
SCRATCH_KMERS=$SCRATCH_DATA/kmers_$k

mkdir -p $SCRATCH_OUT
mkdir -p $SCRATCH_KMERS

WORK=/home/$USERNAME/earth
CODE=$WORK/code
KMERS=$WORK/data/kmers_$k
OUT=$WORK/out

py=/mnt/HA/opt/python/intel/2017/intelpython3/bin/python3.5
script=8_sweep.py
labels=$WORK/data/labels.csv

cp $CODE/$script $SCRATCH/
cp $labels $SCRATCH_DATA/
cp -r $KMERS/* $SCRATCH_KMERS/

$py $CODE/params.py
mv params.csv $SCRATCH/

qsub sub_sweep_array.sh $k

exit 0
