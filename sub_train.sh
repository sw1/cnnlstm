#!/bin/bash
#$ -S bin/bash
#$ -j y
#$ -cwd
#$ -M sw424@drexel.edu
#$ -l h_rt=48:00:00
#$ -l ua=haswell
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
#module load python/intelpython/3.6.3

k=4
class=lake
USERNAME=sw424
coding=1

SCRATCH=/scratch/$USERNAME/train_${class}_${coding}_stateful_128b_clip
SCRATCH_OUT=$SCRATCH/out
SCRATCH_DATA=$SCRATCH/data
SCRATCH_KMERS=$SCRATCH_DATA/kmers_$k

rm -rf $SCRATCH

mkdir -p $SCRATCH_OUT
mkdir -p $SCRATCH_KMERS
mkdir -p $SCRATCH/logs_k$k

CODE=/home/$USERNAME/earth/code
KMERS=/home/$USERNAME/earth/data/kmers_$k
OUT=/home/$USERNAME/earth/out

py=/mnt/HA/opt/python/intel/2017/intelpython3/bin/python3.5
#py=/mnt/HA/opt/python/intel/2018.1.023/intelpython3/bin/python3
script=7_model_stateful.py
labels=/home/$USERNAME/earth/data/labels.csv

cp -r $KMERS/* $SCRATCH_KMERS/
cp $CODE/$script $SCRATCH/
cp $labels $SCRATCH_DATA/

cd $SCRATCH

$py $script $k $class $coding

cp $SCRATCH_OUT/* $OUT/

exit 0
