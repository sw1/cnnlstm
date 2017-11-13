#!/bin/bash
#$ -S bin/bash
#$ -j y
#$ -cwd
#$ -M sw424@drexel.edu
#$ -l h_rt=48:00:00
#$ -P rosenPrj
#$ -l ua=haswell
#$ -pe shm 6
#$ -l mem_free=10G
#$ -l h_vmem=10.5G
#$ -q all.q
#$ -t 1-12

. /etc/profile.d/modules.sh
module load shared
module load proteus
module load sge/univa
module load gcc/4.8.1
module load python/intelpython/3.5.3

k=$1
USERNAME=sw424

SCRATCH=/scratch/$USERNAME/sweep
SCRATCH_OUT=$SCRATCH/out

OUT=/home/$USERNAME/earth/out

py=/mnt/HA/opt/python/intel/2017/intelpython3/bin/python3.5
script=8_sweep.py

cd $SCRATCH

$py $script $SGE_TASK_ID $k

cp $SCRATCH_OUT/* $OUT/

exit 0
