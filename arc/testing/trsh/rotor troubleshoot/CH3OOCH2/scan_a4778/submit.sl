#!/bin/bash -l
#SBATCH -p long
#SBATCH -J a4778
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --time=5-0:00:00
#SBATCH --mem-per-cpu=985

which 16
export g16root=/opt

echo "============================================================"
echo "Job ID : $SLURM_JOB_ID"
echo "Job Name : $SLURM_JOB_NAME"
echo "Starting on : $(date)"
echo "Running on node : $SLURMD_NODENAME"
echo "Current directory : $(pwd)"
echo "============================================================"

WorkDir=/scratch/duminda/$SLURM_JOB_NAME-$SLURM_JOB_ID
SubmitDir=`pwd`

GAUSS_SCRDIR=/scratch/duminda/g16/$SLURM_JOB_NAME-$SLURM_JOB_ID
export GAUSS_SCRDIR

mkdir -p $GAUSS_SCRDIR
mkdir -p $WorkDir

cd $WorkDir
. $g16root/g16/bsd/g16.profile

cp $SubmitDir/input.gjf .
cp $SubmitDir/check.chk .

g16 < input.gjf > input.log
formchk check.chk check.fchk
cp * $SubmitDir/

rm -rf $GAUSS_SCRDIR
rm -rf $WorkDir

