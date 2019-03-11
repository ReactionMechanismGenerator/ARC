#!/usr/bin/env python
# encoding: utf-8


##################################################################


submit_scripts = {
    'Slurm': {
        # Gaussian09 on C3DDB
        'gaussian': """#!/bin/bash -l
#SBATCH -p defq
#SBATCH -J {name}
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --time={t_max}
#SBATCH --mem-per-cpu 4500

module add c3ddb/gaussian/09.d01
which g09

echo "============================================================"
echo "Job ID : $SLURM_JOB_ID"
echo "Job Name : $SLURM_JOB_NAME"
echo "Starting on : $(date)"
echo "Running on node : $SLURMD_NODENAME"
echo "Current directory : $(pwd)"
echo "============================================================"

WorkDir=/scratch/users/{un}/$SLURM_JOB_NAME-$SLURM_JOB_ID
SubmitDir=`pwd`

GAUSS_SCRDIR=/scratch/users/{un}/g09/$SLURM_JOB_NAME-$SLURM_JOB_ID
export  GAUSS_SCRDIR

mkdir -p $GAUSS_SCRDIR
mkdir -p $WorkDir

cd  $WorkDir
. $g09root/g09/bsd/g09.profile

cp $SubmitDir/input.gjf .

g09 < input.gjf > input.log
formchk  check.chk check.fchk
cp * $SubmitDir/

rm -rf $GAUSS_SCRDIR
rm -rf $WorkDir

""",

        # Orca on C3DDB:
        'orca': """#!/bin/bash -l
#SBATCH -p defq
#SBATCH -J {name}
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --time={t_max}
#SBATCH --mem-per-cpu 4500

module add c3ddb/orca/4.0.0
module add c3ddb/openmpi/2.0.2
which orca

export ORCA_DIR=/cm/shared/c3ddb/orca/4.0.0/
export PATH=$PATH:$ORCA_DIR

echo "============================================================"
echo "Job ID : $SLURM_JOB_ID"
echo "Job Name : $SLURM_JOB_NAME"
echo "Starting on : $(date)"
echo "Running on node : $SLURMD_NODENAME"
echo "Current directory : $(pwd)"
echo "============================================================"


WorkDir=/scratch/users/{un}/$SLURM_JOB_NAME-$SLURM_JOB_ID
SubmitDir=`pwd`

mkdir -p $WorkDir
cd  $WorkDir

cp $SubmitDir/input.inp .

${ORCA_DIR}/orca input.inp > input.log
cp * $SubmitDir/

rm -rf $WorkDir

""",

        # Molpro 2015 on RMG
        'molpro': """#!/bin/bash -l
#SBATCH -p long
#SBATCH -J {name}
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --time={t_max}
#SBATCH --mem-per-cpu={mem_cpu}

export PATH=/opt/molpro/molprop_2015_1_linux_x86_64_i8/bin:$PATH

echo "============================================================"
echo "Job ID : $SLURM_JOB_ID"
echo "Job Name : $SLURM_JOB_NAME"
echo "Starting on : $(date)"
echo "Running on node : $SLURMD_NODENAME"
echo "Current directory : $(pwd)"
echo "============================================================"

sdir=/scratch/{un}/$SLURM_JOB_NAME-$SLURM_JOB_ID
mkdir -p $sdir

molpro -n 8 -d $sdir input.in

rm -rf $sdir

""",
    },


    'OGE': {
        # Gaussian16 on Pharos
        'gaussian': """#!/bin/bash -l

#$ -N {name}
#$ -l long
#$ -l harpertown
#$ -l h_rt={t_max}
#$ -pe singlenode 6
#$ -l h=!node60.cluster
#$ -cwd
#$ -o out.txt
#$ -e err.txt

echo "Running on node:"
hostname

g16root=/opt
GAUSS_SCRDIR=/scratch/{un}/{name}
export g16root GAUSS_SCRDIR
. $g16root/g16/bsd/g16.profile
mkdir -p /scratch/{un}/{name}

g16 input.gjf

rm -r /scratch/{un}/{name}

""",
        # Gaussian03 on Pharos
        'gaussian03_pharos': """#!/bin/bash -l

#$ -N {name}
#$ -l long
#$ -l harpertown
#$ -l h_rt={t_max}
#$ -pe singlenode 6
#$ -l h=!node60.cluster
#$ -cwd
#$ -o out.txt
#$ -e err.txt

echo "Running on node:"
hostname

g03root=/opt
GAUSS_SCRDIR=/scratch/{un}/{name}
export g03root GAUSS_SCRDIR
. $g03root/g03/bsd/g03.profile
mkdir -p /scratch/{un}/{name}

g03 input.gjf

rm -r /scratch/{un}/{name}

""",
        # QChem 4.4 on Pharos:
        'qchem': """#!/bin/bash -l

#$ -N {name}
#$ -l long
#$ -l harpertown
#$ -l h_rt={t_max}
#$ -pe singlenode 6
#$ -l h=!node60.cluster
#$ -cwd
#$ -o out.txt
#$ -e err.txt

echo "Running on node:"
hostname

export QC=/opt/qchem
export QCSCRATCH=/scratch/{un}/{name}
export QCLOCALSCR=/scratch/{un}/{name}/qlscratch
. $QC/qcenv.sh

mkdir -p /scratch/{un}/{name}/qlscratch

qchem -nt 6 input.in output.out

rm -r /scratch/{un}/{name}

""",
        # Molpro 2012 on Pharos
        'molpro': """#! /bin/bash -l

#$ -N {name}
#$ -l long
#$ -l harpertown
#$ -l h_rt={t_max}
#$ -pe singlenode 6
#$ -l h=!node60.cluster
#$ -cwd
#$ -o out.txt
#$ -e err.txt

export PATH=/opt/molpro2012/molprop_2012_1_Linux_x86_64_i8/bin:$PATH

sdir=/scratch/{un}
mkdir -p /scratch/{un}/qlscratch

molpro -d $sdir -n 6 input.in
""",
    }
}
