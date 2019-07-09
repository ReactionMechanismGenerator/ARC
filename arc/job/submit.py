#!/usr/bin/env python
# encoding: utf-8

"""
Submit scripts
sorted in a dictionary with server names as keys
"""

##################################################################


submit_scripts = {
    'c3ddb': {
        # Gaussian 09
        'gaussian': """#!/bin/bash -l
#SBATCH -p defq
#SBATCH -J {name}
#SBATCH -N 1
#SBATCH -n {cpus}
#SBATCH --time={t_max}
#SBATCH --mem-per-cpu {mem_per_cpu}

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
export GAUSS_SCRDIR

mkdir -p $GAUSS_SCRDIR
mkdir -p $WorkDir

cd $WorkDir
. $g09root/g09/bsd/g09.profile

cp $SubmitDir/input.gjf .
cp $SubmitDir/check.chk .

g09 < input.gjf > input.log
formchk  check.chk check.fchk
cp * $SubmitDir/

rm -rf $GAUSS_SCRDIR
rm -rf $WorkDir

""",

        # Orca
        'orca': """#!/bin/bash -l
#SBATCH -p defq
#SBATCH -J {name}
#SBATCH -N 1
#SBATCH -n {cpus}
#SBATCH --time={t_max}
#SBATCH --mem-per-cpu {mem_per_cpu}

module add c3ddb/orca/4.1.2
module add c3ddb/openmpi/3.1.3
which orca

export ORCA_DIR=/cm/shared/modulefiles/c3ddb/orca/4.1.2/
export OMPI_DIR=/cm/shared/modulefiles/c3ddb/openmpi/3.1.3/
export PATH=$PATH:$ORCA_DIR
export PATH=$PATH:$OMPI_DIR

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
cd $WorkDir

cp $SubmitDir/input.inp .

${ORCA_DIR}/orca input.inp > input.log
cp * $SubmitDir/

rm -rf $WorkDir

""",
    },

    'rmg': {
        # Gaussian 16
        'gaussian': """#!/bin/bash -l
#SBATCH -p long
#SBATCH -J {name}
#SBATCH -N 1
#SBATCH -n {cpus}
#SBATCH --time={t_max}
#SBATCH --mem-per-cpu={mem_per_cpu}

which 16

echo "============================================================"
echo "Job ID : $SLURM_JOB_ID"
echo "Job Name : $SLURM_JOB_NAME"
echo "Starting on : $(date)"
echo "Running on node : $SLURMD_NODENAME"
echo "Current directory : $(pwd)"
echo "============================================================"

WorkDir=/scratch/{un}/$SLURM_JOB_NAME-$SLURM_JOB_ID
SubmitDir=`pwd`

GAUSS_SCRDIR=/scratch/{un}/g16/$SLURM_JOB_NAME-$SLURM_JOB_ID
export GAUSS_SCRDIR

mkdir -p $GAUSS_SCRDIR
mkdir -p $WorkDir

cd $WorkDir
. $g16root/g16/bsd/g16.profile

cp $SubmitDir/input.gjf .

g16 < input.gjf > input.log
formchk check.chk check.fchk
cp * $SubmitDir/

rm -rf $GAUSS_SCRDIR
rm -rf $WorkDir

""",
        # Molpro 2015
        'molpro': """#!/bin/bash -l
#SBATCH -p long
#SBATCH -J {name}
#SBATCH -N 1
#SBATCH -n {cpus}
#SBATCH --time={t_max}
#SBATCH --mem-per-cpu={mem_per_cpu}

export PATH=/opt/molpro/molprop_2015_1_linux_x86_64_i8/bin:$PATH

echo "============================================================"
echo "Job ID : $SLURM_JOB_ID"
echo "Job Name : $SLURM_JOB_NAME"
echo "Starting on : $(date)"
echo "Running on node : $SLURMD_NODENAME"
echo "Current directory : $(pwd)"
echo "============================================================"

sdir=/scratch/{un}/$SLURM_JOB_NAME-$SLURM_JOB_ID
SubmitDir=`pwd`

mkdir -p $sdir
cd $sdir

cp $SubmitDir/input.in .

molpro -n {cpus} -d $sdir input.in

cp input.* $SubmitDir/
cp geometry*.* $SubmitDir/

rm -rf $sdir

""",
        # Gromacs
        'gromacs': """#!/bin/bash -l
#SBATCH -p long
#SBATCH -J {name}
#SBATCH -N 1
#SBATCH --time={t_max}

echo "============================================================"
echo "Job ID : $SLURM_JOB_ID"
echo "Job Name : $SLURM_JOB_NAME"
echo "Starting on : $(date)"
echo "Running on node : $SLURMD_NODENAME"
echo "Current directory : $(pwd)"
echo "============================================================"

python mdconf.py -s {size}

""",
    },

    'pharos': {
        # Gaussian 16
        'gaussian': """#!/bin/bash -l

#$ -N {name}
#$ -l long{architecture}
#$ -l h_rt={t_max}
#$ -pe singlenode {cpus}
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
        # Gaussian 03
        'gaussian03_pharos': """#!/bin/bash -l

#$ -N {name}
#$ -l long{architecture}
#$ -l h_rt={t_max}
#$ -pe singlenode {cpus}
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
        # QChem 4.4
        'qchem': """#!/bin/bash -l

#$ -N {name}
#$ -l long{architecture}
#$ -l h_rt={t_max}
#$ -pe singlenode {cpus}
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

qchem -nt {cpus} input.in output.out

rm -r /scratch/{un}/{name}

""",
        # Molpro 2012
        'molpro': """#! /bin/bash -l

#$ -N {name}
#$ -l long{architecture}
#$ -l h_rt={t_max}
#$ -pe singlenode {cpus}
#$ -l h=!node60.cluster
#$ -cwd
#$ -o out.txt
#$ -e err.txt

export PATH=/opt/molpro2012/molprop_2012_1_Linux_x86_64_i8/bin:$PATH

sdir=/scratch/{un}
mkdir -p /scratch/{un}/qlscratch

molpro -d $sdir -n {cpus} input.in

""",
        # OneDMin
        'onedmin': """#! /bin/bash -l

#$ -N {name}
#$ -l long{architecture}
#$ -l h_rt={t_max}
#$ -pe singlenode {cpus}
#$ -l h=!node60.cluster
#$ -cwd
#$ -o out.txt
#$ -e err.txt

WorkDir=`pwd`
cd
sdir=/scratch/{un}
mkdir -p /scratch/{un}/onedmin
cd $WorkDir

~/auto1dmin/exe/auto1dmin.x < input.in > output.out

""",
    }
}
