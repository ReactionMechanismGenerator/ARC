#!/usr/bin/env python
# encoding: utf-8


##################################################################


submit_scripts = {
    'gaussian': """#!/bin/bash
#$ -N {name}
#$ -l long
#$ -l h_rt=120:00:00
#$ -l harpertown
#$ -m ae
#$ -pe singlenode 8
#$ -cwd
#$ -o out.txt
#$ -e err.txt

PATH=$PATH:/home/{un}
export LD_LIBRARY_PATH=/opt/intel/Compiler/11.0/074/bin/intel64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/intel/Compiler/11.0/074/mkl/lib/em64t:$LD_LIBRARY_PATH
g03root=/opt
GAUSS_SCRDIR=/scratch/{un}
export g03root GAUSS_SCRDIR
. $g03root/g03/bsd/g03.profile
export LD_LIBRARY_PATH=$HOME'/gcc/gcc-4.7/lib64':$LD_LIBRARY_PATH
export PATH LIBPATH GAUSS_SCRDIR g03root
export PATH=$PATH:$SGE_ROOT/bin/lx24-amd64
export PATH=$PATH:/opt/mpich2-1.2.1p1/bin

WorkDir=`pwd`
cd
source .bashrc
mkdir -p /scratch/{un}
cd $WorkDir

g03 input.gjf

""",
    'qchem': """#!/bin/bash
 
#$ -N {name}
#$ -l long
#$ -l h_rt=120:00:00
#$ -l harpertown
#$ -m ae
#$ -pe singlenode 8
#$ -cwd
#$ -o out.txt
#$ -e err.txt

export QC=/opt/qchem
export QCSCRATCH=/scratch/{un}
export QCLOCALSCR=/scratch/{un}/qlscratch
. $QC/qcenv.sh

WorkDir=`pwd`
cd
source .bashrc
mkdir -p /scratch/{un}
mkdir -p /scratch/{un}/qlscratch
cd $WorkDir

qchem -nt 6 input.in output.out

""",
    'molpro': """#!/bin/bash
#SBATCH -p normal
#SBATCH -J {name}
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem-per-cpu=2048

echo "============================================================"
echo "Job ID : $SLURM_JOB_ID"
echo "Job Name : $SLURM_JOB_NAME"
echo "Starting on : $(date)"
echo "Running on node : $SLURMD_NODENAME"
echo "Current directory : $(pwd)"
echo "============================================================"

WorkDir=`pwd`
cd
source .bashrc
sdir=/scratch/{un}/$SLURM_JOB_ID
mkdir -p $sdir
export TMPDIR=$sdir
cd $WorkDir

molpro input.in

rm -rf $sdir

""",
#     'molpro': """#! /bin/bash
#
# #$ -N {name}
# #$ -l long
# #$ -l h_rt=120:00:00
# #$ -l harpertown
# #$ -m ae
# #$ -pe singlenode 8
# #$ -cwd
# #$ -o out.txt
# #$ -e err.txt
#
# export MPICH2_ROOT=/opt/mpich2-1.2.1p1
# export PATH=/opt/molpro2012/molprop_2012_1_Linux_x86_64_i8/bin:$PATH
# export LD_LIBRARY_PATH=/opt/gcc-4.7/lib:$LD_LIBRARY_PATH
#
# WorkDir=`pwd`
# cd
# source .bashrc
# mkdir -p /scratch/{un}
# mkdir -p /scratch/{un}/qlscratch
# cd $WorkDir
#
# molpro -n 6 input.in
# """,
}
