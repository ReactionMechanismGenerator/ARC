#!/usr/bin/env python
# encoding: utf-8


##################################################################

# These submit scripts must be filled with {0} = job_name, {1} = user_name
submit_sctipts = {
    'gaussian03': """#!/bin/bash
#$ -N {0}
#$ -l long
#$ -l h_rt=120:00:00
#$ -l harpertown
#$ -m ae
#$ -pe singlenode 8
#$ -cwd
#$ -o out.txt
#$ -e err.txt

PATH=$PATH:/home/alongd
export LD_LIBRARY_PATH=/opt/intel/Compiler/11.0/074/bin/intel64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/intel/Compiler/11.0/074/mkl/lib/em64t:$LD_LIBRARY_PATH
g03root=/opt
GAUSS_SCRDIR=/scratch/alongd
export g03root GAUSS_SCRDIR
. $g03root/g03/bsd/g03.profile
export LD_LIBRARY_PATH=$HOME'/gcc/gcc-4.7/lib64':$LD_LIBRARY_PATH
export PATH LIBPATH GAUSS_SCRDIR g03root
export PATH=$PATH:$SGE_ROOT/bin/lx24-amd64
export PATH=$PATH:/opt/mpich2-1.2.1p1/bin

WorkDir=`pwd`
cd
source .bashrc
mkdir -p /scratch/{1}
cd $WorkDir

g03 input.in

""",
    'qchem': """#!/bin/bash
 
#$ -N {0}
#$ -l long
#$ -l h_rt=120:00:00
#$ -l harpertown
#$ -m ae
#$ -pe singlenode 8
#$ -cwd
#$ -o out.txt
#$ -e err.txt

WorkDir=`pwd`
cd
source .bashrc
mkdir -p /scratch/{1}
mkdir -p /scratch/{1}/qlscratch
cd $WorkDir

qchem -nt 6 input.in output.out

""",
    'molpro_2015': """#!/bin/bash
#SBATCH -J {0}
#SBATCH --nodelist=node[08]

WorkDir=`pwd`
cd
source .bashrc
sdir=/scratch/{1}/$SLURM_JOB_ID
mkdir -p $sdir
export TMPDIR=$sdir
cd $WorkDir

molpro input.in

rm -rf $sdir

""",
    'molpro_2012': """#! /bin/bash
 
#$ -N {0}
#$ -l long
#$ -l h_rt=120:00:00
#$ -l harpertown
#$ -m ae
#$ -pe singlenode 8
#$ -cwd
#$ -o out.txt
#$ -e err.txt

WorkDir=`pwd`
cd
source .bashrc
mkdir -p /scratch/{1}
mkdir -p /scratch/{1}/qlscratch
cd $WorkDir

molpro -n 6 input.in
""",
}
