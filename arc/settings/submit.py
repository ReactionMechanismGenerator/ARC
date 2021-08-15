"""
Submit scripts and incore commands
"""

# commands to execute ESS incore (without cluster software submission)
# stored as a dictionary with server and software as primary and secondary keys
incore_commands = {
    'local': {
        'gaussian': ['g16 < input.gjf > input.log', 'formchk check.chk check.fchk'],
    },
}

# Submission scripts for pipe.py stored as a dictionary with server as the key
pipe_submit = {
    'local': """#!/bin/bash -l
#SBATCH -p normal
#SBATCH -J {name}
#SBATCH -N 1
#SBATCH -n {cpus}
#SBATCH --time={t_max}
#SBATCH --mem-per-cpu={memory}
#SBATCH --array=1-{max_task_num}
#SBATCH -o out.txt
#SBATCH -e err.txt

source activate arc_env

python {arc_path}/arc/job/scripts/pipe.py {hdf5_path}

""",
}

# Submission scripts stored as a dictionary with server and software as primary and secondary keys
submit_scripts = {
    'local': {
        'gaussian': """#!/bin/bash -l
#SBATCH -p normal
#SBATCH -J {name}
#SBATCH -N 1
#SBATCH -n {cpus}
#SBATCH --time={t_max}
#SBATCH --mem-per-cpu={memory}
#SBATCH -o out.txt
#SBATCH -e err.txt

export g16root=/home/gridsan/groups/GRPAPI/Software
export PATH=$g16root/g16/:$g16root/gv:$PATH
which g16

echo "============================================================"
echo "Job ID : $SLURM_JOB_ID"
echo "Job Name : $SLURM_JOB_NAME"
echo "Starting on : $(date)"
echo "Running on node : $SLURMD_NODENAME"
echo "Current directory : $(pwd)"
echo "============================================================"


GAUSS_SCRDIR=/state/partition1/user/{un}/$SLURM_JOB_NAME-$SLURM_JOB_ID
export $GAUSS_SCRDIR
. $g16root/g16/bsd/g16.profile

mkdir -p $GAUSS_SCRDIR

g16 < input.gjf > input.log

rm -rf $GAUSS_SCRDIR

        """,
        'orca': """#!/bin/bash -l
#SBATCH -p normal
#SBATCH -J {name}
#SBATCH -N 1
#SBATCH -n {cpus}
#SBATCH --time={t_max}
#SBATCH --mem-per-cpu={memory}
#SBATCH -o out.txt
#SBATCH -e err.txt

echo "============================================================"
echo "Job ID : $SLURM_JOB_ID"
echo "Job Name : $SLURM_JOB_NAME"
echo "Starting on : $(date)"
echo "Running on node : $SLURMD_NODENAME"
echo "Current directory : $(pwd)"
echo "============================================================"

WorkDir=/state/partition1/user/{un}/$SLURM_JOB_NAME-$SLURM_JOB_ID
SubmitDir=`pwd`

#openmpi
export PATH=/home/gridsan/alongd/openmpi-3.1.4/bin:$PATH
export LD_LIBRARY_PATH=/home/gridsan/alongd/openmpi-3.1.4/lib:$LD_LIBRARY_PATH

#Orca
orcadir=/home/gridsan/alongd/orca_4_2_1_linux_x86-64_openmpi314
export PATH=/home/gridsan/alongd/orca_4_2_1_linux_x86-64_openmpi314:$PATH
export LD_LIBRARY_PATH=/home/gridsan/alongd/orca_4_2_1_linux_x86-64_openmpi314:$LD_LIBRARY_PATH
echo "orcaversion"
which orca
mkdir -p $WorkDir
cd $WorkDir
cp $SubmitDir/input.in .

$orcadir/orca input.in > input.log
cp input.log  $SubmitDir/
rm -rf  $WorkDir

""",
        'molpro': """#!/bin/bash -l
#SBATCH -p long
#SBATCH -J {name}
#SBATCH -N 1
#SBATCH -n {cpus}
#SBATCH --time={t_max}
#SBATCH --mem-per-cpu={memory}
#SBATCH -o out.txt
#SBATCH -e err.txt

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

cp "$SubmitDir/input.in" .

molpro -n {cpus} -d $sdir input.in

cp input.* "$SubmitDir/"
cp geometry*.* "$SubmitDir/"

rm -rf $sdir

""",
    },

    'atlas': {
        # Atlas uses HTCondor, see docs here: https://htcondor.readthedocs.io/en/latest/
        # Gaussian 09
        'gaussian': """Universe      = vanilla

+JobName      = "{name}"

log           = job.log
output        = out.txt
error         = err.txt

getenv        = True
+g09root      = "/Local/ce_dana"
+PATH         = "$(g09root)/g09:$PATH"
+GAUSS_EXEDIR = "$(g09root)/g09:$GAUSS_EXEDIR"
environment   = "GAUSS_EXEDIR=/Local/ce_dana/g09 GAUSS_SCRDIR=/storage/ce_dana/{un}/scratch/g09/ g09root=/Local/ce_dana"

should_transfer_files = no

executable = job.sh

request_cpus  = {cpus}
request_memory = {memory}MB

queue

""",
        # will be renamed to ``job.sh`` when uploaded
        'gaussian_job': """#!/bin/csh

mkdir -p /storage/ce_dana/{un}/scratch/g09/

source /Local/ce_dana/g09/bsd/g09.login

/Local/ce_dana/g09/g09 < input.gjf > input.log

""",

        # Orca
        'orca': """Universe      = vanilla

+JobName      = "{name}"

log           = job.log
output        = out.txt
error         = err.txt

getenv        = True
+WorkDir      = "/storage/ce_dana/{un}/scratch/orca/{name}"
environment   = "WorkDir=/storage/ce_dana/{un}/scratch/orca/{name}"

should_transfer_files = no

executable = job.sh

request_cpus  = {cpus}
request_memory = {memory}MB

queue

""",
        # will be renamed to ``job.sh`` when uploaded
        'orca_job': """#!/bin/bash -l

export OrcaDir=/Local/ce_dana/orca_4_0_1_2_linux_x86-64_openmpi202
export PATH=$PATH:$OrcaDir

export OMPI_Dir=/Local/ce_dana/openmpi-2.0.2/bin
export PATH=$PATH:$OMPI_Dir

export LD_LIBRARY_PATH=/Local/ce_dana/openmpi-2.0.2/lib:$LD_LIBRARY_PATH

SubmitDir=`pwd`

which orca

mkdir -p $WorkDir
cd $WorkDir

cp "$SubmitDir/input.in" .

${OrcaDir}/orca input.in > input.log
cp * "$SubmitDir/"

rm -rf $WorkDir

""",
    },

    'server1': {
        # Gaussian 16
        'gaussian': """#!/bin/bash -l
#SBATCH -p long
#SBATCH -J {name}
#SBATCH -N 1
#SBATCH -n {cpus}
#SBATCH --time={t_max}
#SBATCH --mem-per-cpu={memory}
#SBATCH -o out.txt
#SBATCH -e err.txt

export g16root=/opt
which g16

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

cp "$SubmitDir/input.gjf" .
cp "$SubmitDir/check.chk" .

g16 < input.gjf > input.log
formchk check.chk check.fchk
cp * "$SubmitDir/"

rm -rf $GAUSS_SCRDIR
rm -rf $WorkDir

""",
        # QChem 5.2
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

source /opt/qchem/qcenv.sh

export QC=/opt/qchem
export QCSCRATCH=/scratch/{un}/{name}
export QCLOCALSCR=/scratch/{un}/{name}/qlscratch
. $QC/qcenv.sh

mkdir -p /scratch/{un}/{name}/qlscratch

qchem -nt {cpus} input.in output.out

rm -r /scratch/{un}/{name}

""",
        # Molpro 2015
        'molpro': """#!/bin/bash -l
#SBATCH -p long
#SBATCH -J {name}
#SBATCH -N 1
#SBATCH -n {cpus}
#SBATCH --time={t_max}
#SBATCH --mem-per-cpu={memory}
#SBATCH -o out.txt
#SBATCH -e err.txt

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

cp "$SubmitDir/input.in" .

molpro -n {cpus} -d $sdir input.in

cp input.* "$SubmitDir/"
cp geometry*.* "$SubmitDir/"

rm -rf $sdir

""",
        # Orca
        'orca': """#!/bin/bash -l
#SBATCH -p normal
#SBATCH -J {name}
#SBATCH -N 1
#SBATCH -n {cpus}
#SBATCH --time={t_max}
#SBATCH --mem-per-cpu={memory}
#SBATCH -o out.txt
#SBATCH -e err.txt

export PATH=/opt/orca_4_2_1_linux_x86-64_openmpi314/:/opt/openmpi-3.1.4/bin/:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/openmpi-3.1.4/lib/:/opt/openmpi-3.1.4/etc
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

cp "$SubmitDir/input.in" .

${ORCA_DIR}/orca input.in > input.log
cp * "$SubmitDir/"

rm -rf $WorkDir

""",
        # TeraChem
        'terachem': """#!/bin/bash -l
#SBATCH -J {name}
#SBATCH -t {t_max}
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu={memory}
#SBATCH -o out.txt
#SBATCH -e err.txt

echo "============================================================"
echo "Job ID : $SLURM_JOB_ID"
echo "Job Name : $SLURM_JOB_NAME"
echo "Starting on : $(date)"
echo "Running on node : $SLURMD_NODENAME"
echo "Current directory : $(pwd)"
echo "============================================================"

module load cuda92/toolkit
module load medsci
module load terachem

terachem input.in > output.out

""",
    },

    'server2': {
        # Gaussian 16
        'gaussian': """#!/bin/bash -l

#$ -N {name}
#$ -l long{architecture}
#$ -l h_rt={t_max}
#$ -l h_vmem={memory}M
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
#$ -l h_vmem={memory}M
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
        # QChem 5.2
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

source /opt/qchem/qcenv.sh

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
        # Orca
        'orca': """#!/bin/bash -l

#$ -N {name}
#$ -l long{architecture}
#$ -l h_rt={t_max}
#$ -l h_vmem={memory}M
#$ -pe singlenode {cpus}
#$ -cwd
#$ -o out.txt
#$ -e err.txt

echo "Running on node:"
hostname

export PATH=/opt/orca/:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/:/usr/local/etc

WorkDir=/scratch/{un}/{name}
SubmitDir=`pwd`

mkdir -p $WorkDir
cd $WorkDir

cp "$SubmitDir/input.in" .

/opt/orca/orca input.in > input.log
cp * "$SubmitDir/"

rm -rf $WorkDir

""",
    },
    'txe1': {
        'orca': """#!/bin/bash -l
#SBATCH -p normal
#SBATCH -J {name}
#SBATCH -N 1
#SBATCH -n {cpus}
#SBATCH --time=5-0:00:00
#SBATCH --mem-per-cpu={memory}
#SBATCH -o out.txt
#SBATCH -e err.txt

echo "============================================================"
echo "Job ID : $SLURM_JOB_ID"
echo "Job Name : $SLURM_JOB_NAME"
echo "Starting on : $(date)"
echo "Running on node : $SLURMD_NODENAME"
echo "Current directory : $(pwd)"
echo "memory per node : $SLURM_MEM_PER_NODE
echo "memory per cpu : $SLURM_MEM_PER_CPU
echo "============================================================"

WorkDir=/state/partition1/user/{un}/$SLURM_JOB_NAME-$SLURM_JOB_ID
SubmitDir=`pwd`

#openmpi
export PATH=/home/gridsan/groups/GRPAPI/Software/openmpi-3.1.4/bin:$PATH
export LD_LIBRARY_PATH=/home/gridsan/groups/GRPAPI/Software/openmpi-3.1.4/lib:$LD_LIBRARY_PATH

#Orca
orcadir=/home/gridsan/groups/GRPAPI/Software/orca_4_2_1_linux_x86-64_openmpi314
export PATH=/home/gridsan/groups/GRPAPI/Software/orca_4_2_1_linux_x86-64_openmpi314:$PATH
export LD_LIBRARY_PATH=/home/gridsan/groups/GRPAPI/Software/orca_4_2_1_linux_x86-64_openmpi314:$LD_LIBRARY_PATH
echo "orcaversion"
which orca
mkdir -p $WorkDir
cd $WorkDir
cp $SubmitDir/input.in .

$orcadir/orca input.in > input.log
cp input.log  $SubmitDir/
rm -rf  $WorkDir
        """,
    },
    'pbs_sample': {
        'gaussian': """#!/bin/bash -l
#PBS -q batch
#PBS -l nodes=1:ppn={cpus}
#PBS -l mem={memory}mb
#PBS -l walltime=48:00:00
#PBS -N {name}
#PBS -o out.txt
#PBS -e err.txt

export g16root=/home/{un}/Software
export PATH=$g16root/g16/:$g16root/gv:$PATH
which g16

. $g16root/g16/bsd/g16.profile

export GAUSS_SCRDIR=/home/{un}/scratch/$SLURM_JOB_NAME-$SLURM_JOB_ID
mkdir -p $GAUSS_SCRDIR
chmod 750 $GAUSS_SCRDIR

g16 < input.gjf > input.log

rm -rf $GAUSS_SCRDIR

    """,
    },
}
