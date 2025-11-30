#!/bin/bash -l

#PBS -q alon_comb_q
#PBS -N test_constraint
#PBS -l select=1:ncpus=4:mem=4000000000:mpiprocs=4
#PBS -o out.txt
#PBS -e err.txt

. ~/.bashrc

PBS_O_WORKDIR="/home/calvin.p/test_constraint_scan"
cd "$PBS_O_WORKDIR"

source /usr/local/g09/setup.sh

GAUSS_SCRDIR="/gtmp/calvin.p/scratch/g09/$PBS_JOBID"

mkdir -p "$GAUSS_SCRDIR"

export GAUSS_SCRDIR="$GAUSS_SCRDIR"

touch initial_time

cd "$GAUSS_SCRDIR"

cp "$PBS_O_WORKDIR/input.gjf" "$GAUSS_SCRDIR"

cleanup() {
    echo "Cleaning scratch: $GAUSS_SCRDIR"
    cp -f "$GAUSS_SCRDIR"/check.chk "$PBS_O_WORKDIR"/ 2>/dev/null
    cp -f "$GAUSS_SCRDIR"/*.rwf "$PBS_O_WORKDIR"/ 2>/dev/null
    rm -rf "$GAUSS_SCRDIR"
}
trap cleanup EXIT TERM INT

if [ -f "$PBS_O_WORKDIR/check.chk" ]; then
    cp "$PBS_O_WORKDIR/check.chk" "$GAUSS_SCRDIR/"
fi

g09 < input.gjf > input.log

cp input.* "$PBS_O_WORKDIR/"

if [ -f check.chk ]; then
    cp check.chk "$PBS_O_WORKDIR/"
fi

rm -vrf "$GAUSS_SCRDIR"

cd "$PBS_O_WORKDIR"

touch final_time
