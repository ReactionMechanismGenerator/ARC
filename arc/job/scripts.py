#!/usr/bin/env python
# encoding: utf-8

import os
import string

##################################################################

# The scripts in this file are for functions in ARC.

# Server node test scripts

server_test_script = dict()
server_test_script['pharos'] = dict()

server_test_script['pharos']['core_test_submit'] = """#!/bin/bash

export PATH=/opt/sge/bin/lx24-amd64:$PATH;
export PATH=/usr/bin/mh:$PATH;
source ~/.bashrc;
export SGE_ROOT=/opt/sge;

f8=working_nodes_8core.txt;
f48=working_nodes_48core.txt;
fwk=working_nodes_all.txt;
ftmp=nodes.tmp;
ftmp8=nodes8.tmp;
ftmp48=nodes48.tmp;

rm *.out $f8 $f48 $fwk $ftmp $ftmp8 $ftmp48;

for n in $(seq 98);
do
  node=node$(printf "%02d" $n)
  qsub -q *@$node.cluster -o $node.out -j y ctest.sh
  echo $(whoami) > test.txt
done;

sleep 6s;

qstat -u $(whoami) | grep "ctest.sh" > $ftmp;
qdel `qstat -u $(whoami) | grep "ctest.sh" | cut -c1-7`;
less $ftmp | cut -c77-78 > $fwk;
sed -ri '/^\s*$/d' $fwk;

seq -w 63 > $ftmp8
cat $fwk $ftmp8 | sort | uniq -d > $f8;

seq 64 98 > $ftmp48
cat $fwk $ftmp48 | sort | uniq -d > $f48;

sed -ri 's/.*/#$ -q *@node&.cluster/' $f8 $f48;

rm *.out $ftmp $ftmp8 $ftmp48;"""

server_test_script['pharos']['core_test'] = """#!/bin/bash

#$ -pe singlenode 8
#$ -l long

sleep 120s;"""