#!/bin/bash
# Script to submit constraint scan test to Zeus

# Create directory on Zeus
ssh calvin.p@zeus.technion.ac.il "mkdir -p /home/calvin.p/test_constraint_scan"

# Copy input file
scp test_constraint_scan_input.gjf calvin.p@zeus.technion.ac.il:/home/calvin.p/test_constraint_scan/input.gjf

# Copy submit script
scp test_constraint_scan_submit.sh calvin.p@zeus.technion.ac.il:/home/calvin.p/test_constraint_scan/submit.sh

# Submit job
ssh calvin.p@zeus.technion.ac.il "cd /home/calvin.p/test_constraint_scan && qsub submit.sh"

echo "Job submitted. To check status:"
echo "ssh calvin.p@zeus.technion.ac.il 'qstat -u calvin.p'"
echo ""
echo "To check results after completion:"
echo "ssh calvin.p@zeus.technion.ac.il 'cd /home/calvin.p/test_constraint_scan && cat input.log | head -100'"
