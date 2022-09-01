# Note: this script assumes the user is running it from the root ARC folder.

# Temporarily change directory to install software (don't use "|| exit" here)
pushd .

# install dependencies
bash devtools/install_autotst.sh
bash devtools/install_gcn_cpu.sh
bash devtools/install_kinbot.sh
bash devtools/install_xtb.sh
bash devtools/install_sella.sh
bash devtools/install_psi4.sh
echo "Done installing external repos."
source ~/.bashrc
echo $PYTHONPATH

# Restore original directory
popd || exit
