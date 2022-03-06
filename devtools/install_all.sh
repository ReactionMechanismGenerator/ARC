# temporarily change directory to install software (don't use "|| exit" here)
pushd .
cd devtools

# install dependencies
bash install_autotst.sh
bash install_gcn_cpu.sh
bash install_kinbot.sh
echo $PYTHONPATH

# Restore original directory
popd || exit
