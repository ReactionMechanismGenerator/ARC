# temporarily change directory to install software (don't use "|| exit" here)
cd devtools

# install dependencies
bash install_autotst.sh
bash install_gcn_cpu.sh
bash install_kinbot.sh

# Restore original directory
popd || exit
