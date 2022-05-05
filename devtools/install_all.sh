# temporarily change directory to install software (don't use "|| exit" here)
pushd .

# install dependencies
bash devtools/install_autotst.sh
bash devtools/install_gcn_cpu.sh
bash devtools/install_kinbot.sh
echo "Done installing external repos."
source ~/.bashrc
echo $PYTHONPATH

# Restore original directory
popd || exit
