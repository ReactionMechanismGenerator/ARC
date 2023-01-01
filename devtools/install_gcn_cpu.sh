# Properly configure the shell to use 'conda activate'.
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh

if [ $(mamba) ] > /dev/null 2>&1; then 
	COMMAND_PKG=conda
	echo "conda found"
	else
	COMMAND_PKG=mamba
	echo "mamba found"
fi


# temporarily change directory to install software
pushd .
cd ..

# clone the repo in the parent directory and update it
echo "Cloning/Updating GCN..."
git clone https://github.com/ReactionMechanismGenerator/TS-GCN
cd TS-GCN || exit
git fetch origin
git checkout main
git pull origin main

# Add to PYTHONPATH
echo "Adding GCN to PYTHONPATH"
export PYTHONPATH=$PYTHONPATH:$(pwd)
echo 'export PYTHONPATH=$PYTHONPATH:'"$(pwd)" >> ~/.bashrc
echo $PYTHONPATH

# create the environment
echo "Creating the GCN-cpu environment..."
source ~/.bashrc
bash devtools/create_env_cpu.sh
conda activate ts_gcn
$COMMAND_PKG env update -f devtools/cpu_environment.yml

# Restore the original directory
cd ../ARC || exit
echo "Done installing GCN-cpu."
popd || exit
