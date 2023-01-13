# Properly configure the shell to use 'conda activate'.
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh

# Check for package manager
if command -v mamba 2>/dev/null; then
    COMMAND_PKG=mamba
elif command -v conda >/dev/null 2>&1; then
    COMMAND_PKG=conda
else
    echo "Error: mamba or conda is not installed. Please download and install mamba or conda - we strongly recommend mamba"
    exit 1
fi

# temporarily change directory to install software
pushd .
cd ..

# clone the repo in the parent directory and update it
echo "Cloning/Updating GCN..."
if [ -d "./TS-GCN" ]; then
    cd TS-GCN
else
git clone https://github.com/ReactionMechanismGenerator/TS-GCN
    cd TS-GCN || exit
fi
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
