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
$COMMAND_PKG clean -a -y
$COMMAND_PKG env create -n ob_env conda-forge::openbabel
conda activate ob_env
$COMMAND_PKG install -c conda-forge pyyaml
conda deactivate