# Properly configure the shell to use 'conda activate'.
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh

#Check if mamba/conda is installed
if [ -x "$(command -v mamba)" ]; then
	echo "mamba is installed."
	COMMAND_PKG=mamba
elif [ -x "$(command -v conda)" ]; then
	echo "conda is installed."
	COMMAND_PKG=conda
else
    echo "mamba and conda are not installed. Please download and install mamba or conda - we strongly recommend mamba"
fi

$COMMAND_PKG create -n xtb_env python=3.7 -y
conda activate xtb_env
$COMMAND_PKG install -c conda-forge xtb -y
$COMMAND_PKG install -c anaconda pyyaml -y
conda deactivate
