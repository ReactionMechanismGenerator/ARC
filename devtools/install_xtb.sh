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


$COMMAND_PKG create -n xtb_env python=3.7 -y
conda activate xtb_env
$COMMAND_PKG install -c conda-forge xtb -y
$COMMAND_PKG install -c anaconda pyyaml -y
conda deactivate
