#!/bin/bash -l

# Check if Micromamba is installed
if [[ -x "$(command -v micromamba)" ]]; then
	echo "Micromamba is installed."
	COMMAND_PKG=micromamba
# Check if Mamba is installed
elif [[ -x "$(command -v mamba)" ]]; then
	echo "Mamba is installed."
	COMMAND_PKG=mamba
# Check if Conda is installed
elif [[ -x "$(command -v conda)" ]]; then
	echo "Conda is installed."
	COMMAND_PKG=conda
else
	echo "Micromamba, Mamba, and Conda are not installed. Please download and install one of them - we strongly recommend Micromamba or Mamba."
	exit 1
fi

# Set up Conda/Micromamba environment
if [[ ${COMMAND_PKG} == "micromamba" ]]; then
	eval "$(micromamba shell hook --shell=bash)"
	micromamba activate base
	BASE=${MAMBA_ROOT_PREFIX}
	# shellcheck source=/dev/null
	. "${BASE}/etc/profile.d/micromamba.sh"
else
	BASE=$(conda info --base)
	# shellcheck source=/dev/null
	. "${BASE}/etc/profile.d/conda.sh"
fi

# create the environment
echo "Creating the Crest environment..."
${COMMAND_PKG} create -n crest_env -c conda-forge python=3.10 crest=2.12 -y
# Activate the environment
if [[ ${COMMAND_PKG} == "micromamba" ]]; then
	micromamba activate crest_env
else
	conda activate crest_env
fi

echo "Done installing Crest environment."
