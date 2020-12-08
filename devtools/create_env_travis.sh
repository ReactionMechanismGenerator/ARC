# Activate the arc_env before running this script.
# This script does the following tasks:
# 	- installs PyTorch (cpu version) in the environment
# 	- installs torch torch-geometric in the environment


# get OS type
unameOut="$(uname -s)"
case "${unameOut}" in
    Linux*)     machine=Linux;;
    Darwin*)    machine=MacOS;;
    CYGWIN*)    machine=Cygwin;;
    MINGW*)     machine=MinGw;;
    *)          machine="UNKNOWN:${unameOut}"
esac
echo "Running ${machine}..."


# installs the cpu version of PyTorch
# source: https://pytorch.org/get-started/locally/

# "cpuonly" works for Linux and Windows
CUDA="cpuonly"
# Mac does not use "cpuonly"
if [ $machine == "Mac" ]
then
    CUDA=" "
fi  
CUDA_VERSION="cpu"

echo "Installing PyTorch with requested CUDA version..."
echo "Running: conda install pytorch torchvision $CUDA -c pytorch"
conda install pytorch torchvision $CUDA -c pytorch

# source: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
echo "Installing torch-geometric..."
echo "Using CUDA version: $CUDA_VERSION"
# get PyTorch version
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
echo "Using PyTorch version: $TORCH_VERSION"

pip install torch-scatter==latest+$CUDA_VERSION -f https://pytorch-geometric.com/whl/torch-$TORCH_VERSION.html
pip install torch-sparse==latest+$CUDA_VERSION -f https://pytorch-geometric.com/whl/torch-$TORCH_VERSION.html
pip install torch-cluster==latest+$CUDA_VERSION -f https://pytorch-geometric.com/whl/torch-$TORCH_VERSION.html
pip install torch-spline-conv==latest+$CUDA_VERSION -f https://pytorch-geometric.com/whl/torch-$TORCH_VERSION.html
pip install torch-geometric
