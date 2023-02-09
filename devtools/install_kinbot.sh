# Properly configure the shell to use 'conda activate'.
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh

# temporarily change directory to install software, and move one directory up in the tree
pushd .
cd ..

# clone the repo in the parent directory and update it
echo "Cloning/Updating KinBot..."
wget https://github.com/zadorlab/KinBot/archive/refs/tags/v2.0.6.tar.gz
tar -xvf "v2.0.6.tar.gz"
cd KinBot-2.0.6 || exit
conda activate arc_env
python setup.py build
python setup.py install

# Add to PYTHONPATH
echo "Adding KinBot to PYTHONPATH"
export PYTHONPATH=$PYTHONPATH:$(pwd)
echo 'export PYTHONPATH=$PYTHONPATH:'"$(pwd)" >> ~/.bashrc
echo $PYTHONPATH

# Restore the original directory
cd ../ARC || exit
echo "Done installing Kinbot."
popd || exit
