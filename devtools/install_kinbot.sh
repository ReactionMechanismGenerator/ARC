# Properly configure the shell to use 'conda activate'.
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh

# temporarily change directory to install software, and move one directory up in the tree
pushd .
cd ..

# clone the repo in the parent directory and update it
echo "Cloning/Updating KinBot..."
git clone https://github.com/zadorlab/KinBot
cd KinBot || exit
git fetch origin
git checkout -b stable c9a915c9521a85fe989d328d825803661182e0d9 # This is the last stable installation.
# git pull origin master # commented to avoid pulling the latest version, since it is not stable.
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
