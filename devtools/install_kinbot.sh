# temporarily change directory to install software, and move one directory up in the tree
pushd .
cd ..

# clone the repo in the parent directory and update it
echo "Cloning/Updating KinBot..."
git clone https://github.com/zadorlab/KinBot
cd KinBot || exit
git fetch origin
git checkout master
git pull origin master
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
