# temporarily change directory to install software, and move one directory up in the tree
pushd .
cd ..

# clone the repo in the parent directory
echo "Cloning/Updating AutoTST..."
git clone https://github.com/ReactionMechanismGenerator/AutoTST
cd AutoTST || exit
git fetch origin
git checkout main
git pull origin main

# Add to PYTHONPATH
echo "Adding AutoTST to PYTHONPATH..."
export PYTHONPATH=$PYTHONPATH:$(pwd)
echo 'export PYTHONPATH=$PYTHONPATH:'"$(pwd)" >> ~/.bashrc
echo $PYTHONPATH

# create the environment
echo "Creating the AutoTST environment..."
source ~/.bashrc
conda env create -f environment.yml
conda activate tst_env
conda install -c anaconda yaml -y
source ~/.bashrc

# Restore the original directory
cd ../ARC || exit
echo "Done installing AutoTST."
popd || exit
