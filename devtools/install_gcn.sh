# temporarily change directory to install software
pushd .
cd ../..

# clone the repo in the parent directory and update it
echo "Cloning/Updating GCN..."
git clone https://github.com/ReactionMechanismGenerator/TS-GCN
cd TS-GCN || exit
git fetch origin
git checkout master
git pull origin master

# Add to PYTHONPATH
echo "Adding GCN to PYTHONPATH..."
export PYTHONPATH=$PYTHONPATH:$(pwd)
echo 'export PYTHONPATH=$PYTHONPATH:'"$(pwd)" >> ~/.bashrc

# create the environment
echo "Creating the GCN environment..."
conda deactivate
make conda_env

# Restore the original directory
echo "Done installing GCN."
popd || exit
