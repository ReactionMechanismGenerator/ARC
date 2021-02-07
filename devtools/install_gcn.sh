# temporarily change directory to install software
pushd ..

# clone the repo in the parent directory
git clone https://github.com/ReactionMechanismGenerator/TS-GCN

# create the environment
cd TS-GCN || exit
make conda_env

# Restore original directory
popd || exit
