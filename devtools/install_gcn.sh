# temporarily change directory to install software
pushd ..

# clone the repo in the parent directory
cd ..
git clone https://github.com/ReactionMechanismGenerator/TS-GCN

# create the environment
bash TS-GCN/create_env.sh

# Restore original directory
popd
