# temporarily change directory to install software
pushd ..

# clone the repo in the parent directory
git clone https://github.com/ReactionMechanismGenerator/TS-GCN

# create the environment
cd TS-GCN || exit
cd devtools || exit
bash create_env_cpu.sh

# Restore original directory
popd || exit
