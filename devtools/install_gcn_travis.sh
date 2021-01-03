# temporarily change directory to install software
pushd ..

# clone the repo in the parent directory
git clone https://github.com/ReactionMechanismGenerator/TS-GCN
cd TS-GCN/devtools

# create the environment
bash create_env_cpu.sh

# Restore original directory
popd
