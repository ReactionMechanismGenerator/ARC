# temporarily change directory to install software
pushd ..

# clone the repo in the parent directory
cd ..
git clone https://github.com/ReactionMechanismGenerator/TS-GCN

# create the environment
bash TS-GCN/devtools/create_env_cpu.sh

# Restore original directory
popd
