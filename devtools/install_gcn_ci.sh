# temporarily change directory to install software
pushd ..

# clone the repo in the parent directory
git clone https://github.com/ReactionMechanismGenerator/TS-GCN
git checkout fdedb2ff2cb41f0b329fbc8b464593f4ccbe6c2a

# create the environment
cd TS-GCN/devtools
bash create_env_cpu.sh

# Restore original directory
popd
