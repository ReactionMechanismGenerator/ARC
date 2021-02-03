# temporarily change directory to install software
pushd ..

# clone the repo in the parent directory
git clone https://github.com/ReactionMechanismGenerator/TS-GCN
cd TS-GCN
git checkout update_devtools_GH_actions


# create the environment
# cd TS-GCN/devtools
cd devtools
bash create_env_cpu.sh

# Restore original directory
popd
