# temporarily change directory to install software
pushd ..

# clone the repo in the parent directory
git clone https://github.com/ReactionMechanismGenerator/TS-GCN
cd TS-GCN
git checkout update_devtools_GH_actions
# use `source activate` on TS-GCN
git checkout bba0dc66371c961005d17c00afa2411580e3bd26


# create the environment
# cd TS-GCN/devtools
cd devtools
bash create_env_cpu.sh

# Restore original directory
popd
