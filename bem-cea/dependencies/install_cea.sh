#!/bin/bash

working_dir=$1
conda_env_name=duct-cea

echo "Fetching micromamba"
cd $working_dir
# Require bzip2 to extract archive
sudo apt-get -y install bzip2
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/1.0.0 | tar -xvj bin/micromamba

echo "Activate micromamba"
export MAMBA_ROOT_PREFIX=$working_dir/micromamba
mkdir $MAMBA_ROOT_PREFIX
eval "$(./bin/micromamba shell hook -s posix)"

echo "Cloning CEA"
git -C $working_dir clone https://github.com/cooling-singapore/CEA_for_DUCT.git
cd "$working_dir"/CEA_for_DUCT && git checkout 4bc3e0dc98978ceb3da02d2259f4576d6f7c8172

echo "Installing CEA dependencies in conda environment $conda_env_name"
micromamba create -n $conda_env_name --file $working_dir/CEA_for_DUCT/conda-lock.yml -y
micromamba activate $conda_env_name
# For pythonOCC
sudo apt-get -y install libgl1

# Proc dependencies
python3 -m pip install h5py geojson pandas pyyaml jinja2

echo "Installing CEA in conda environment $conda_env_name"
python3 -m pip install "$working_dir"/CEA_for_DUCT
