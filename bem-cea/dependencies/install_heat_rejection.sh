#!/bin/bash

working_dir=$1
conda_env_name=duct-cea

cd $working_dir

echo "Activate micromamba"
export MAMBA_ROOT_PREFIX=$working_dir/micromamba
eval "$(./bin/micromamba shell hook -s posix)"

echo "Install heat rejection plugin"
git -C $working_dir clone https://github.com/cooling-singapore/cea-heat-rejection-plugin.git
cd "$working_dir"/cea-heat-rejection-plugin && git checkout ea96ba3

micromamba activate $conda_env_name
python -m pip install -e "$working_dir"/cea-heat-rejection-plugin

echo "Add heat rejection plugin to config"
# TODO: Apply plugin in workflow instead
cea-config add-plugins cea_heat_rejection_plugin.heat_rejection.HeatRejectionPlugin
