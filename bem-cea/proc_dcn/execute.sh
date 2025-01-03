#!/bin/bash

if [ "$1" == "default" ]; then
  working_dir=`pwd`
  conda_env_name=duct-cea

  echo "Working directory: $working_dir"
  cd "$working_dir"

  echo "Activate micromamba"
  export MAMBA_ROOT_PREFIX=$working_dir/micromamba
  eval "$(./bin/micromamba shell hook -s posix)"

  micromamba activate $conda_env_name

  echo "Run processor.py on $2"
  python processor.py "$2"

  exit $?

else
  exit 1
fi
