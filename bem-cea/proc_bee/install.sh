#!/bin/bash

if [ "$1" == "default" ]; then
  working_dir=`pwd`
  echo "Working directory: $working_dir"

  dependencies_dir="$( cd "$( dirname "${BASH_SOURCE[0]}")"/../dependencies && pwd )"

  source $dependencies_dir/install_cea.sh $working_dir
  source $dependencies_dir/install_heat_rejection.sh $working_dir

  exit $?
else
  exit 1
fi
