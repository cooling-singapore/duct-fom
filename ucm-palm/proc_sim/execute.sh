#!/bin/bash

export WD_PATH=`pwd`

if [ "$1" == "nscc" ]; then
  echo "Working directory: $WD_PATH"

  export PALM_PATH=$PALM_PATH_23
  export PATH=$PALM_PATH/build/bin:${PATH}
  echo "PALM_PATH: $PALM_PATH"
  echo "PATH: $PATH"

  if [[ -v "PALM_PATH" ]]; then
    echo "Checking PALM_PATH environment variable: found"
  else
    echo "Checking PALM_PATH environment variable: not found!"
    exit 2
  fi

  if ls $PALM_PATH > /dev/null 2>&1;then
    echo "Checking PALM4U path at '$PALM_PATH': found"
  else
    echo "Checking PALM4U path at '$PALM_PATH': not found!"
    exit 3
  fi

  if which palmrun > /dev/null 2>&1; then
    echo "Checking PALM4U executable 'palmrun': found"
  else
    echo "Checking PALM4U executable 'palmrun': not found!"
    exit 4
  fi

  echo "Run processor.py on $2"
  module load miniforge3/23.10
  conda activate $WD_PATH/venv
  python3.10 processor.py $2

  exit $?

else
  exit 1
fi
