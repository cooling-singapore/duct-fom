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

  echo "Install conda environment [`pwd`]"
  module load miniforge3/23.10
  conda env remove --prefix $WD_PATH/venv
  conda create --yes --prefix $WD_PATH/venv python=3.10
  conda activate $WD_PATH/venv
  conda install --yes -c conda-forge gdal==3.9.1 rasterio==1.3.10 numpy==2.0.0 pyproj==3.6.1 h5py==3.11.0 pydantic==1.10.14 requests==2.32.3 scipy==1.14.0 affine==2.4.0 netCDF4==1.7.1

  echo "Install SaaS Middleware"
  rm -rf saas-middleware
  git clone https://github.com/cooling-singapore/saas-middleware
  pip3 install ./saas-middleware

  exit $?

else
  exit 1
fi
