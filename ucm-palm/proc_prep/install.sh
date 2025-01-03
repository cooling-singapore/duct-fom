#!/bin/bash

if [ "$1" == "default" ]; then
  if [ -f ~/.bashrc ]; then
    . ~/.bashrc
  fi

  echo "Install conda environment [`pwd`]"
  eval "$($HOME/anaconda3/bin/conda shell.bash hook)"
  conda env remove --name ucmpalm-prep-env
  conda create --yes --name ucmpalm-prep-env python=3.10
  conda activate ucmpalm-prep-env
  conda install --yes -c conda-forge gdal=3.8.3 rasterio=1.3.9 numpy=1.23.5 pyproj=3.6.1 h5py=3.11.0 pydantic=1.10.14 requests=2.28.2 scipy=1.14.0 affine=2.4.0 netCDF4=1.6.5

  echo "Install SaaS Middleware"
  rm -rf saas-middleware
  git clone https://github.com/cooling-singapore/saas-middleware
  pip3 install ./saas-middleware

  exit $?

else
  exit 1
fi
