#!/bin/bash

if [ "$1" == "default" ]; then
  echo "Working directory: [`pwd`]"

  echo "Install python virtual environment [`pwd`]"
  rm -rf venv
  python3.10 -m venv venv

  echo "Install python dependencies [`pwd`]"
  source venv/bin/activate

  pip install git+https://github.com/cooling-singapore/saas-middleware
  pip install gmsh==4.11.1 rasterio==1.3.11 python-dotenv==1.1.0 numpy==1.23.5 pyproj==3.7.1 pyvista==0.45.2 scipy==1.15.3 h5py==3.14.0

  exit $?

else
  exit 1
fi
