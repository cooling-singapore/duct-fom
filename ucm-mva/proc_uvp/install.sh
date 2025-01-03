#!/bin/bash

if [ "$1" == "gce-ubuntu-22.04" ]; then
  echo "Working directory: [`pwd`]"

  echo "Install saas-middleware-sdk [`pwd`]"
  rm -rf saas-middleware-sdk
  git clone https://github.com/cooling-singapore/saas-middleware-sdk

  echo "Install python virtual environment [`pwd`]"
  rm -rf venv
  python3.10 -m venv venv

  echo "Install python dependencies [`pwd`]"
  source venv/bin/activate
  pip3 install --upgrade pip
  pip3 install ./saas-middleware-sdk
  pip3 install -r ../requirements.txt

  exit $?

else
  exit 1
fi
