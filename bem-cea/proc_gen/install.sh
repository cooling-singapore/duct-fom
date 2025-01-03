#!/bin/bash

if [ "$1" == "default" ]; then
  echo "Create python virtual environment"
  python3 -m venv ./venv

  echo "Activate virtual environment"
  source ./venv/bin/activate

  echo "Install python dependencies"
  python3 -m pip install -r ./requirements.txt

  exit $?
else
  exit 1
fi
