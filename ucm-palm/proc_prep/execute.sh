#!/bin/bash

if [ "$1" == "default" ]; then
  if [ -f ~/.bashrc ]; then
    . ~/.bashrc
  fi

  echo "Run processor.py on $2"
  eval "$($HOME/anaconda3/bin/conda shell.bash hook)"
  conda activate ucmpalm-prep-env
  python3.10 processor.py $2

  exit $?

else
  exit 1
fi
