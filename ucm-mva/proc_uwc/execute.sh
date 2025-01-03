#!/bin/bash

if [ "$1" == "gce-ubuntu-22.04" ]; then

  echo "Run processor.py on $2"
  source venv/bin/activate
  python3.10 processor.py $2
  exit $?

else
  exit 1
fi
