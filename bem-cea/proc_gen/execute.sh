#!/bin/bash

if [ "$1" == "default" ]; then
  echo "Activate virtual environment"
  source venv/bin/activate

  echo "Run processor.py on $2"
  python3 processor.py "$2"

  exit $?

else
  exit 1
fi
