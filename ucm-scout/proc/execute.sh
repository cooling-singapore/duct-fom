#!/bin/bash

if [ "$1" == "default" ]; then
  echo "Run processor.py on $2"
  export CASE_TEMPLATE_PATH=""
  source venv/bin/activate
  python3.10 processor.py $2
  exit $?

else
  exit 1
fi
