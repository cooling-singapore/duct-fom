#!/bin/bash

export WD_PATH=`pwd`

if [ "$1" == "nscc" ]; then
  if [[ -v "WRF_PATH" ]]; then
    echo "Checking WRF_PATH environment variable: found"
  else
    echo "Checking WRF_PATH environment variable: not found!"
    exit 2
  fi

  export WRF_DIR=$WRF_PATH/mWRF_SG-A
  if ls $WRF_DIR > /dev/null 2>&1;then
    echo "Checking WRF_DIR path at '$WRF_DIR': found"
  else
    echo "Checking WRF_DIR path at '$WRF_DIR': not found!"
    exit 3
  fi

  export WPS_DIR=$WRF_PATH/WPS-A
  if ls $WPS_DIR > /dev/null 2>&1;then
    echo "Checking WPS_DIR path at '$WPS_DIR': found"
  else
    echo "Checking WPS_DIR path at '$WPS_DIR': not found!"
    exit 4
  fi

  echo "Export environment variables"
  export GEOGRID_EXE=$WPS_DIR/geogrid.exe
  export UNGRIB_EXE=$WPS_DIR/ungrib.exe
  export METGRID_EXE=$WPS_DIR/metgrid.exe
  export REAL_EXE=$WRF_DIR/run/real.exe

  # loop through the array and check if each file exists
  executables=("GEOGRID_EXE" "UNGRIB_EXE" "METGRID_EXE" "REAL_EXE")
  for var_name in "${executables[@]}"; do
      file_path="${!var_name}"  # Get the value of the environment variable

      if [ -e "$file_path" ]; then
          echo "Checking $var_name path at '$file_path': found."
      else
          echo "Checking $var_name path at '$file_path': not found!"
          exit 5
      fi
  done

  echo "Export environment variables"
  export DATA_PATH=$HOME/scratch/wrf/data
  export MODEL_PATH=`pwd`/../model
  export TEMPLATE_VERSION='a'
  export USE_NETCDF_VERSION=4
  export USE_PBS=yes

  echo "Run processor.py on $2"
  module load miniforge3/23.10
  conda activate $WD_PATH/venv
  python processor.py $2


elif [ "$1" == "gce-ubuntu-20.04" ]; then
  echo "Export environment variables"
  export WRF_DIR=$HOME/mWRF_SG
  export WPS_DIR=$HOME/WPS
  export GEOGRID_EXE=$WPS_DIR/geogrid.exe
  export UNGRIB_EXE=$WPS_DIR/ungrib.exe
  export METGRID_EXE=$WPS_DIR/metgrid.exe
  export REAL_EXE=$WRF_DIR/run/real.exe

  # loop through the array and check if each file exists
  executables=("GEOGRID_EXE" "UNGRIB_EXE" "METGRID_EXE" "REAL_EXE")
  for var_name in "${executables[@]}"; do
      file_path="${!var_name}"  # Get the value of the environment variable

      if [ -e "$file_path" ]; then
          echo "Checking $var_name path at '$file_path': found."
      else
          echo "Checking $var_name path at '$file_path': not found!"
          exit 5
      fi
  done

  # >>> conda initialize >>>
  # !! Contents within this block are managed by 'conda init' !!
  __conda_setup="$("$HOME/miniconda3/bin/conda" 'shell.bash' 'hook' 2> /dev/null)"
  if [ $? -eq 0 ]; then
      eval "$__conda_setup"
  else
      if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
          . "$HOME/miniconda3/etc/profile.d/conda.sh"
      else
          export PATH="$HOME/miniconda3/bin:$PATH"
      fi
  fi
  unset __conda_setup
  # <<< conda initialize <<<

  export DATA_PATH=$HOME/data
  export MODEL_PATH=`pwd`/../model
  export USE_NETCDF_VERSION=3
  export USE_PBS=no

  echo "Run processor.py on $2"
  source activate $WD_PATH/venv
  python processor.py $2

  exit $?

else
  exit 1

fi
