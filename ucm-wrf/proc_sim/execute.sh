#!/bin/bash

export WD_PATH=`pwd`

if [ "$1" == "nscc" ]; then
  echo "Working directory: $WD_PATH"

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

  echo "Export environment variables"
  export MODEL_PATH=`pwd`/../model
  export TEMPLATE_VERSION='a'
  export WRF_EXE=$WRF_DIR/run/wrf.exe
  export USE_NETCDF_VERSION=4
  export USE_PBS=yes

  # loop through the array and check if each file exists
  executables=("WRF_EXE")
  for var_name in "${executables[@]}"; do
      file_path="${!var_name}"  # Get the value of the environment variable

      if [ -e "$file_path" ]; then
          echo "Checking $var_name path at '$file_path': found."
      else
          echo "Checking $var_name path at '$file_path': not found!"
          exit 5
      fi
  done

  export SCRATCH_PATH=$HOME/scratch/wrf
  if ls $SCRATCH_PATH > /dev/null 2>&1; then
    echo "Checking SCRATCH_PATH: found"
  else
    echo "Checking SCRATCH_PATH: not found!"
    exit 5
  fi

  echo "Run processor.py on $2 $3"
  module load miniforge3/23.10
  conda activate $WD_PATH/venv
  python processor.py $2 $3

  exit $?

elif [ "$1" == "gce-ubuntu-20.04" ]; then
  echo "Export environment variables"
  export BUILD_LIBRARIES=$HOME/dependencies
  export WRFIO_NCD_LARGE_FILE_SUPPORT=1
  export HDF5=$BUILD_LIBRARIES/grib2
  export LIBPNG=$BUILD_LIBRARIES/grib2
  export ZLIB=$BUILD_LIBRARIES/grib2
  export PATH=$BUILD_LIBRARIES/netcdf/bin:$PATH
  export PATH=$BUILD_LIBRARIES/mpich/bin:$PATH
  export NETCDF=$BUILD_LIBRARIES/netcdf
  export JASPERLIB=$BUILD_LIBRARIES/grib2/lib
  export JASPERINC=$BUILD_LIBRARIES/grib2/include

  export WRF_DIR=$HOME/mWRF_SG
  export WRF_EXE="$WRF_DIR/main/wrf.exe"
  export MODEL_PATH=`pwd`/../model
  export USE_NETCDF_VERSION=3
  export USE_PBS=no

  # loop through the array and check if each file exists
  executables=("WRF_EXE")
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

  echo "Run processor.py on $2 $3"
  source activate $WD_PATH/venv
  python processor.py $2 $3

  exit $?

else
  exit 1
fi
