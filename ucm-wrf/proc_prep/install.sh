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

  echo "Install conda environment [`pwd`]"
  module load miniforge3/23.10
  conda env remove --prefix $WD_PATH/venv
  conda create --yes --prefix $WD_PATH/venv python=3.10
  conda activate $WD_PATH/venv
  conda install --yes -c conda-forge gdal==3.9.1 rasterio==1.3.10 numpy=2.0.0 h5py=3.11.0 pydantic=1.10.14 netCDF4==1.7.1

  echo "Install SaaS Middleware"
  rm -rf saas-middleware
  git clone https://github.com/cooling-singapore/saas-middleware
  pip install ./saas-middleware

  exit $?

elif [ "$1" == "gce-ubuntu-20.04" ]; then
  echo "Check if mWRF_SG and WPS are already installed"
  GEOGRID_EXE="$HOME/WPS/geogrid.exe"
  METGRID_EXE="$HOME/WPS/metgrid.exe"
  UNGRIB_EXE="$HOME/WPS/ungrib.exe"
  REAL_EXE="$HOME/mWRF_SG/main/real.exe"
  if [ -f $GEOGRID_EXE ] && [ -f $METGRID_EXE ] && [ -f $UNGRIB_EXE ] && [ -f $REAL_EXE ];then
    echo "Looks like mWRF_SG and WPS are already installed. Skipping install."

  else
    echo "Update system"
    sudo apt-get -y update
    sudo apt-get -y upgrade

    echo "Install dependencies"
    sudo apt-get -y install libpq-dev gdal-bin libgdal-dev gfortran csh m4 netcdf-bin

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

    echo "Make directories"
    mkdir $BUILD_LIBRARIES

    echo "Install NetCDF"
    export CC=gcc
    export CXX=g++
    export FC=gfortran
    export FCFLAGS=-m64
    export F77=gfortran
    export FFLAGS=-m64

    cd $BUILD_LIBRARIES
    curl --output netcdf-4.1.3.tar.gz https://www2.mmm.ucar.edu/wrf/OnLineTutorial/compile_tutorial/tar_files/netcdf-4.1.3.tar.gz
    tar xzvf netcdf-4.1.3.tar.gz
    rm -f netcdf-4.1.3.tar.gz
    cd netcdf-4.1.3
    ./configure --prefix=$BUILD_LIBRARIES/netcdf --disable-dap --disable-netcdf-4 --disable-shared
    make
    make install

    echo "Install MPICH"
    cd $BUILD_LIBRARIES
    curl --output mpich-3.0.4.tar.gz https://www2.mmm.ucar.edu/wrf/OnLineTutorial/compile_tutorial/tar_files/mpich-3.0.4.tar.gz
    tar xzvf mpich-3.0.4.tar.gz
    rm -f mpich-3.0.4.tar.gz
    cd mpich-3.0.4
    ./configure --prefix=$BUILD_LIBRARIES/mpich
    make
    make install

    echo "Install Jasper"
    cd $BUILD_LIBRARIES
    curl --output jasper-1.900.1.tar.gz https://www2.mmm.ucar.edu/wrf/OnLineTutorial/compile_tutorial/tar_files/jasper-1.900.1.tar.gz
    tar xzvf jasper-1.900.1.tar.gz
    rm -f jasper-1.900.1.tar.gz
    cd jasper-1.900.1
    ./configure --prefix=$BUILD_LIBRARIES/grib2
    make
    make install

    echo "Install Zlib"
    cd $BUILD_LIBRARIES
    curl --output zlib-1.2.7.tar.gz https://www2.mmm.ucar.edu/wrf/OnLineTutorial/compile_tutorial/tar_files/zlib-1.2.7.tar.gz
    tar xzvf zlib-1.2.7.tar.gz
    rm -f zlib-1.2.7.tar.gz
    cd zlib-1.2.7
    ./configure --prefix=$BUILD_LIBRARIES/grib2
    make
    make install

    echo "Install LibPNG"
    export CPPFLAGS="-I $BUILD_LIBRARIES/grib2/include"
    export LDFLAGS="-L$BUILD_LIBRARIES/grib2/lib"

    cd $BUILD_LIBRARIES
    curl --output libpng-1.2.50.tar.gz https://www2.mmm.ucar.edu/wrf/OnLineTutorial/compile_tutorial/tar_files/libpng-1.2.50.tar.gz
    tar xzvf libpng-1.2.50.tar.gz
    rm -f libpng-1.2.50.tar.gz
    cd libpng-1.2.50
    ./configure --prefix=$BUILD_LIBRARIES/grib2
    make
    make install

    export CPPFLAGS=
    export LDFLAGS=

    echo "Install HDF5"
    cd $BUILD_LIBRARIES
    curl --output hdf5-1.10.5.tar.gz https://support.hdfgroup.org/ftp/HDF5/current/src/hdf5-1.10.5.tar.gz
    tar xzvf hdf5-1.10.5.tar.gz
    rm -f hdf5-1.10.5.tar.gz
    cd hdf5-1.10.5
    ./configure --prefix=$BUILD_LIBRARIES/grib2 CC=mpicc CXX=mpic++ FC=mpif90 F90=mpif90 --enable-parallel --disable-shared --enable-fortran2003 --enable-fortran --with-zlib=$BUILD_LIBRARIES/grib2
    make
    make install
    sudo ldconfig

    echo "Install mWRF_SG"
    export WRF_DIR=$HOME/mWRF_SG

    cd $HOME
    git clone https://github.com/cooling-singapore/mWRF_SG.git

    cd $WRF_DIR
    ./clean -a
    ./configure <<< $'34\r1\r'
    ./compile em_real > log.compile

    echo "Install WPS (Release 4.2)"
    export WPS_HOME=$HOME/WPS

    cd $HOME
    git clone https://github.com/wrf-model/WPS

    cd $WPS_HOME
    git checkout b98f858
    ./clean
    ./configure <<< $'3\r'
    ./compile >& log.compile

  fi

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

  echo "Install conda environment [`pwd`]"
  conda env remove --prefix $WD_PATH/venv
  conda create --yes --prefix $WD_PATH/venv python=3.10
  conda activate $WD_PATH/venv
  conda install --yes -c conda-forge gdal rasterio numpy h5py pydantic scipy

  echo "Install SaaS Middleware"
  rm -rf saas-middleware
  git clone https://github.com/cooling-singapore/saas-middleware
  pip install ./saas-middleware

  conda deactivate

  exit $?

else
  exit 1
fi