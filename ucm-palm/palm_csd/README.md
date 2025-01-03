# PALM Create Static Driver (palm_csd)

Processing tool for creating PALM Input Data Standard (PIDS) conform static driver files from rastered netCDF input.


# Installation

First make sure to satisfy the **Software Requirements**. On Debian-based Linux Distributions this can be achieved by the following command:

``` bash
sudo apt-get install python3-pip
```

Also some additional python dependencies are needed, which can be installed using `pip`. In case you wand to use a virtual environment for these dependencies, please make sure to create one first. Afterwards you can install the python dependencies by executing the following command:

``` bash
python3 -m pip install -r requirements.txt
```

Now the palmrun GUI can be installed with the following commands (please replace `<install-prefix>` with the desired installation directory):

``` bash
bash install -p <install-prefix>
export PATH=<install-prefix>/bin:${PATH}
```

The following optional command permanently adds this installation to your bash environment:

``` bash
echo 'export PATH=<install-prefix>/bin:${PATH}' >> ~/.bashrc
```

Type `bash install -h` to get all available option of the `install` script.


# Usage

After a successful installation, the executable `palm_csd` has been linked into the directory `<install-prefix>/bin`. In case you have installed the python dependencies inside a virtual environment, that environment needs to be active, whenever you wand to use `palm_csd`. Next, you need to create a configuration file and provide all necessary netCDF input files. To start the creation of a PIDS conform static driver file, simply type the following command into your terminal:

```bash
palm_csd <path/to/csd-config>
```

Don't forget to replace `<path/to/csd-config>` with the path to a valid configuration file. An example configuration file can be found in subdirectory [share/](share/csd_default.config).


# License

This repository is free and open-source software licensed under the [GNU GPLv3](LICENSE)
