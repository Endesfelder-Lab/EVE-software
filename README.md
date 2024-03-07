# Eve - General-purpose software for eve-SMLM localization
![](Eve.png)
## Overview
Eve is a software package that provides a plethora of options to localize emitters from single molecule localization microscopy (SMLM) experiments performed on a MetaVision event-based sensor (eveSMLM).
## Version
The `main` branch contains the latest running version of the software. The latest developments are stored in the `develop` branch.
## Installation
This software was tested under Linux Ubuntu 20.04 and Windows 10. Besides the drift correction module that contains and uses pre-compiled dll-files and is therefore only running on Windows, everthing is running under Linux and Windows.

The software requires Python 3.9.18.
### Installation instructions
1. First download Eve or clone it to your local repository using git:
    ```bash
    git clone https://github.com/Endesfelder-Lab/Eve.git
    ```
2. Install required python dependencies
    #### With `virtualenv`
    Replace `PYTHON_PATH` by your python path, e.g. `/usr/bin` and `ENVIRONMENT_PATH` by the path to the virtual environments on your machine and follow the instructions below:
    ```bash
    virtualenv -p PYTHON_PATH/python3.9.18 ENVIRONMENT_PATH/Eve
    source ENVIRONMENT_PATH/Eve/bin/activate
    pip install -r requirements.txt
    ```
    #### With `conda`
   Simply follow the instructions below:
    ```bash
    cd Eve
    conda env create -f environment_eve.yml
    conda activate Eve
    ```
3. Optional:
   
    Eve can read and process event-based data in `.npy` and `.hdf5` format. Additionally the `.raw` format of [Prophesee](https://www.prophesee.ai/) can be used. If you have `.raw` data that you want to analyze you need to install the [Metavision SDK from Prophesee](https://docs.prophesee.ai/stable/installation/index.html). 

## Getting started
