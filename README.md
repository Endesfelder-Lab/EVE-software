# Eve - General-purpose software for eve-SMLM localization
![](Eve.png)
## Overview
Eve is a software package that provides a plethora of options to localize emitters from single molecule localization microscopy (SMLM) experiments performed on a MetaVision event-based sensor (eveSMLM).
## Version
The `main` branch contains the latest running version of the software. The latest developments are stored in the `develop` branch.
## Installation
This software was tested under Linux Ubuntu 20.04 and Windows 10. Besides the drift correction module that contains and uses pre-compiled dll-files and is therefore only running on Windows, everthing is running under Linux and Windows.

The software requires Python 3.9.18.

### Installation with `virtualenv`
Set-up virtual environment wiith virtualenv and install the required packages with pip. Replace `PATH_TO_PYTHON` by your python path, e.g. `/usr/bin`
```bash
$ virtualenv -p PATH_TO_PYTHON/python3.9.18 env_name
$ source env_name/bin/activate
$ pip install -r requirements.txt
```

### Installation with `conda`
```bash
cd Eve
conda env create -f environment_droplet.yml
conda activate Eve
```


## Getting started
