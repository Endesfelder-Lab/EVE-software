# Linux Installation Guide

To utilize EVE-software on Linux, you must install OpenEB first. This is the metavision SDK from prophesee. This installation is necessary for usage of EVE software features.

## OpenEB Installation
You must install OpenEB: [https://github.com/prophesee-ai/openeb](https://github.com/prophesee-ai/openeb)

## Prerequisites
Ensure you install all prerequisites mentioned in the OpenEB documentation, including the **pybind11** installation:

**Prerequisites:**
```bash
sudo apt update
sudo apt -y install apt-utils build-essential software-properties-common wget unzip curl git cmake
sudo apt -y install libopencv-dev libboost-all-dev libusb-1.0-0-dev libprotobuf-dev protobuf-compiler
sudo apt -y install libhdf5-dev hdf5-tools libglew-dev libglfw3-dev libcanberra-gtk-module ffmpeg
sudo apt -y install python3.9-dev
```

**Pybind11:**
```bash
wget https://github.com/pybind/pybind11/archive/v2.11.0.zip
unzip v2.11.0.zip
cd pybind11-2.11.0/
mkdir build && cd build
cmake .. -DPYBIND11_TEST=OFF
cmake --build .
sudo cmake --build . --target install
```

## Compilation Steps
Follow these specific steps to compile:

1. **Create and open the build directory:**
   ```bash
   mkdir -p build && cd build
   ```

2. **Generate the makefiles using CMake:**

Important to specify the python executable, EVE is 3.9
   ```bash
   cmake .. -DBUILD_TESTING=OFF -DPython3_EXECUTABLE=/usr/bin/python3.9
   ```

3. **Compile:**
   ```bash
   cmake --build . --config Release -- -j 4
   ```

Make sure everything compiles properly. 
   
# Running EVE

Make sure both EVE and OpenEB are in the same directory. If not, adjust the path to the setup env accordingly.

Go to the base directory of EVE
```bash
cd /path/to/EVE-software
```

Run the main GUI.py while linking the necessary env from openeb.
```bash
../openeb/build/utils/scripts/setup_env.sh python3.9 GUI.py
```

This should?!? work.
