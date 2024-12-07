# Reduce-order Unscent Kalman Filter

## Dependences

### Build tools

1. Install build tools using
```bash
sudo apt-get install build-essential
```
2. Install libconfig using
```
sudo apt-get install libconfig++-dev
```


### Armadillo library 6.2
1.  If not present already, install LAPACK, Boost and BLAS  with
```bash
sudo apt-get install liblapack-dev
sudo apt-get install libblas-dev
sudo apt-get install libboost-dev
```
2. Install Armadillo using 
```bash
sudo apt-get install libarmadillo-dev
```

### MPI
1. Install MPI using
```bash
sudo apt-get update && sudo apt-get install infiniband-diags ibverbs-utils \
     libibverbs-dev libfabric1 libfabric-dev libpsm2-dev -y
sudo apt-get install openmpi-bin openmpi-common libopenmpi-dev libgtk2.0-dev
sudo apt-get install librdmacm-dev libpsm2-dev
```
2. Check if MPI is installed and version using
```bash
which mpicc
which mpiexec
```
```bash
mpicc --version
mpiexec –version
```

### CMake
1. Install CMake using
```bash
sudo apt-get install cmake
```

### Doxygen
1. Install Doxygen using
```bash
sudo apt-get install doxygen
```

## Generate Documentation
1. To generate documentation, run the following command in the project root directory
```bash
doxygen Doxyfile
```
2. The documentation will be generated in the `doc` folder in the project root directory

## Build Instructions
1. Clone the repository using
```bash
git clone 
```
2. Go to the project root directory
```bash
cd <path-to-project-root>
```
3. Create a build directory
```bash
mkdir build
cd build
```
4. Run CMake
```bash
cmake ..
```
5. Build the project
```bash
make
```


## Install Instructions

1. Run the following command in the build directory
```bash
chmod +x install.sh
./install.sh
```

## Python Bindings

1. Install pybind11 in your virtual env using
```bash
pip install pybind11
```

2. After building the project, copy the generated shared library to the python directory

```bash
cp *.so ../python/roukf-py/
```

3. Install the python package using
```bash
pip install -e .
```

4. Run the python script using
```bash
python usage.py
```