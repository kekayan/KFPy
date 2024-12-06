rm -rf build
rm -rf roukf.egg-info
mkdir build
cd build
cmake ..
make
cd ..
cp build/*.so python/roukf/

# cd python
# conda activate roukf && pip install .

