#!/bin/bash
#	Checks for Eclipse or NetBeans
if [ -d "${PWD}/Release" ]; then
	echo "Eclipse configuration files has been detected"
	sudo ln -sf ${PWD}/Release/libkalman.so /usr/local/lib/libkalman.so
	sudo ln -sf ${PWD}/Debug/libkalman.so /usr/local/lib/libkalman-dev.so
	else
	#	Checks for NetBeans
	if [ -d "${PWD}/dist" ]; then
		echo "Netbeans configuration files has been detected"
		sudo ln -sf ${PWD}/dist/Release/GNU-Linux/libkalman.so /usr/local/lib/libkalman.so
		sudo ln -sf ${PWD}/dist/Debug/GNU-Linux/libkalman.so /usr/local/lib/libkalman-dev.so
	else
	# Checks if there is a CMakeLists.txt file to compile with cmake
		if [ -f "${PWD}/CMakeLists.txt" ]; then
			echo "CMake compilation & linking will be performed."
			mkdir ${PWD}/build
			cd build
			cmake ..
			make clean
			make
			sudo ln -sf ${PWD}/build/libkalman.so /usr/local/lib/libkalman.so
			sudo ln -sf ${PWD}/build/libkalman_static.a /usr/local/lib/libkalman_static.a
		fi
	fi
fi

sudo mkdir /usr/local/include/kalman
sudo mkdir /usr/local/include/kalman/mapping
sudo mkdir /usr/local/include/kalman/io

sudo ln -sf ${PWD}/*.h /usr/local/include/kalman
sudo ln -sf ${PWD}/mapping/*.h /usr/local/include/kalman/mapping
sudo ln -sf ${PWD}/io/*.h /usr/local/include/kalman/io
sudo ldconfig
