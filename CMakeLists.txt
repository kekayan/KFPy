# Specified the minimum required version of cmake-------------------------------
CMAKE_MINIMUM_REQUIRED(VERSION 3.5)
#-------------------------------------------------------------------------------
# set(CMAKE_PREFIX_PATH  "/hpc/knan475/miniconda3/envs/kfpy/")
SET(CMAKE_BUILD_TYPE Debug)
SET(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Project name, CMakeLists files in this project can
# refer to the root source directory of the project as ${kalman_SOURCE_DIR}
# and to the root binary directory of the project as ${kalman_BINARY_DIR}.
#-------------------------------------------------------------------------------
PROJECT(kalman)
SET(LIB_NAME ${PROJECT_NAME})

# Find ITK, and include the necessary headers-----------------------------------
#-------------------------------------------------------------------------------
#FIND_PACKAGE(ITK REQUIRED)
#INCLUDE(${ITK_USE_FILE})

# Find VTK, and include the necessary headers-----------------------------------
#-------------------------------------------------------------------------------
#SET (VTK_DIR /home/carlos/develop/VTK-7.1.0/build-release)
#FIND_PACKAGE(VTK REQUIRED)
#INCLUDE(${VTK_USE_FILE})

#SET (ParaView_DIR /home/carlos/develop/ParaView-v5.3.0/build-release)
#IF (NOT ParaView_BINARY_DIR)
#  FIND_PACKAGE(ParaView REQUIRED)
#  INCLUDE(${PARAVIEW_USE_FILE})
#ENDIF()

# Find MPI, and include the necessary headers-----------------------------------
#-------------------------------------------------------------------------------
FIND_PACKAGE(MPI REQUIRED)
FIND_PACKAGE(OpenMP)
FIND_PACKAGE(Python3 REQUIRED COMPONENTS Interpreter Development)
FIND_PACKAGE(pybind11 REQUIRED)
FIND_PACKAGE(BLAS REQUIRED)
FIND_PACKAGE(LAPACK REQUIRED)
FIND_PACKAGE(Armadillo REQUIRED)

# Directories that need to be included (containing headers)---------------------
#-------------------------------------------------------------------------------
INCLUDE_DIRECTORIES(
	${MPI_INCLUDE_PATH}
	${kalman_SOURCE_DIR}/io/
	${kalman_SOURCE_DIR}/mapping/
	${kalman_SOURCE_DIR}/
	${Python3_INCLUDE_DIRS}/
	${pybind11_INCLUDE_DIRS}/
	${ARMADILLO_INCLUDE_DIRS}
)

# Compilation flags-------------------------------------------------------------
#-------------------------------------------------------------------------------
SET(GCC_COMPILE_FLAGS "-std=c++11 -Wall -Wextra -O3 -fPIC")

SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS}" )
set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS})


# List of all necessary source files--------------------------------------------
#-------------------------------------------------------------------------------
SET(kalman_SRCs
	./mapping/SigmoidParameterMapper.cpp
	./mapping/IdentityParameterMapper.cpp
	./mapping/ExponentialParameterMapper.cpp
	./mapping/CompositeParameterMapper.cpp
	./mapping/AbstractParameterMapper.cpp
	# ./io/ConfigurationFileReader.cpp
	./StaticROUKF.cpp
	./SigmaPointsGenerator.cpp
	./ROUKF.cpp
	./MappedROUKF.cpp
	./AbstractROUKF.cpp
	./ROUKFPy.cpp
)

SET( CMAKE_VERBOSE_MAKEFILE ON )

# Compile the library-----------------------------------------------------------
#-------------------------------------------------------------------------------
ADD_LIBRARY(${PROJECT_NAME} SHARED ${kalman_SRCs})
ADD_LIBRARY(${PROJECT_NAME}_static STATIC ${kalman_SRCs})


# Add source files explicitly
add_library(roukf_lib STATIC
    ROUKF.cpp
    AbstractROUKF.cpp  # Make sure this is included
    SigmaPointsGenerator.cpp
)

# Create the Python module
pybind11_add_module(roukf_py ROUKFPy.cpp)

# Link against the static library
target_link_libraries(roukf_py PRIVATE 
    roukf_lib
    MPI::MPI_CXX
    ${ARMADILLO_LIBRARIES}  # If you're using Armadillo
    ${LAPACK_LIBRARIES}
    ${BLAS_LIBRARIES}
)

add_executable(roukf_example example.cpp)
target_link_libraries(roukf_example
    PRIVATE
    roukf_lib
    MPI::MPI_CXX
    ${ARMADILLO_LIBRARIES}
    ${LAPACK_LIBRARIES}
    ${BLAS_LIBRARIES}
)


add_executable(example2 example2.cpp)
target_link_libraries(example2
    PRIVATE
    roukf_lib
    MPI::MPI_CXX
    ${ARMADILLO_LIBRARIES}
    ${LAPACK_LIBRARIES}
    ${BLAS_LIBRARIES}
)