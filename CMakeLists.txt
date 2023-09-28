# Specified the minimum required version of cmake-------------------------------
CMAKE_MINIMUM_REQUIRED(VERSION 3.5)
#-------------------------------------------------------------------------------

SET(CMAKE_BUILD_TYPE Debug)

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


# Directories that need to be included (containing headers)---------------------
#-------------------------------------------------------------------------------
INCLUDE_DIRECTORIES(
	${MPI_INCLUDE_PATH}
	${kalman_SOURCE_DIR}/io/
	${kalman_SOURCE_DIR}/mapping/
	${kalman_SOURCE_DIR}/
)

# Compilation flags-------------------------------------------------------------
#-------------------------------------------------------------------------------
SET(GCC_COMPILE_FLAGS "-std=c++11 -Wall -Wextra -O3")

SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COMPILE_FLAGS}" )

# List of all necessary source files--------------------------------------------
#-------------------------------------------------------------------------------
SET(kalman_SRCs
	./mapping/SigmoidParameterMapper.cpp
	./mapping/IdentityParameterMapper.cpp
	./mapping/ExponentialParameterMapper.cpp
	./mapping/CompositeParameterMapper.cpp
	./mapping/AbstractParameterMapper.cpp
	./io/ConfigurationFileReader.cpp
	./StaticROUKF.cpp
	./SigmaPointsGenerator.cpp
	./ROUKF.cpp
	./MappedROUKF.cpp
	./AbstractROUKF.cpp
)

SET( CMAKE_VERBOSE_MAKEFILE ON )

# Compile the library-----------------------------------------------------------
#-------------------------------------------------------------------------------
ADD_LIBRARY(${PROJECT_NAME} SHARED ${kalman_SRCs})
ADD_LIBRARY(${PROJECT_NAME}_static STATIC ${kalman_SRCs})
