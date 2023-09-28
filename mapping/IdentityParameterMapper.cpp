/*
 * IdentityParameterMapper.cpp
 *
 *  Created on: Feb 26, 2018
 *      Author: Gonzalo D. Maso Talou
 */

#include "IdentityParameterMapper.h"

IdentityParameterMapper::IdentityParameterMapper() : AbstractParameterMapper(){
}

vector<double> IdentityParameterMapper::map(vector<double> problemParameters){
	return problemParameters;
}

vector<double> IdentityParameterMapper::unmap(vector<double> kalmanParameters){
	return kalmanParameters;
}
