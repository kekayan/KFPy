/*
 * ExponentialParameterMapper.cpp
 *
 *  Created on: Feb 26, 2018
 *      Author: Gonzalo D. Maso Talou
 */

#include "ExponentialParameterMapper.h"
#include <math.h>

ExponentialParameterMapper::ExponentialParameterMapper() : AbstractParameterMapper(){
}

vector<double> ExponentialParameterMapper::map(vector<double> problemParameters){
	vector<double> kalmanParameters;
	for(vector<double>::iterator it = problemParameters.begin(); it != problemParameters.end(); ++it) {
		kalmanParameters.push_back(log(*it));
	}

	return kalmanParameters;
}

vector<double> ExponentialParameterMapper::unmap(vector<double> kalmanParameters){
	vector<double> problemParameters;
	for(vector<double>::iterator it = kalmanParameters.begin(); it != kalmanParameters.end(); ++it) {
		problemParameters.push_back(exp(*it));
	}

	return problemParameters;
}
