/*
 * sigmoidParameterMapper.cpp
 *
 *  Created on: Feb 26, 2018
 *      Author: Gonzalo D. Maso Talou
 */

#include "SigmoidParameterMapper.h"

#include <math.h>

SigmoidParameterMapper::SigmoidParameterMapper(double min, double max) : AbstractParameterMapper() {
	this->min = min;
	this->max = max;
}

vector<double> SigmoidParameterMapper::map(vector<double> problemParameters){
	vector<double> kalmanParameters;
	for(vector<double>::iterator it = problemParameters.begin(); it != problemParameters.end(); ++it) {
		kalmanParameters.push_back( -log( (max-min) / (*it-min) - 1 ) );
	}

	return kalmanParameters;
}

vector<double> SigmoidParameterMapper::unmap(vector<double> kalmanParameters){
	vector<double> problemParameters;
	for(vector<double>::iterator it = kalmanParameters.begin(); it != kalmanParameters.end(); ++it) {
		problemParameters.push_back( 1 / (1+exp(-*it)) * (max - min) + min );
	}

	return problemParameters;
}
