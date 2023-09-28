/*
 * CompositeParameterMapper.cpp
 *
 *  Created on: 26 de fev de 2018
 *      Author: Gonzalo D. Maso Talou
 */

#include "CompositeParameterMapper.h"
#include "iostream"

CompositeParameterMapper::CompositeParameterMapper(vector<int> paramsPerMapper, vector<AbstractParameterMapper *> mappers) : AbstractParameterMapper() {
	this->mappers = mappers;
	this->paramsPerMapper = paramsPerMapper;
}

CompositeParameterMapper::~CompositeParameterMapper() {
	// TODO Auto-generated destructor stub
}

vector<double> CompositeParameterMapper::map(vector<double> problemParameters) {
	vector<double> kalmanParameters;
	int currentParameter = 0;
	for(unsigned int i = 0; i < paramsPerMapper.size(); ++i) {
		vector<double> currentMapping(problemParameters.begin()+currentParameter, problemParameters.begin()+currentParameter+paramsPerMapper[i]);
		vector<double> currentParameters = mappers[i]->map(currentMapping);
		kalmanParameters.insert( kalmanParameters.end(), currentParameters.begin(), currentParameters.end() );
		currentParameter += paramsPerMapper[i];
	}

	return kalmanParameters;
}

vector<double> CompositeParameterMapper::unmap(vector<double> kalmanParameters) {
	vector<double> problemParameters;
	int currentParameter = 0;
	for(unsigned int i = 0; i < paramsPerMapper.size(); ++i) {
		vector<double> currentMapping(kalmanParameters.begin()+currentParameter, kalmanParameters.begin()+currentParameter+paramsPerMapper[i]);
		vector<double> currentParameters = mappers[i]->unmap(currentMapping);
		problemParameters.insert( problemParameters.end(), currentParameters.begin(), currentParameters.end() );
		currentParameter += paramsPerMapper[i];
	}

	return problemParameters;
}
