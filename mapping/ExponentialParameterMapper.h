/*
 * ExponentialParameterMapper.h
 *
 *  Created on: Feb 26, 2018
 *      Author: Gonzalo D. Maso Talou
 */

#ifndef EXPONENTIALPARAMETERMAPPER_H_
#define EXPONENTIALPARAMETERMAPPER_H_

#include "AbstractParameterMapper.h"

/**
 * Implementation of the AbstractParameterMapper class that map the problem parameters
 * into the kalman parameters using a log function.
 */
class ExponentialParameterMapper: public AbstractParameterMapper {
public:
	/**
	 * Dummy constructor.
	 */
	ExponentialParameterMapper();

	/**
	 * Maps the problem parameters into the space of parameters where kalman filter optimize
	 * them.
	 * @param problemParameters Set of problem parameters.
	 * @return Set of kalman parameters
	 */
	vector<double> map(vector<double> problemParameters);
	/**
	 * Maps the kalman parameters into the space of problem parameters.
	 * @param kalmanParameters Set of kalman parameters.
	 * @return Set of problem parameters.
	 */
	vector<double> unmap(vector<double> kalmanParameters);
};

#endif /* EXPONENTIALPARAMETERMAPPER_H_ */
