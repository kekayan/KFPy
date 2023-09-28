/*
 * IdentityParameterMapper.h
 *
 *  Created on: Feb 26, 2018
 *      Author: Gonzalo D. Maso Talou
 */

#ifndef IDENTITYPARAMETERMAPPER_H_
#define IDENTITYPARAMETERMAPPER_H_

#include "AbstractParameterMapper.h"

/**
 * Implementation of the AbstractParameterMapper class that maps the problem parameters
 * directly in the kalman parameter space without modifications.
 */
class IdentityParameterMapper: public AbstractParameterMapper {
public:
	/**
	 * Dummy constructor.
	 */
	IdentityParameterMapper();

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

#endif /* IDENTITYPARAMETERMAPPER_H_ */
