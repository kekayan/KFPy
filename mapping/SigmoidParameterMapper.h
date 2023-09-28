/*
 * sigmoidParameterMapper.h
 *
 *  Created on: Feb 26, 2018
 *      Author: Gonzalo D. Maso Talou
 */

#ifndef SIGMOIDPARAMETERMAPPER_H_
#define SIGMOIDPARAMETERMAPPER_H_

#include "AbstractParameterMapper.h"

/**
 * Implementation of the AbstractParameterMapper class that map the problem parameters
 * into the kalman parameters using a sigmoid function with a user preseted range.
 */
class SigmoidParameterMapper: public AbstractParameterMapper {
	/** Minimum value in the problem parameters range. */
	double min;
	/** Maximum value in the problem parameters range. */
	double max;
public:
	/**
	 * Constructor that sets the range of mapping to (min,max)
	 * @param min	Minimum value in the problem parameters range.
	 * @param max	Maximum value in the problem parameters range.
	 */
	SigmoidParameterMapper(double min, double max);

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

#endif /* SIGMOIDPARAMETERMAPPER_H_ */
