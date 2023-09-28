/*
 * AbstractPameterMapper.h
 *
 *  Created on: Feb 26, 2018
 *      Author: Gonzalo D. Maso Talou
 */

#ifndef ABSTRACTPARAMETERMAPPER_H_
#define ABSTRACTPARAMETERMAPPER_H_

#include <vector>

using namespace std;

/**
 * Abstract class that model the functions required for a parameters mapping function.
 * The mapping functions will map the problem parameters into a better-behaved space
 * of parameter for the kalman optimization. The map and unmap functions are inverse
 * one to other.
 */
class AbstractParameterMapper {
public:
	/**	Dummy constructor. */
	AbstractParameterMapper();

	/**
	 * Maps the problem parameters into the space of parameters where kalman filter optimize
	 * them.
	 * @param problemParameter Set of problem parameters.
	 * @return Set of kalman parameters
	 */
	virtual vector<double> map(vector<double> problemParameter) = 0;
	/**
	 * Maps the kalman parameters into the space of problem parameters.
	 * @param kalmanParameter Set of kalman parameters.
	 * @return Set of problem parameters.
	 */
	virtual vector<double> unmap(vector<double> kalmanParameter) = 0;
};

#endif /* ABSTRACTPARAMETERMAPPER_H_ */
