/*
 * CompositeParameterMapper.h
 *
 *  Created on: 26 de fev de 2018
 *      Author: Gonzalo D. Maso Talou
 */

#ifndef COMPOSITEPARAMETERMAPPER_H_
#define COMPOSITEPARAMETERMAPPER_H_

#include <vector>

#include "AbstractParameterMapper.h"

using namespace std;

/**
 * Implementation of the AbstractParameterMapper class using a composite design pattern.
 * It allows to specify a mapper for each parameter (or a subset of parameters).
 */
class CompositeParameterMapper: public AbstractParameterMapper {
	/**	Set of mapper that map the problem parameters in kalman parameters. */
	vector<AbstractParameterMapper *> mappers;
	/**	Set of the number of parameters treated by each mapper. */
	vector<int> paramsPerMapper;
public:
	/**
	 * Class constructor.
	 * @param paramsPerMapper Set of the number of parameters treated by each mapper.
	 * @param mappers Set of mapper that map the problem parameters in kalman parameters.
	 */
	CompositeParameterMapper(vector<int> paramsPerMapper, vector<AbstractParameterMapper *> mappers);
	/**
	 * Dummy destructor.
	 */
	virtual ~CompositeParameterMapper();

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

#endif /* COMPOSITEPARAMETERMAPPER_H_ */
