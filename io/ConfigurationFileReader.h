/*
 * ConfigurationReader.h
 *
 *  Created on: Feb 27, 2018
 *      Author: Gonzalo D. Maso Talou
 */

#ifndef CONFIGURATIONFILEREADER_H_
#define CONFIGURATIONFILEREADER_H_

#include "../AbstractROUKF.h"
#include <iostream>

using namespace std;

/**
 * File reader class that after parsing the target file generates a Kalman filter
 * instance with appropriate mappers.
 */
class ConfigurationFileReader {
	/**	Configuration file path. */
	string filename;
	/**	Observations loaded from configuration file */
	vector<double> observations;
	/**	Quantity of internal states. */
	int nStates;
	/**	Quantity of parameters. */
	int nParameters;
	/**	Quantity of observations. */
	int nObservations;
	/**	Singleton attribute of the generated kalman filter. */
	AbstractROUKF *roukfModel;

public:

	/**	Types of filter that can be explicit in the configuration file. */
	enum FILTER_TYPES {MODEL_ROUKF = 0, MODEL_MAPPED_ROUKF = 1};
	/**	Types of mappers that can be explicit in the configuration file. */
	enum MAPPER_TYPES {IDENTITY = 0, EXPONENTIAL = 1, SIGMOIDAL = 2};
	/**
	 * Reader constructor.
	 */
	ConfigurationFileReader(string filename);
	/**
	 * Returns the kalman instance associated with the configuration file.
	 * @return Kalman model.
	 */
	AbstractROUKF *getInstance();

	/**
	 * Returns the observations read from the configuration file after executing getInstance.
	 * @return	Observations vector.
	 */
	vector<double> getObservations();

	/**
	 * Getter for @p nParameters
	 * @return @p nParameters
	 */
	int getNParameters() const;

	/**
	 * Getter for @p nStates
	 * @return @p nStates
	 */
	int getNStates() const;

	/**
	 * Getter for @p nObservations
	 * @return @p nObservations
	 */
	int getNObservations() const;
};

#endif /* CONFIGURATIONFILEREADER_H_ */
