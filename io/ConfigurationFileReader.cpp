/*
 * ConfigurationReader.cpp
 *
 *  Created on: Feb 27, 2018
 *      Author: Gonzalo D. Maso Talou
 */

#include "ConfigurationFileReader.h"

#include "../MappedROUKF.h"
#include "../ROUKF.h"

#include "../SigmaPointsGenerator.h"

#include "../mapping/AbstractParameterMapper.h"
#include "../mapping/IdentityParameterMapper.h"
#include "../mapping/ExponentialParameterMapper.h"
#include "../mapping/SigmoidParameterMapper.h"

#include <libconfig.h++>

using namespace libconfig;

ConfigurationFileReader::ConfigurationFileReader(string filename) {
	this->filename = filename;
	this->nStates = 0;
	this->nParameters = 0;
	this->nObservations = 0;
	this->roukfModel = NULL;
}

AbstractROUKF* ConfigurationFileReader::getInstance() {

	if(roukfModel)
		return roukfModel;

	Config config;

	try {
		config.readFile(filename.c_str());
	} catch (const FileIOException &fioex) {
		std::cerr << "I/O error while reading configuration file for ROUKF." << std::endl;
    } catch (const ParseException &pex) {
        std::cerr << "Parse error at " << pex.getFile() << ":" << pex.getLine()
                << " - " << pex.getError() << std::endl;
    }

	int typeROUKF = 0;
	try {
		typeROUKF = config.lookup("FilterType");
	} catch (const SettingNotFoundException &nfex) {
		cerr << "Configuration file must explicit the FilterType (\"MappedROUKF\" | \"ROUKF\")." << endl;
	}

	try {
		nStates = config.lookup("States");
		nParameters = config.lookup("Parameters");
		nObservations = config.lookup("Observations");
	} catch (const SettingNotFoundException &nfex) {
		cerr << "States, Parameters or Observations number is missing." << endl;
	}

	vector<double> initialGuess;
	vector<double> parameterUncertainty;
	try {
		Setting& sVectorDoubles = config.lookup("InitialGuess");
		for (int i = 0; i < sVectorDoubles.getLength(); ++i) {
			initialGuess.push_back((double) sVectorDoubles[i]);
		}
		Setting& sVectorDoubles2 = config.lookup("ParameterUncertainty");
		for (int i = 0; i < sVectorDoubles2.getLength(); ++i) {
			parameterUncertainty.push_back((double) sVectorDoubles2[i]);
		}
	} catch (const SettingNotFoundException &nfex) {
		cerr << "Error while reading InitialGuess or ParameterUncertainty fields." << endl;
	}

	vector<AbstractParameterMapper *> parameterMapping;
	vector<int> parametersPerMapping;
	try {
		Setting& mapperList = config.lookup("ParameterMapping");
		for (int i = 0; i < mapperList.getLength(); ++i) {
			int mapType = (int) mapperList[i]["type"];
			switch (mapType) {
			case IDENTITY:
				parametersPerMapping.push_back(mapperList[i]["numParam"]);
				parameterMapping.push_back(new IdentityParameterMapper());
				break;
			case EXPONENTIAL:
				parametersPerMapping.push_back((int) mapperList[i]["numParam"]);
				parameterMapping.push_back(new ExponentialParameterMapper());
				break;
			case SIGMOIDAL:
				parametersPerMapping.push_back(mapperList[i]["numParam"]);
				parameterMapping.push_back(new SigmoidParameterMapper((double) (mapperList[i]["min"]), (double) (mapperList[i]["max"])));
				break;
			default:
				parametersPerMapping.push_back(mapperList[i]["numParam"]);
				parameterMapping.push_back(new IdentityParameterMapper());
				break;
			}
		}
	} catch (const SettingNotFoundException &nfex) {
		cerr << "Error while reading ParameterMapping fields." << endl;
	}

	vector<double> observationsUncertainty;
	try {
		Setting& sVectorDoubles = config.lookup("ObservationsValues");
		for (int i = 0; i < sVectorDoubles.getLength(); ++i) {
			observations.push_back((double) sVectorDoubles[i]);
		}
		Setting& sVectorDoubles2 = config.lookup("ObservationsUncertainty");
		for (int i = 0; i < sVectorDoubles2.getLength(); ++i) {
			observationsUncertainty.push_back((double) sVectorDoubles2[i]);
		}
	} catch (const SettingNotFoundException &nfex) {
		cerr << "Error while reading ObservationsValues or ObservationUncertainty fields." << endl;
	}

	SigmaPointsGenerator::SIGMA_DISTRIBUTION sigmaDistribution = SigmaPointsGenerator::SIMPLEX;
	try {
		int sigmaType = config.lookup("SigmaDistribution");
		sigmaDistribution = static_cast<SigmaPointsGenerator::SIGMA_DISTRIBUTION>(sigmaType);
	} catch (const SettingNotFoundException &nfex) {
		cerr << "Error while reading SigmaDistribution field." << endl;
	}

	double tol = 1E-5;
	try {
		tol = config.lookup("ConvergenceTol");
	} catch (const SettingNotFoundException &nfex) {
		cerr << "Error while reading ConvergenceTol field." << endl;
	}

	int maxIt = 1000;
	try {
		maxIt = config.lookup("MaxIterations");
	} catch (const SettingNotFoundException &nfex) {
		cerr << "Error while reading MaxIterations field." << endl;
	}

	switch (typeROUKF) {
	case MODEL_ROUKF:
		roukfModel = new ROUKF(nObservations, nStates, nParameters,
				&(observationsUncertainty[0]), &(parameterUncertainty[0]),
				sigmaDistribution);
		break;
	case MODEL_MAPPED_ROUKF:
		roukfModel = new MappedROUKF(nObservations, nStates, nParameters,
				observationsUncertainty, parameterUncertainty, sigmaDistribution,
				new CompositeParameterMapper(parametersPerMapping, parameterMapping));
		break;
	default:
		roukfModel = new ROUKF(nObservations, nStates, nParameters,
				&(observationsUncertainty[0]), &(parameterUncertainty[0]),
				sigmaDistribution);
		break;
	}
	roukfModel->setParameters(&(initialGuess[0]));
	roukfModel->setTolerance(tol);
	roukfModel->setMaxIterations(maxIt);

	return roukfModel;

}

vector<double> ConfigurationFileReader::getObservations() {
	return observations;
}

int ConfigurationFileReader::getNParameters() const
{
	return nParameters;
}

int ConfigurationFileReader::getNStates() const
{
	return nStates;
}

int ConfigurationFileReader::getNObservations() const
{
	return nObservations;
}
