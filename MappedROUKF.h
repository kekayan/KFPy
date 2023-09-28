/*
 * MappedROUKF.h
 *
 *	Reduce-order Unscent Kalman Filter for state independent operators
 *	Based on Fernando Mut implementation.
 *
 *	Dependences:
 *		Armadillo library 6.2 - http://sourceforge.net/projects/arma/files/armadillo-6.200.2.tar.gz
 *
 *  Created on: Oct 26, 2015
 *      Author: Gonzalo D. Maso Talou
 */

#ifndef MAPPED_ROUKF_H_
#define MAPPED_ROUKF_H_

#include <armadillo>
#include <mpi.h>
#include <vector>

#include "AbstractROUKF.h"
#include "mapping/CompositeParameterMapper.h"
#include "SigmaPointsGenerator.h"

/**
 * An example for the usage of the library:
 *
 *	@code
 * 	int (*ptA)(double*, int) = NULL;
 *	void (*ptH)(double*, int, double*, int) = NULL;
 *	ptA = &forwardOpElasticity;
 *	ptH = &observerOpElasticity;
 *
 *	 //	Initialization parameters
 *	for (int i = 0; i < nParameters; i++) {
 *		initialGuess[i] = 2.5E5;			// -> X0
 *		parametersUncertainties[i] =1E8;	// -> U0^{-1}
 *	}
 *	ROUKF *kalmanInstance = new ROUKF(nStates, nParameters, statesUncertainties,
 *	parametersUncertainties);
 *
 *	//	Set initial condition
 *	kalmanInstance->setState(initialGuess, nParameters);
 *
 *	for (int it = 0; it < 3000; it++) {
 *		error = kalmanInstance->executeStep(observation, nStates, ptA, ptH);
 *	}
 *
 *	//	Get the Kalman estimation
 *	kalmanInstance->getState(&sol);			// -> XSol
 *	@endcode
 */

/**
 * Class that implements the reduced order unscented Kalman filter.
 */
class MappedROUKF : public AbstractROUKF {

	/** Mapping function between the problem parameters and the kalman parameters. */
	CompositeParameterMapper *mapper;
public:

	/**	Reparametrization type. */
	enum MAPPING_TYPE {
		DEFAULT, POSITIVE, RANGED
	};

	/**
	 *	Creates the covariance matrixes and sigma points associated with the extended
	 * vector X and their uncertainty.
	 * @param nObservations Quantity of observations.
	 * @param nStates Quantity of states.
	 * @param nParameters Quantity of parameters.
	 * @param observationsUncertainty Vector with the uncertainty of each state in X.
	 * @param parametersUncertainty Vector with the uncertainty of each parameter in Theta.
	 * @param sigmaDistribution	Type of sigmas applied to assess the unscented transform.
	 */
	MappedROUKF(int nObservations, int nStates, int nParameters,
			vector<double> observationsUncertainty, vector<double> parametersUncertainty,
			SigmaPointsGenerator::SIGMA_DISTRIBUTION sigmaDistribution);

	/**
	 *	Creates the covariance matrixes and sigma points associated with the extended
	 * vector X and their uncertainty. The chosen mapping is the same for all parameters.
	 * @param nObservations Quantity of observations.
	 * @param nStates Quantity of states.
	 * @param nParameters Quantity of parameters.
	 * @param observationsUncertainty Vector with the uncertainty of each state in X.
	 * @param parametersUncertainty Vector with the uncertainty of each parameter in Theta.
	 * @param sigmaDistribution	Type of sigmas applied to assess the unscented transform.
	 * @param mappingType Type of mapping between the problem and the kalman parameters.
	 * @param mappingParameters Parameters for the chosen mapping type.
	 */
	MappedROUKF(int nObservations, int nStates, int nParameters,
			vector<double> observationsUncertainty, vector<double> parametersUncertainty,
			SigmaPointsGenerator::SIGMA_DISTRIBUTION sigmaDistribution, MappedROUKF::MAPPING_TYPE mappingType, vector<double > mappingParameters);

	/**
	 *	Creates the covariance matrixes and sigma points associated with the extended
	 * vector X and their uncertainty. The mapping between the problem and the kalman parameters
	 * is given by the user customized @p mapper mapping function.
	 * @param nObservations Quantity of observations.
	 * @param nStates Quantity of states.
	 * @param nParameters Quantity of parameters.
	 * @param observationsUncertainty Vector with the uncertainty of each state in X.
	 * @param parametersUncertainty Vector with the uncertainty of each parameter in Theta.
	 * @param sigmaDistribution	Type of sigmas applied to assess the unscented transform.
	 * @param mapper User customized mapping function between the problem and the kalman parameters.
	 */
	MappedROUKF(int nObservations, int nStates, int nParameters,
			vector<double> observationsUncertainty, vector<double> parametersUncertainty,
			SigmaPointsGenerator::SIGMA_DISTRIBUTION sigmaDistribution, CompositeParameterMapper *mapper);

	/**
	 * Void destructor.
	 */
	~MappedROUKF();

	/**
	 * Performs one step of the Kalman filtering process in serial execution of the sigma points.
	 * @param Zkhatc	Current observations estimations.
	 * @param A	Forward operator.
	 * @param H	Observation operator;
	 * @return	Current L2 norm of the errors across all observations.
	 */
	double executeStep(vector<double> Zkhatc, forwardOp A, observationOp H);
	/**
	 * Performs one step of the Kalman filtering process with parallel execution of the sigma points.
	 * @param Zkhatc	Current observations estimations.
	 * @param A	Forward operator.
	 * @param H	Observation operator;
	 * @param seed Sigma point ID for the current MPI process.
	 * @param local_comm Communicator of all MPI processes that solve the sigma point @p seed.
	 * @param masters_comm Communicator of the master MPI processes of each sigma point @p seed.
	 * @return	Current L2 norm of the errors across all observations.
	 */
	double executeStepParallel(vector<double> Zkhatc, forwardOp A, observationOp H,
			int seed, MPI_Comm local_comm, MPI_Comm masters_comm);

	/**
	 * Returns to the initial state of the kalman filter. Not fully tested
	 * @param nObservations Quantity of observations.
	 * @param nStates Quantity of states.
	 * @param nParameters Quantity of parameters.
	 * @param observationsUncertainty Vector with the uncertainty of each state in X.
	 * @param parametersUncertainty Vector with the uncertainty of each parameter in Theta.
	 * @param sigmaDistribution	Type of sigmas applied to assess the unscented transform.
	 */
	void reset(int nObservations, int nStates, int nParameters,
			vector<double> observationsUncertainty, vector<double> parametersUncertainty,
			SigmaPointsGenerator::SIGMA_DISTRIBUTION sigmaDistribution);
	/**
	 *	Replaces the current @p mapper for the new CompositeParameterMapper instance.
	 * @param mapper New mapper instance.
	 */
	void replaceMapper(CompositeParameterMapper *mapper);

	/**
	 * Return the current parameter estimatives in the array @p ThetaC.
	 * @param ThetaC Current parameter estimatives.
	 */
	virtual void getParameters(double **ThetaC) override;
	/**
	 * Sets the initial parameter estimatives.
	 * @param ThetaC Initial parameter estimatives.
	 */
	virtual void setParameters(double *ThetaC) override;

};

#endif /* MAPPED_ROUKF_H_ */
