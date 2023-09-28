/*
 * ROUKF.h
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

#ifndef STATICROUKF_H_
#define STATICROUKF_H_

#include <armadillo>
#include <mpi.h>
#include <vector>

#include "SigmaPointsGenerator.h"

using namespace std;

typedef int (*forwardOp)(double *, int, double *, int);
typedef void (*observationOp)(double *, int, double *, int);

/**
 * Class that implements the reduced order unscented Kalman filter without
 * statistical propagation of the internal state.
 */
class StaticROUKF {

	/**	Parameters vector.	*/
	arma::mat Theta;
	/**	U part of the covariance matrix	after LU factorization.	*/
	arma::mat U;
	/**	U squared.	*/
	arma::mat U2;
	/**	L part of the covariance matrix	after LU factorization concerning to the parameter part of the extended state vector.	*/
	arma::mat LTheta;
	/** Observations confidence matrix.	*/
	arma::sp_mat Wi;

	/**	Matrix with sigma points as columns. */
	arma::mat sigma;
	/** Matrix with sigma points weighted as rows. */
	arma::mat Dsigma;
	/** Matrix @p sigma times @p Dsigma . */
	arma::mat Pa;

	/**	Vector with the observations errors after the last iteration. */
	arma::mat error;

	/**	Quantity of observations. */
	int nObservations;
	/**	Quantity of parameters. */
	int nParameters;
	/**	Quantity of states. */
	int nStates;
	/**	Weight for each sigma point. */
	double alpha;

public:

	/**	Reparametrization type. Not implemented yet.	*/
	enum PARAM_TYPE {DEFAULT,POSITIVE,RANGED_LOG_DIST,RANGED_NORMAL_DIST};

	/**
	 *	Creates the covariance matrixes and sigma points associated with the extended
	 * vector X and their uncertainty.
	 * @param nObservations Quantity of observations.
	 * @param nStates Quantity of states.
	 * @param nParameters Quantity of parameters.
	 * @param statesUncertainty Vector with the uncertainty of each state in X.
	 * @param parametersUncertainty Vector with the uncertainty of each parameter in Theta.
	 * @param sigmaDistribution	Type of sigmas applied to assess the unscented transform. Only SIMPLEX is available by now.
	 */
	StaticROUKF(int nObservations, int nStates, int nParameters,
			double *statesUncertainty, double *parametersUncertainty,
			SigmaPointsGenerator::SIGMA_DISTRIBUTION sigmaDistribution);

	/**
	 * Performs one step of the Kalman filtering process in serial execution of the sigma points.
	 * @param Zkhatc	Current observations estimations.
	 * @param A	Forward operator.
	 * @param H	Observation operator;
	 * @return	Current L2 norm of the errors across all observations.
	 */
	double executeStep(double *Zkhatc, forwardOp A, observationOp H);
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
	double executeStepParallel(double *Zkhatc, forwardOp A, observationOp H,
			int seed, MPI_Comm local_comm, MPI_Comm masters_comm);

	/**
	 * Returns to the initial state of the kalman filter. Not fully tested
	 * @param nObservations Quantity of observations.
	 * @param nStates Quantity of states.
	 * @param nParameters Quantity of parameters.
	 * @param statesUncertainty Vector with the uncertainty of each state in X.
	 * @param parametersUncertainty Vector with the uncertainty of each parameter in Theta.
	 * @param sigmaDistribution	Type of sigmas applied to assess the unscented transform. Only SIMPLEX is available by now.
	 */
	void reset(int nObservations, int nStates, int nParameters,
			double *statesUncertainty, double *parametersUncertainty,
			SigmaPointsGenerator::SIGMA_DISTRIBUTION sigmaDistribution);

	/**
	 * Prints the private attributes of the ROUKF instance.
	 */
	void toString();

	/**
	 * Returns a vector with the standard variation of each parameter at the current iteration.
	 * @return Vector with the standard variation of each parameter at the current iteration.
	 */
	vector<double> getParametersStd();
	/**
	 * Getter of the field @p Theta.
	 * @param ThetaC Field @p Theta converted to STL array.
	 */
	virtual void getParameters(double **ThetaC);
	/**
	 * Setter of the field @p Theta.
	 * @param ThetaC Array of parameters.
	 */
	virtual void setParameters(double *ThetaC);

	/**
	 * Returns the error associated to each observation at the current iteration in a STL array.
	 * @param err Error associated to each observation at the current iteration.
	 */
	void getError(double **err);
	/**
	 * Returns the error associated to the @p numObservation observation.
	 * @param numObservation Index of the observation of interest.
	 */
	double getObsError(int numObservation);

	/**
	 * Return the number of observations used in this instance of the kalman filter.
	 * @return Number of observations used in this instance of the kalman filter.
	 */
	int getObservations() const;
	/**
	 * Return the number of states used in this instance of the kalman filter.
	 * @return Number of states used in this instance of the kalman filter.
	 */
	int getStates() const;
};

#endif /* StatelessROUKF_H_ */
