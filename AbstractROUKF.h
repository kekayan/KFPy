/*
 * AbstractROUKF.h
 *
 *  Created on: Feb 26, 2018
 *      Author: Gonzalo D. Maso Talou
 */

#ifndef ABSTRACTROUKF_H_
#define ABSTRACTROUKF_H_

#include <armadillo>
#include <mpi.h>
#include <vector>

#include "mapping/AbstractParameterMapper.h"
#include "SigmaPointsGenerator.h"

using namespace std;

/**	Type definition for the forward operator	*/
typedef int (*forwardOp)(double *, int, double *, int);
/**	Type definition for the observation operator	*/
typedef void (*observationOp)(double *, int, double *, int);

using namespace arma;
using namespace std;

class AbstractROUKF {

protected:
	/**	States vector.	*/
	arma::mat X;
	/**	Parameters vector.	*/
	arma::mat Theta;
	/**	U part of the covariance matrix	after LU factorization.	*/
	arma::mat U;
	/**	U squared.	*/
	arma::mat U2;
	/**	L part of the covariance matrix	after LU factorization concerning to the state part of the extended state vector.	*/
	arma::mat LX;
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

	/** Convergence tolerance. */
	double tolerance;
	/** Maximum number of iterations. */
	double maxIterations;
	/**	Previous iteration error. */
	double prevError;
	/**	Current iteration error. */
	double currError;
	/** Current iteration. */
	long long int currIt;

public:

	/**
	 *	Abstract constructor.
	 */
	AbstractROUKF();
	/**
	 *	Virtual destructor.
	 */
	virtual ~AbstractROUKF();

	/**
	 * Prints the private attributes of the ROUKF instance.
	 */
	void toString();

	/**
	 * Return if the filter has converged according to the specified @p tolerance.
	 * @param isConvergenceRelative If the convergence is relative or absolute.
	 * @return	If it is converged.
	 */
	bool hasConverged(bool isConvergenceRelative);

	/**
	 * Returns a vector with the standard variation of each parameter at the current iteration.
	 * @return Vector with the standard variation of each parameter at the current iteration.
	 */
	vector<double> getParametersStd();

	/**
	 * Getter of the field @p X.
	 * @param XC Field @p X converted to STL array.
	 */
	void getState(double **XC);
	/**
	 * Setter of the field @p X.
	 * @param XC Array of states.
	 */
	void setState(double *XC);

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

	/**
	 * Getter of the field @p maxIterations.
	 * @return Field @p maxIterations.
	 */
	double getMaxIterations() const;
	/**
	 * Setter of the field @p maxIterations.
	 * @param maxIterations Maximum number of iterations.
	 */
	void setMaxIterations(double maxIterations);
	/**
	 * Getter of the field @p tolerance.
	 * @return Field @p tolerance.
	 */
	double getTolerance() const;
	/**
	 * Setter of the field @p tolerance.
	 * @param tolerance Maximum tolerance allowed.
	 */
	void setTolerance(double tolerance);

};

#endif /* ABSTRACTROUKF_H_ */
