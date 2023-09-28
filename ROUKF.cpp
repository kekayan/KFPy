/*
 * ROUKF.cpp
 *
 *  Created on: Oct 26, 2015
 *      Author: Gonzalo D. Maso Talou
 */

#include "iostream"
#include "ROUKF.h"
#include <cmath>

ROUKF::ROUKF(int nObservations, int nStates, int nParameters, double* observationsUncertainty,
		double* parametersUncertainty, SigmaPointsGenerator::SIGMA_DISTRIBUTION sigmaDistribution) : AbstractROUKF(){

	this->nObservations = nObservations;
	this->nStates = nStates;
	this->nParameters = nParameters;

	X = zeros(nStates, 1);
	Theta = zeros(nParameters, 1);

	LTheta = eye(nParameters, nParameters);
	LX = zeros(nStates, nParameters);
	U = eye(nParameters, nParameters);
	Wi = speye(nObservations, nObservations);

	mat diagParameter(parametersUncertainty, 1, nParameters);
	U.diag() = 1. / diagParameter;
	mat diagObservations(observationsUncertainty, 1, nObservations);
	Wi.diag() = 1. / diagObservations;

	SigmaPointsGenerator::generateSigmaPoints(nParameters, sigmaDistribution, &sigma);
	alpha = 1. / sigma.n_cols;

	Dsigma = alpha * eye(sigma.n_cols, sigma.n_cols);
	Dsigma = Dsigma * sigma.t();
	Pa = sigma * Dsigma;

}

ROUKF::~ROUKF(){
}

double ROUKF::executeStep(double *zkhatc, forwardOp A, observationOp H) {

	//	Matrixes
	mat Thetak(nParameters, sigma.n_cols), Xk(nStates, sigma.n_cols), Zk(
			nObservations, sigma.n_cols);
	mat HL;
	mat invU = inv(U);
	mat C = chol(invU);

	//	Column vectors
	mat thetak, xk, zk;
	mat zkhat(zkhatc, nObservations, 1);

	double *thetakdata, *xkdata, *zkdata;
	thetakdata = new double[nParameters];
	xkdata = new double[nStates];
	zkdata = new double[nObservations];

	for (unsigned int i = 0; i < sigma.n_cols; i++) {
		//	Sampling
		mat s = sigma.col(i);
		xk = X + LX * C.t() * s;
		thetak = Theta + LTheta * C.t() * s;

		//	Transform theta_k to physical values
		//	TODO Transform to physical values

		//	Propagate sigma point
		//	Copies xk to a primitive type in order to achieve armadillo independence in extern programs
		memcpy(thetakdata, thetak.memptr(), nParameters * sizeof(double));
		memcpy(xkdata, xk.memptr(), nStates * sizeof(double));

		(*A)(xkdata, nStates, thetakdata, nParameters);
		thetak = mat(thetakdata, nParameters, 1);
		xk = mat(xkdata, nStates, 1);

		//	Perform observation
		(*H)(xk.memptr(), nStates, zkdata, nObservations);
		zk = mat(zkdata, nObservations, 1);
		//	Transform theta_k to assimilation values
		//	TODO Transform to data assimilation values

		Xk.col(i) = xk;
		Thetak.col(i) = thetak;
		Zk.col(i) = zk;
	}
	//	New state and its associated observation
	xk = mean(Xk, 1);
	thetak = mean(Thetak, 1);
	mat zkMean = mean(Zk, 1);	// Only constant alpha
	error = zkhat - zkMean;

	//	Update covariance matrixes
	LX = Xk * Dsigma;
	LTheta = Thetak * Dsigma;
	HL = Zk * Dsigma;
	U = Pa + HL.t() * (Wi * HL);

	invU = inv(U);

	//	Compute new estimate
	X = xk + LX * invU * HL.t() * (Wi * error);
	Theta = thetak + LTheta * invU * HL.t() * (Wi * error);

	delete[] thetakdata;
	delete[] xkdata;
	delete[] zkdata;

	prevError = currError;
	currError = norm(error, 2);
	++currIt;

	return currError;
}

double ROUKF::executeStepParallel(double* zkhatc, forwardOp A, observationOp H,
		int sigmaPoint, MPI_Comm world_comm, MPI_Comm sigmaMasters_comm) {

	//	Matrixes
	mat Thetak(nParameters, sigma.n_cols), Xk(nStates, sigma.n_cols),
			Zk(nObservations, sigma.n_cols);
	mat HL;
	mat invU = inv(U);
	mat C = chol(invU);

	//	Column vectors
	mat thetak(nParameters, 1);
	mat xk(nStates, 1);
	mat zk(nObservations, 1);
	mat zkhat(zkhatc, nObservations, 1);

	double *thetakdata, *xkdata, *zkdata;
	thetakdata = new double[nParameters];
	xkdata = new double[nStates];
	zkdata = new double[nObservations];

	//	Sampling
	mat s = sigma.col(sigmaPoint);
	xk = X + LX * C.t() * s;
	thetak = Theta + LTheta * C.t() * s;

	//	Propagate sigma point
	//	Copies xk to a primitive type in order to achieve armadillo independence in extern programs
	memcpy(thetakdata, thetak.memptr(), nParameters * sizeof(double));
	memcpy(xkdata, xk.memptr(), nStates * sizeof(double));

	(*A)(xkdata, nStates, thetakdata, nParameters);
	thetak = mat(thetakdata, nParameters, 1);
	xk = mat(xkdata, nStates, 1);

	//	Perform observation
	(*H)(xk.memptr(), nStates, zkdata, nObservations);
	zk = mat(zkdata, nObservations, 1);

	if (sigmaMasters_comm != MPI_COMM_NULL) {
		//	Masters of each solver (rank < (nParameters + 1)) interchange data from executions
		cout << "Masters gathering info ";
		MPI_Gather(xk.memptr(), nStates, MPI_DOUBLE, Xk.memptr(), nStates, MPI_DOUBLE,
				0, sigmaMasters_comm);
		cout << "xk ";
		MPI_Gather(thetak.memptr(), nParameters, MPI_DOUBLE, Thetak.memptr(),
				nParameters, MPI_DOUBLE, 0, sigmaMasters_comm);
		cout << "thetak ";
		MPI_Gather(zk.memptr(), nObservations, MPI_DOUBLE, Zk.memptr(), nObservations, MPI_DOUBLE,
				0, sigmaMasters_comm);
		cout << "zk." << endl;
		;
	}

	//	Main master broadcasts the sigma points processing to all the workers in all solvers.
	MPI_Bcast(Xk.memptr(), (sigma.n_cols) * nStates, MPI_DOUBLE, 0,
			world_comm);
	MPI_Bcast(Thetak.memptr(), (sigma.n_cols) * nParameters, MPI_DOUBLE, 0,
			world_comm);
	MPI_Bcast(Zk.memptr(), (sigma.n_cols) * nObservations, MPI_DOUBLE, 0,
			world_comm);

	//	New state and its associated observation
	xk = mean(Xk, 1);
	thetak = mean(Thetak, 1);
	mat zkMean = mean(Zk, 1);	// Only for simplex case (other must use the weighted mean)
	error = zkhat - zkMean;

	//	Update covariance matrixes
	LX = Xk * Dsigma;
	LTheta = Thetak * Dsigma;
	HL = Zk * Dsigma;
	U = Pa + HL.t() * (Wi * HL);

	invU = inv(U);

	//	Compute new estimate
	X = xk + LX * invU * HL.t() * (Wi * error);
	Theta = thetak + LTheta * invU * HL.t() * (Wi * error);

	delete[] thetakdata;
	delete[] xkdata;
	delete[] zkdata;

	prevError = currError;
	currError = norm(error, 2);
	++currIt;

	return currError;
}

void ROUKF::reset(int nObservations, int nStates, int nParameters, double* observationsUncertainty,
		double* parametersUncertainty, SigmaPointsGenerator::SIGMA_DISTRIBUTION sigmaDistribution) {
	this->nObservations = nObservations;
	this->nStates = nStates;
	this->nParameters = nParameters;

	X = zeros(nStates, 1);
	Theta = zeros(nParameters, 1);

	LTheta = eye(nParameters, nParameters);
	LX = zeros(nStates, nParameters);
	U = eye(nParameters, nParameters);
	Wi = eye(nObservations, nObservations);

	mat diagParameter(parametersUncertainty, 1, nParameters);
	U.diag() = 1. / diagParameter;
	mat diagState(observationsUncertainty, 1, nObservations);
	Wi.diag() = 1. / diagState;

	SigmaPointsGenerator::generateSigmaPoints(nParameters, sigmaDistribution, &sigma);
	alpha = 1. / sigma.n_cols;

	Dsigma = alpha * eye(sigma.n_cols, sigma.n_cols);
	Dsigma = Dsigma * sigma.t();
	Pa = sigma * Dsigma;

}
