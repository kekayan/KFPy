/*
 * ROUKF.cpp
 *
 *  Created on: Oct 26, 2015
 *      Author: Gonzalo D. Maso Talou
 */

#include "StaticROUKF.h"

using namespace arma;

StaticROUKF::StaticROUKF(int nObservations, int nStates, int nParameters, double* observationsUncertainty,
		double* parametersUncertainty, SigmaPointsGenerator::SIGMA_DISTRIBUTION sigmaDistribution) {

	this->nObservations = nObservations;
	this->nStates = nStates;
	this->nParameters = nParameters;

	Theta = zeros(nParameters, 1);

	LTheta = eye(nParameters, nParameters);
	U = eye(nParameters, nParameters);
	Wi = eye(nObservations, nObservations);

	mat diagParameter(parametersUncertainty, 1, nParameters);
	U.diag() = 1. / diagParameter;
	mat diagObservations(observationsUncertainty, 1, nObservations);
	Wi.diag() = 1. / diagObservations;

	SigmaPointsGenerator::generateSigmaPoints(nParameters,sigmaDistribution,&sigma);
	alpha = 1. / sigma.n_cols;

	Dsigma = alpha * eye(sigma.n_cols, sigma.n_cols);
	Dsigma = Dsigma * sigma.t();
	Pa = sigma * Dsigma;

}

double StaticROUKF::executeStep(double *zkhatc, forwardOp A, observationOp H) {

	//	Matrixes
	mat Thetak(nParameters, sigma.n_cols), Xk(nStates, sigma.n_cols), Zk(
			nObservations, sigma.n_cols);
	mat HL;
	mat invU = inv(U);
	mat C = chol(invU);

	//	Column vectors
	mat thetak, zk;
	mat zkhat(zkhatc, nObservations, 1);

	double *thetakdata, *xkdata, *zkdata;
	thetakdata = new double[nParameters];
	xkdata = new double[nStates];
	zkdata = new double[nObservations];

	for (unsigned int i = 0; i < sigma.n_cols; i++) {
		//	Sampling
		mat s = sigma.col(i);
		thetak = Theta + LTheta * C.t() * s;

		//	Propagate sigma point
		//	Copies xk to a primitive type in order to achieve armadillo independence in extern programs
		memcpy(thetakdata, thetak.memptr(), nParameters * sizeof(double));

		(*A)(xkdata, nStates, thetakdata, nParameters);
		thetak = mat(thetakdata, nParameters, 1);

		//	Perform observation
		(*H)(xkdata, nStates, zkdata, nObservations);
		zk = mat(zkdata, nObservations, 1);

		Thetak.col(i) = thetak;
		Zk.col(i) = zk;
	}
	//	New state and its associated observation
	thetak = mean(Thetak, 1);
	mat zkMean = mean(Zk, 1);	// Only for simplex case (other must use the weighted mean)
	error = zkhat - zkMean;

	//	Update covariance matrixes
	LTheta = Thetak * Dsigma;
	HL = Zk * Dsigma;
	U = Pa + HL.t() * (Wi * HL);

	invU = inv(U);

	//	Compute new estimate
	Theta = thetak + LTheta * invU * HL.t() * (Wi * error);

	delete[] thetakdata;
	delete[] xkdata;
	delete[] zkdata;

	return norm(error, 2);
}

double StaticROUKF::executeStepParallel(double* zkhatc, forwardOp A, observationOp H,
		int sigmaPoint, MPI_Comm world_comm, MPI_Comm sigmaMasters_comm) {

	//	Matrixes
	mat Thetak(nParameters, sigma.n_cols), Xk(nStates, sigma.n_cols),
			Zk(nObservations, sigma.n_cols);
	mat HL;
	mat invU = inv(U);
	mat C = chol(invU);

	//	Column vectors
	mat thetak, zk;
	mat zkhat(zkhatc, nObservations, 1);

	double *thetakdata, *xkdata, *zkdata;
	thetakdata = new double[nParameters];
	xkdata = new double[nStates];
	zkdata = new double[nObservations];

	//	Sampling
	mat s = sigma.col(sigmaPoint);
	thetak = Theta + LTheta * C.t() * s;

	//	Propagate sigma point
	//	Copies xk to a primitive type in order to achieve armadillo independence in extern programs
	memcpy(thetakdata, thetak.memptr(), nParameters * sizeof(double));

	(*A)(xkdata, nStates, thetakdata, nParameters);
	thetak = mat(thetakdata, nParameters, 1);
	(*H)(xkdata, nStates, zkdata, nObservations);
	zk = mat(zkdata, nObservations, 1);

	cout << "Sync with masters" << endl;
	if (sigmaMasters_comm != MPI_COMM_NULL) {
		//	Masters of each solver (rank < (nParameters + 1)) interchange data from executions
		cout << "Masters gathering info ";
		MPI_Gather(thetak.memptr(), nParameters, MPI_DOUBLE, Thetak.memptr(),
				nParameters, MPI_DOUBLE, 0, sigmaMasters_comm);
		cout << "thetak ";
		MPI_Gather(zk.memptr(), nObservations, MPI_DOUBLE, Zk.memptr(), nObservations, MPI_DOUBLE,
				0, sigmaMasters_comm);
		cout << "zk." << endl;;
	}

	//	Main master broadcasts the sigma points processing to all the workers in all solvers.
	cout << "Masters broadcasting" << endl;
	MPI_Bcast(Thetak.memptr(), (sigma.n_cols) * nParameters, MPI_DOUBLE, 0,
			world_comm);
	MPI_Bcast(Zk.memptr(), (sigma.n_cols) * nObservations, MPI_DOUBLE, 0,
			world_comm);

	//	New state and its associated observation
	thetak = mean(Thetak, 1);
	mat zkMean = mean(Zk, 1);	// Only for simplex case (other must use the weighted mean)
	error = zkhat - zkMean;

	//	Update covariance matrixes
	LTheta = Thetak * Dsigma;
	HL = Zk * Dsigma;
	U = Pa + HL.t() * (Wi * HL);

	invU = inv(U);

	//	Compute new estimate
	Theta = thetak + LTheta * invU * HL.t() * (Wi * error);

	delete[] thetakdata;
	delete[] xkdata;
	delete[] zkdata;

	return norm(error, 2);
}

void StaticROUKF::getParameters(double** thetac) {
	*thetac = Theta.memptr();
}

void StaticROUKF::setParameters(double *thetac) {
	Theta = mat(thetac, nParameters, 1);
}

void StaticROUKF::getError(double** err) {
	*err = error.memptr();
}

double StaticROUKF::getObsError(int numObservation) {
	return error.at(numObservation);
}

void StaticROUKF::reset(int nObservations, int nStates, int nParameters, double* observationsUncertainty,
		double* parametersUncertainty, SigmaPointsGenerator::SIGMA_DISTRIBUTION sigmaDistribution) {
	this->nObservations = nObservations;
	this->nStates = nStates;
	this->nParameters = nParameters;

	Theta = zeros(nParameters, 1);

	LTheta = eye(nParameters, nParameters);
	U = eye(nParameters, nParameters);
	Wi = eye(nObservations, nObservations);

	mat diagParameter(parametersUncertainty, 1, nParameters);
	U.diag() = 1. / diagParameter;
	mat diagState(observationsUncertainty, 1, nObservations);
	Wi.diag() = 1. / diagState;

	SigmaPointsGenerator::generateSigmaPoints(nParameters,sigmaDistribution,&sigma);
	alpha = 1. / sigma.n_cols;

	Dsigma = alpha * eye(sigma.n_cols, sigma.n_cols);
	Dsigma = Dsigma * sigma.t();
	Pa = sigma * Dsigma;

}

void StaticROUKF::toString() {

	Theta.print("Theta:");
	U.print("U:");
	LTheta.print("LTheta:");
	sigma.print("sigma:");
	Dsigma.print("Dsigma:");
	Pa.print("Pa:");
	Wi.print("Wi:");

}

vector<double> StaticROUKF::getParametersStd(){
	mat std = sqrt(1. /U.diag());
	return arma::conv_to< vector<double> >::from(std);
}

int StaticROUKF::getObservations() const
{
	return nObservations;
}

int StaticROUKF::getStates() const
{
	return nStates;
}
