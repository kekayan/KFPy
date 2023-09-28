/*
 * ROUKF.cpp
 *
 *  Created on: Oct 26, 2015
 *      Author: Gonzalo D. Maso Talou
 */

#include "iostream"
#include "MappedROUKF.h"
#include "mapping/IdentityParameterMapper.h"
#include "mapping/ExponentialParameterMapper.h"
#include "mapping/SigmoidParameterMapper.h"
#include <cmath>

using namespace arma;
using namespace std;

MappedROUKF::MappedROUKF(int nObservations, int nStates, int nParameters, vector<double> observationsUncertainty, vector<double> parametersUncertainty,
		SigmaPointsGenerator::SIGMA_DISTRIBUTION sigmaDistribution) :
		AbstractROUKF() {

	this->nObservations = nObservations;
	this->nStates = nStates;
	this->nParameters = nParameters;

	X = zeros(nStates, 1);
	Theta = zeros(nParameters, 1);

	LTheta = eye(nParameters, nParameters);
	LX = zeros(nStates, nParameters);
	U = eye(nParameters, nParameters);
	Wi = speye(nObservations, nObservations);

	mat diagParameter(&(parametersUncertainty[0]), 1, nParameters);
	U.diag() = 1. / diagParameter;
	mat diagObservations(&(observationsUncertainty[0]), 1, nObservations);
	Wi.diag() = 1. / diagObservations;

	SigmaPointsGenerator::generateSigmaPoints(nParameters, sigmaDistribution, &sigma);
	alpha = 1. / sigma.n_cols;

	Dsigma = alpha * eye(sigma.n_cols, sigma.n_cols);
	Dsigma = Dsigma * sigma.t();
	Pa = sigma * Dsigma;

	vector<AbstractParameterMapper *> mappers;
	mappers.push_back(new IdentityParameterMapper());
	vector<int> paramsPerMapper = { (int) sigma.n_cols };
	mapper = new CompositeParameterMapper(paramsPerMapper, mappers);
}

MappedROUKF::MappedROUKF(int nObservations, int nStates, int nParameters, vector<double> observationsUncertainty, vector<double> parametersUncertainty,
		SigmaPointsGenerator::SIGMA_DISTRIBUTION sigmaDistribution, MAPPING_TYPE mappingType, vector<double> mappingParameters) :
		AbstractROUKF() {
	this->nObservations = nObservations;
	this->nStates = nStates;
	this->nParameters = nParameters;

	X = zeros(nStates, 1);
	Theta = zeros(nParameters, 1);

	LTheta = eye(nParameters, nParameters);
	LX = zeros(nStates, nParameters);
	U = eye(nParameters, nParameters);
	Wi = speye(nObservations, nObservations);

	mat diagParameter(&(parametersUncertainty[0]), 1, nParameters);
	U.diag() = 1. / diagParameter;
	mat diagObservations(&(observationsUncertainty[0]), 1, nObservations);
	Wi.diag() = 1. / diagObservations;

	SigmaPointsGenerator::generateSigmaPoints(nParameters, sigmaDistribution, &sigma);
	alpha = 1. / sigma.n_cols;

	Dsigma = alpha * eye(sigma.n_cols, sigma.n_cols);
	Dsigma = Dsigma * sigma.t();
	Pa = sigma * Dsigma;

	vector<AbstractParameterMapper *> mappers;
	vector<int> paramsPerMapper = { (int) sigma.n_cols };
	switch (mappingType) {
	case DEFAULT:
		mappers.push_back(new IdentityParameterMapper());
		break;
	case POSITIVE:
		mappers.push_back(new ExponentialParameterMapper());
		break;
	case RANGED:
		mappers.push_back(new SigmoidParameterMapper(mappingParameters[0], mappingParameters[1]));
		break;
	default:
		cout << "Unrecognize type of parameter mapping, default is employed" << endl;
		mappers.push_back(new IdentityParameterMapper());
		break;
	}
	mapper = new CompositeParameterMapper(paramsPerMapper,mappers);
}

MappedROUKF::MappedROUKF(int nObservations, int nStates, int nParameters, vector<double> observationsUncertainty, vector<double> parametersUncertainty,
		SigmaPointsGenerator::SIGMA_DISTRIBUTION sigmaDistribution, CompositeParameterMapper *mapper) :
		AbstractROUKF() {
	this->nObservations = nObservations;
	this->nStates = nStates;
	this->nParameters = nParameters;

	X = zeros(nStates, 1);
	Theta = zeros(nParameters, 1);

	cout << X << endl;
	cout << Theta << endl;

	LTheta = eye(nParameters, nParameters);
	LX = zeros(nStates, nParameters);
	U = eye(nParameters, nParameters);
	Wi = speye(nObservations, nObservations);

	mat diagParameter(&(parametersUncertainty[0]), 1, nParameters);
	U.diag() = 1. / diagParameter;
	mat diagObservations(&(observationsUncertainty[0]), 1, nObservations);
	Wi.diag() = 1. / diagObservations;

	cout << U << endl;
	cout << Wi << endl;

	SigmaPointsGenerator::generateSigmaPoints(nParameters, sigmaDistribution, &sigma);
	alpha = 1. / sigma.n_cols;

	Dsigma = alpha * eye(sigma.n_cols, sigma.n_cols);
	Dsigma = Dsigma * sigma.t();
	Pa = sigma * Dsigma;

	this->mapper = mapper;
}

MappedROUKF::~MappedROUKF(){
}

double MappedROUKF::executeStep(vector<double> zkhatc, forwardOp A, observationOp H) {

	//	Matrixes
	mat Thetak(nParameters, sigma.n_cols), Xk(nStates, sigma.n_cols), Zk(nObservations, sigma.n_cols);
	mat HL;
	mat invU = inv(U);
	mat C = chol(invU);

	//	Column vectors
	mat thetak, xk, zk;
	mat zkhat(&(zkhatc[0]), nObservations, 1);

	double *zkdata;
	zkdata = new double[nObservations];

	for (unsigned int i = 0; i < sigma.n_cols; i++) {
		//	Sampling
		mat s = sigma.col(i);
		xk = X + LX * C.t() * s;
		thetak = Theta + LTheta * C.t() * s;

		//	Transform theta_k -kalman parameters- to problem values -problem parameters-

		//	Propagate sigma point
		//	Copies xk to a primitive type in order to achieve armadillo independence in extern programs
		vector<double> thetakdata(thetak.memptr(), thetak.memptr() + nParameters);
		vector<double> xkdata(xk.memptr(), xk.memptr() + nStates);

		thetakdata = mapper->unmap(thetakdata);
		(*A)(&(xkdata[0]), nStates, &(thetakdata[0]), nParameters);
		thetakdata = mapper->map(thetakdata);

		thetak = mat(&(thetakdata[0]), nParameters, 1);
		xk = mat(&(xkdata[0]), nStates, 1);

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

	delete[] zkdata;
	prevError = currError;
	currError = norm(error, 2);
	++currIt;

	return currError;
}

double MappedROUKF::executeStepParallel(vector<double> zkhatc, forwardOp A, observationOp H, int sigmaPoint, MPI_Comm world_comm, MPI_Comm sigmaMasters_comm) {

	//	Matrixes
	mat Thetak(nParameters, sigma.n_cols), Xk(nStates, sigma.n_cols), Zk(nObservations, sigma.n_cols);
	mat HL;
	mat invU = inv(U);
	mat C = chol(invU);

	//	Column vectors
	mat thetak(nParameters, 1);
	mat xk(nStates, 1);
	mat zk(nObservations, 1);
	mat zkhat(&(zkhatc[0]), nObservations, 1);

	double *zkdata;
	zkdata = new double[nObservations];

	//	Sampling
	mat s = sigma.col(sigmaPoint);
	xk = X + LX * C.t() * s;
	thetak = Theta + LTheta * C.t() * s;

	//	Propagate sigma point
	//	Copies xk to a primitive type in order to achieve armadillo independence in extern programs
	vector<double> thetakdata(thetak.memptr(), thetak.memptr() + nParameters);
	vector<double> xkdata(xk.memptr(), xk.memptr() + nStates);

	thetakdata = mapper->unmap(thetakdata);
	(*A)(&(xkdata[0]), nStates, &(thetakdata[0]), nParameters);
	thetakdata = mapper->map(thetakdata);

	thetak = mat(&(thetakdata[0]), nParameters, 1);
	xk = mat(&(xkdata[0]), nStates, 1);

	//	Perform observation
	(*H)(xk.memptr(), nStates, zkdata, nObservations);
	zk = mat(zkdata, nObservations, 1);

	if (sigmaMasters_comm != MPI_COMM_NULL) {
		//	Masters of each solver (rank < (nParameters + 1)) interchange data from executions
//		cout << "Masters gathering info ";
		MPI_Gather(xk.memptr(), nStates, MPI_DOUBLE, Xk.memptr(), nStates,
		MPI_DOUBLE, 0, sigmaMasters_comm);
//		cout << "xk ";
		MPI_Gather(thetak.memptr(), nParameters, MPI_DOUBLE, Thetak.memptr(), nParameters, MPI_DOUBLE, 0, sigmaMasters_comm);
//		cout << "thetak ";
		MPI_Gather(zk.memptr(), nObservations, MPI_DOUBLE, Zk.memptr(), nObservations, MPI_DOUBLE, 0, sigmaMasters_comm);
//		cout << "zk." << endl;
		;
	}

	//	Main master broadcasts the sigma points processing to all the workers in all solvers.
	MPI_Bcast(Xk.memptr(), (sigma.n_cols) * nStates, MPI_DOUBLE, 0, world_comm);
	MPI_Bcast(Thetak.memptr(), (sigma.n_cols) * nParameters, MPI_DOUBLE, 0, world_comm);
	MPI_Bcast(Zk.memptr(), (sigma.n_cols) * nObservations, MPI_DOUBLE, 0, world_comm);

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

	delete[] zkdata;

	prevError = currError;
	currError = norm(error, 2);
	++currIt;

	return currError;
}

void MappedROUKF::reset(int nObservations, int nStates, int nParameters, vector<double> observationsUncertainty, vector<double> parametersUncertainty, SigmaPointsGenerator::SIGMA_DISTRIBUTION sigmaDistribution) {
	this->nObservations = nObservations;
	this->nStates = nStates;
	this->nParameters = nParameters;

	X = zeros(nStates, 1);
	Theta = zeros(nParameters, 1);

	LTheta = eye(nParameters, nParameters);
	LX = zeros(nStates, nParameters);
	U = eye(nParameters, nParameters);
	Wi = eye(nObservations, nObservations);

	mat diagParameter(&(parametersUncertainty[0]), 1, nParameters);
	U.diag() = 1. / diagParameter;
	mat diagState(&(observationsUncertainty[0]), 1, nObservations);
	Wi.diag() = 1. / diagState;

	SigmaPointsGenerator::generateSigmaPoints(nParameters, sigmaDistribution, &sigma);
	alpha = 1. / sigma.n_cols;

	Dsigma = alpha * eye(sigma.n_cols, sigma.n_cols);
	Dsigma = Dsigma * sigma.t();
	Pa = sigma * Dsigma;

}

void MappedROUKF::getParameters(double** thetac) {
	vector<double> theta(Theta.memptr(), Theta.memptr() + nParameters);
	theta = mapper->unmap(theta);
	*thetac = new double[nParameters];
	memcpy(*thetac, &(theta[0]), nParameters * sizeof(double));
}

void MappedROUKF::setParameters(double *thetac) {
	vector<double> theta(thetac, thetac + nParameters);
	theta = mapper->map(theta);
	Theta = mat(&(theta[0]), nParameters, 1);
}

void MappedROUKF::replaceMapper(CompositeParameterMapper* mapper){
	if(this->mapper){
		//	Remap kalman parameters from previous kalman parameters space into the new one.
		vector<double> thetakdata(Theta.memptr(), Theta.memptr() + nParameters);
		thetakdata = this->mapper->unmap(thetakdata);
		delete this->mapper;
		thetakdata = mapper->map(thetakdata);
		Theta = mat(&(thetakdata[0]), nParameters, 1);
	}
	this->mapper = mapper;
}
