/*
 * AbstractROUKF.cpp
 *
 *  Created on: Feb 26, 2018
 *      Author: Gonzalo D. Maso Talou
 */

#include "AbstractROUKF.h"

AbstractROUKF::AbstractROUKF() {
	currIt = 0;
	currError = 0;
	prevError = 0;
}

AbstractROUKF::~AbstractROUKF() {
}

void AbstractROUKF::getParameters(double** thetac) {
	*thetac = Theta.memptr();
}

void AbstractROUKF::setParameters(double *thetac) {
	Theta = mat(thetac, nParameters, 1);
}

void AbstractROUKF::getState(double** xc) {
	*xc = X.memptr();
}

void AbstractROUKF::setState(double* xc) {
	X = mat(xc, nStates, 1);
}

void AbstractROUKF::getError(double** err) {
	*err = error.memptr();
}

double AbstractROUKF::getObsError(int numObservation) {
	return error.at(numObservation);
}

void AbstractROUKF::toString() {

	X.print("X:");
	Theta.print("Theta:");
	U.print("U:");
	LX.print("LX:");
	LTheta.print("LTheta:");
	sigma.print("sigma:");
	Dsigma.print("Dsigma:");
	Pa.print("Pa:");
	Wi.print("Wi:");

}

vector<double> AbstractROUKF::getParametersStd() {
	mat std = sqrt(1. / U.diag());
	return arma::conv_to<vector<double> >::from(std);
}

int AbstractROUKF::getObservations() const
{
	return nObservations;
}

int AbstractROUKF::getStates() const
{
	return nStates;
}

double AbstractROUKF::getMaxIterations() const
{
	return maxIterations;
}

void AbstractROUKF::setMaxIterations(double maxIterations)
		{
	this->maxIterations = maxIterations;
}

double AbstractROUKF::getTolerance() const
{
	return tolerance;
}

void AbstractROUKF::setTolerance(double tolerance)
		{
	this->tolerance = tolerance;
}

bool AbstractROUKF::hasConverged(bool isConvergenceRelative) {
	if (currIt > 1) {
		double diff = abs(currError - prevError);

		if (isConvergenceRelative)
			diff /= prevError;

		return diff < tolerance;
	}
	return false;
}
