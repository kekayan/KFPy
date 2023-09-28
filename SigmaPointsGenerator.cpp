/*
 * sigmaPointsGenerator.cpp
 *
 *  Created on: Nov 26, 2015
 *      Author: Gonzalo D. Maso Talou
 */

#include "SigmaPointsGenerator.h"

using namespace std;

void SigmaPointsGenerator::generateSigmaPoints(int nParameters,
		SIGMA_DISTRIBUTION distribution, arma::mat* sigma) {

	switch (distribution) {
	case CANONIC:
		canonicSigmaPoints(nParameters, sigma);
		break;
	case SIMPLEX:
		simplexSigmaPoints(nParameters, sigma);
		break;
	case STAR:
		starSigmaPoints(nParameters, sigma);
		break;
	case SIMPLEX_STAR:
		simplexStarSigmaPoints(nParameters,sigma);
		break;
	default:
		cout << "Unrecognize type of sigma point distribution" << endl;
		break;
	}
}

void SigmaPointsGenerator::canonicSigmaPoints(int nParameters,
		arma::mat* sigma) {
	int nSigmas = 2 * nParameters;

	arma::mat sigmas = arma::zeros(nParameters, nSigmas);
	sigmas.diag().fill(sqrt((double)nParameters));
	//	Some armadillo magic
	sigmas.cols(nParameters, nSigmas - 1) = fliplr(
			sigmas.cols(0, nParameters - 1) * -1);

	*sigma = sigmas;
}

void SigmaPointsGenerator::simplexSigmaPoints(int nParameters,
		arma::mat* sigma) {
	*sigma = getSimplexSigmaPoints(nParameters, 1. / (nParameters + 1));
}

void SigmaPointsGenerator::starSigmaPoints(int nParameters, arma::mat* sigma) {
	int nSigmas = 2 * nParameters + 1;

	arma::mat sigmas = arma::zeros(nParameters, nSigmas);
	sigmas.diag().fill(sqrt((2. * (double) nParameters + 1.) / 2.));
	//	Some armadillo magic
	sigmas.cols(nParameters, nSigmas - 2) = fliplr(
			sigmas.cols(0, nParameters - 1) * -1);

	*sigma = sigmas;
}

arma::mat SigmaPointsGenerator::getSimplexSigmaPoints(int nPoints,
		double weight) {

	arma::mat sigma = arma::zeros(nPoints, nPoints + 1);

	double currWeight = 1. / sqrt((double) (nPoints * (nPoints + 1)) * weight);
	if (nPoints == 1) {
		sigma.at(0, 0) = -currWeight;
		sigma.at(0, 1) = currWeight;
		return sigma;
	}
	sigma.submat(0, 0, nPoints - 2, nPoints - 1) = getSimplexSigmaPoints(
			nPoints - 1, weight);
	for (int col = 0; col < nPoints; col++)
		sigma.at(nPoints - 1, col) = -currWeight;

	sigma.at(nPoints - 1, nPoints) = nPoints * (currWeight);

	return sigma;
}

void SigmaPointsGenerator::simplexStarSigmaPoints(int nParameters,
		arma::mat* sigma) {
	arma::mat sigmas = arma::zeros(nParameters, nParameters + 2);
	sigmas.cols(0,nParameters) = getSimplexSigmaPoints(nParameters, 1. / (nParameters + 1));

	sigmas = sigmas * ( (nParameters+2)/(nParameters+1) );
	*sigma = sigmas;
}
