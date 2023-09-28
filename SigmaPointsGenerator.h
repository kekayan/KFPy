/*
 * sigmaPointsGenerator.h
 *
 *  Created on: Nov 26, 2015
 *      Author: Gonzalo D. Maso Talou
 */

#ifndef SIGMAPOINTSGENERATOR_H_
#define SIGMAPOINTSGENERATOR_H_

#include <armadillo>

/**
 * Static class which generates different distributions of sigma points.
 */
class SigmaPointsGenerator {
public:
	/**	Enumeration for the different kinds of sigma points distributions. */
	enum SIGMA_DISTRIBUTION {SIMPLEX,CANONIC,STAR,SIMPLEX_STAR};

	/**
	 * Generates the matrix sigma where each column is a sigma point of the type @p distribution.
	 * @param nParameters	Quantity of parameters to estimate.
	 * @param distribution	Type of distribution used to generate the sigma points.
	 * @param sigma	Output matrix with one sigma point per column.
	 */
	static void generateSigmaPoints(int nParameters, SIGMA_DISTRIBUTION distribution, arma::mat*sigma);

protected:
	/**
	 * Canonic sigma points (2 points per parameter symmetrics to the centroid.)
	 * @param nParameters	Quantity of parameters
	 * @param sigma	Output matrix with one sigma point per column.
	 */
	static void canonicSigmaPoints(int nParameters, arma::mat *sigma);
	/**
	 * Simplex sigma points.
	 * @param nParameters	Quantity of parameters
	 * @param sigma	Output matrix with one sigma point per column.
	 */
	static void simplexSigmaPoints(int nParameters, arma::mat *sigma);
	/**
	 * Same as canonic sigma points with an additional point at the centroid.
	 * @param nParameters	Quantity of parameters
	 * @param sigma	Output matrix with one sigma point per column.
	 */
	static void starSigmaPoints(int nParameters, arma::mat *sigma);
	/**
	 * Same as simplex sigma points with an additional point at the centroid.
	 * @param nParameters	Quantity of parameters
	 * @param sigma	Output matrix with one sigma point per column.
	 */
	static void simplexStarSigmaPoints(int nParameters, arma::mat *sigma);

private:
	/**
	 * Recursive generation of simplex sigma points.
	 * @param nPoints	Quantity of points to generate.
	 * @param weight	Weight for the current generated points.
	 * @return	Matrix of sigma points.
	 */
	static arma::mat getSimplexSigmaPoints(int nPoints, double weight);
};

#endif /* SIGMAPOINTSGENERATOR_H_ */
