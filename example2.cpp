#include "ROUKF.h"
#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>

int forwardOperator(double* x, int nStates, double* theta, int nParameters);
void observationOperator(double* x, int nStates, double* z, int nObservations);
std::vector<double> dataGenerator(int numSteps, double X0 = 1.0, double Phi = 0.9);

std::vector<double> dataGenerator(int numSteps, double X0, double Phi) {
    std::vector<double> timeSeries(numSteps);
    double x1 = X0;
    for (int i = 0; i < numSteps; i++) {
        x1 = Phi * x1;
        timeSeries[i] = x1;
    }
    return timeSeries;
}

int forwardOperator(double* x, int nStates, double* theta, int nParameters) {
    try {
        x[0] = theta[0] * x[0];
        theta[0] = theta[0];
        return 1;
    } catch (...) {
        std::cerr << "Error in forward operator" << std::endl;
        return 0;
    }
}

void observationOperator(double* x, int nStates, double* z, int nObservations) {
    try {
        for (int i = 0; i < nObservations; i++) {
            z[i] = x[i];
        }
    } catch (...) {
        std::cerr << "Error in observation operator" << std::endl;
    }
}

int main() {
    const int numSteps = 10000;
    const int nObservations = 1;
    const int nStates = 1;
    const int nParameters = 1;
    
    // Generate observations
    std::vector<double> observations = dataGenerator(numSteps);
    
    double parametersUncertainties[nParameters] = {10.0};
    double statesUncertainties[nStates] = {1.0};
    
    double initialGuess[nStates] = {1.0};
    double initialParameters[nParameters] = {0.0};  

    ROUKF* kalmanFilter = new ROUKF(
        nObservations, 
        nStates, 
        nParameters, 
        statesUncertainties,
        parametersUncertainties, 
        SigmaPointsGenerator::SIGMA_DISTRIBUTION::CANONIC
    );

    kalmanFilter->setState(initialGuess);
    kalmanFilter->setParameters(initialParameters);

    // Main loop
    for (int i = 0; i < numSteps; i++) {
        double observe[nObservations] = {observations[i]};
        
        double error = kalmanFilter->executeStep(
            observe,
            forwardOperator, 
            observationOperator
        );
        
        double* stateEstimate = nullptr;
        double* params = nullptr;
        kalmanFilter->getState(&stateEstimate);
        kalmanFilter->getParameters(&params);

        std::cout << "Error: " << error << ", State estimate: " << stateEstimate[0] << std::endl;
        std::cout << "Parameters: " << params[0] << std::endl;
    }

    // Reset the Kalman filter
    kalmanFilter->reset(
        nObservations,
        nStates,
        nParameters,
        statesUncertainties,
        parametersUncertainties,
        SigmaPointsGenerator::SIGMA_DISTRIBUTION::CANONIC
    );
    std::cout << "Kalman filter has been reset." << std::endl;

    // Clean up
    delete kalmanFilter;
    return 0;
}