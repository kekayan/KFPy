#include "ROUKF.h"
#include <iostream>
#include <algorithm>

int forwardOperator(double* x, int nStates, double* theta, int nParameters) {
    try {
        for (int i = 0; i < std::min(nStates, nParameters); i++) {
            x[i] = theta[i] * x[i];
            theta[i] = theta[i] * 1.01;
        }
        return 1;
    } catch (...) {
        std::cerr << "Error in forward operator" << std::endl;
        return 0;
    }
}

void observationOperator(double* x, int nStates, double* z, int nObservations) {
    try {
        for (int i = 0; i < nObservations; i++) {
            z[i] = (i < nStates) ? x[i] : 0.0;
            x[i] = x[i] * 1.01;
        }
    } catch (...) {
        std::cerr << "Error in observation operator" << std::endl;
    }
}

int main() {
    int nObservations = 10;
    int nStates = 20;
    int nParameters = 5;
    
    double initialGuess[nParameters];
    double parametersUncertainties[nParameters];
    double statesUncertainties[nStates];
   
    for (int i = 0; i < nParameters; i++) {
        parametersUncertainties[i] = 1E8;   // -> U0^{-1}
    }
    
    for (int i = 0; i < nStates; i++) {
        statesUncertainties[i] = 1E2;
        initialGuess[i] = 2.5E5;
    }

    // Initialize ROUKF
    ROUKF* roukf = new ROUKF(nObservations, nStates, nParameters, statesUncertainties,
                            parametersUncertainties, SigmaPointsGenerator::SIGMA_DISTRIBUTION::SIMPLEX);

    std::cout << "Initial guess: ";
    for (int i = 0; i < nStates; i++) {
        std::cout << initialGuess[i] << " ";
    }
    std::cout << std::endl;
   
    roukf->setState(initialGuess);


    double observation[nObservations];
    for (int i = 0; i < nObservations; i++) {
        observation[i] = 1.0;
    }

    for (int i = 0; i < 3; i++) {
        double error = roukf->executeStep(observation, forwardOperator, observationOperator);
        
        double* state = nullptr;
        roukf->getState(&state);
        
        std::cout << "Error: " << error << ", State estimate: ";
        for (int j = 0; j < std::min(5, nParameters); j++) {
            std::cout << state[j] << " ";
        }
        std::cout << std::endl;
    }

    // Clean up
    delete roukf;
    return 0;
}