from roukf import ROUKF, SigmaDistribution
import numpy as np

def forward_operator(x: np.ndarray, n_states: int, theta: np.ndarray, n_parameters: int) -> int:
    try:
        x[0] = theta[0] * x[0]

        theta[0] = theta[0]
        return 1
    except Exception as e:
        print(f"Error in forward operator: {e}")
        return 0

def observation_operator(x: np.ndarray, n_states: int, z: np.ndarray, n_observations: int) -> None:
    try:
        for i in range(n_observations):
            z[i] = x[i]

    except Exception as e:
        print(f"Error in observation operator: {e}")

def data_generator(numSteps, X0 = np.array([1]), Phi = 0.9):
    timeSeries = np.zeros((1, numSteps))
    x1 = X0[0]
    for i in range(numSteps):
        x1 = Phi * x1
        timeSeries[0, i] = x1

    return timeSeries


num_steps = 10000
observations = data_generator(num_steps)
def main():

    # Example dimensions
    n_observations = 1  # Number of observations
    n_states = 1       # Number of states
    n_parameters = 1    # Number of parameters to estimate

    # Initialize initialGuess and parametersUncertainties
    parameters_uncertainty = np.array([10], dtype=np.float64)

    initial_guess = np.array([1], dtype=np.float64)
    initial_parameters = np.full(n_parameters, 0, dtype=np.float64)


    # Create uncertainty arrays
    states_uncertainty = np.array([1], dtype=np.float64)


    # Initialize ROUKF
    kalman_filter = ROUKF(
        n_observations,
        n_states,
        n_parameters,
        states_uncertainty,
        parameters_uncertainty,
        SigmaDistribution.CANONIC
    )

    initial_state = np.ascontiguousarray(initial_guess)
    initial_parameter = np.ascontiguousarray(initial_parameters)
    print("###########################")
    print(initial_state)
    print("###########################")

    # Set initial condition
    kalman_filter.setState(initial_state)
    kalman_filter.setParameters(initial_parameter)



    for i in range(num_steps):
        # Execute a step of the Kalman filter
        observe = observations[:, i]
        error = kalman_filter.executeStep(
            observe,
            forward_operator, 
            observation_operator
        )
        
        state_estimate = kalman_filter.getState()
        params = kalman_filter.getParameters(n_parameters)


        print(f"Error: {error}, State estimate: {state_estimate[:5]}")  # Print first 5 values
        print(params)



    # Reset the Kalman filter
    kalman_filter.reset(
        n_observations,
        n_states,
        n_parameters,
        states_uncertainty,
        parameters_uncertainty,
        SigmaDistribution.CANONIC
    )
    print("Kalman filter has been reset.")

if __name__ == "__main__":
    main()
