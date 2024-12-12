from roukf import ROUKF, SigmaDistribution
import numpy as np

def forward_operator(x: np.ndarray, n_states: int, theta: np.ndarray, n_parameters: int) -> int:
    """
    Forward operator A(x, θ) that propagates the state
    Args:
        x: numpy array - State vector (modified in-place)
        n_states: int - Number of states
        theta: numpy array - Parameter vector (modified in-place)
        n_parameters: int - Number of parameters
    Returns:
        int: Status (1 for success, 0 for failure)
    """
    try:
        # Both x and theta can be modified in-place
        for i in range(min(n_states, n_parameters)):
            x[i] = theta[i] * x[i]
            theta[i] = theta[i] * 1.01  # Example modification of parameters
        return 1
    except Exception as e:
        print(f"Error in forward operator: {e}")
        return 0

def observation_operator(x: np.ndarray, n_states: int, z: np.ndarray, n_observations: int) -> None:
    """
    Observation operator H(x) that maps state to observations
    Args:
        x: numpy array - State vector (modified in-place) I don't know 
        n_states: int - Number of states
        z: numpy array - Observation vector (modified in-place)
        n_observations: int - Number of observations
    """
    try:
        for i in range(n_observations):
            z[i] = x[i] if i < n_states else 0.0
            x[i] = x[i] * 1.01
    except Exception as e:
        print(f"Error in observation operator: {e}")

def main():

    # Example dimensions
    n_observations = 10  # Number of observations
    n_states = 20       # Number of states
    n_parameters = 5    # Number of parameters to estimate

    # Initialize initialGuess and parametersUncertainties
    parameters_uncertainty = np.full(n_parameters, 1E8, dtype=np.float64)

    initial_guess = np.full(n_states, 2.5E5, dtype=np.float64)
    initial_parameters = np.full(n_parameters, 0, dtype=np.float64)

    # Create uncertainty arrays
    states_uncertainty = np.full(n_states, 1E2,dtype=np.float64)


    # Initialize ROUKF
    kalman_filter = ROUKF(
        n_observations,
        n_states,
        n_parameters,
        states_uncertainty,
        parameters_uncertainty,
        SigmaDistribution.SIMPLEX
    )

    initial_state = np.ascontiguousarray(initial_guess)
    initial_parameters = np.ascontiguousarray(initial_parameters)
    print("###########################")
    print(initial_state)
    print("###########################")

    # Set initial condition
    kalman_filter.setState(initial_state)
    kalman_filter.setParameters(initial_parameters)

    observations = np.ones(n_observations)


    for _ in range(3):
        # Execute a step of the Kalman filter
        
        error = kalman_filter.executeStep(
            observations, 
            forward_operator, 
            observation_operator
        )
        
        state_estimate = kalman_filter.getState()
        print(f"Error: {error:.6f}, State estimate: {state_estimate[:5]}")  # Print first 5 values

    
     


    # Reset the Kalman filter
    kalman_filter.reset(
        n_observations,
        n_states,
        n_parameters,
        states_uncertainty,
        parameters_uncertainty,
        SigmaDistribution.SIMPLEX
    )
    print("Kalman filter has been reset.")

if __name__ == "__main__":
    main()

