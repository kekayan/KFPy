from roukf import ROUKF, SigmaDistribution
import numpy as np


def forward_operator(x, n_states, out, n_out):

    out[:] = 2 * x[:n_out]
    return 0

def observation_operator(x, n_states, z, n_observations):

    z[:] = 0.5 * x[:n_observations]
    return 0

# Example dimensions
n_observations = 10  # Number of observations
n_states = 20       # Number of states
n_parameters = 5    # Number of parameters to estimate

# Initialize initialGuess and parametersUncertainties
initial_guess = np.full(n_parameters, 2.5E5, dtype=np.float64)
parameters_uncertainty = np.full(n_parameters, 1E8, dtype=np.float64)


# Create uncertainty arrays
states_uncertainty = np.ones(n_observations, dtype=np.float64)


# Initialize ROUKF
kalman_filter = ROUKF(
    n_observations,
    n_states,
    n_parameters,
    states_uncertainty,
    parameters_uncertainty,
    SigmaDistribution.SIMPLEX
)


# Set initial condition
kalman_filter.setState(initial_guess)


observations = np.random.rand(n_observations)

for _ in range(300):
    # Execute a step of the Kalman filter
    error = kalman_filter.executeStep(
        observations, 
        forward_operator, 
        observation_operator
    )
    print(f"Error after execution step: {error}")

# Get the state and parameter estimates
state_estimate = kalman_filter.getState()
print(f"State estimate: {state_estimate}")


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

