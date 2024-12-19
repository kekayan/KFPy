#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include "ROUKF.h"
#include "AbstractROUKF.h"
#include "SigmaPointsGenerator.h"
#include <iostream>

namespace py = pybind11;

class CallbackStorage {
public:
    static void setCallbacks(const py::function& forward, const py::function& observation) {
        py::gil_scoped_acquire gil;
        forward_func = forward;
        observation_func = observation;
    }
    static py::function forward_func;
    static py::function observation_func;
};

py::function CallbackStorage::forward_func;
py::function CallbackStorage::observation_func;

int forward_wrapper(double* states, int n_states, double* params, int n_params) {
    try {
        py::object result;
        {
            py::gil_scoped_acquire gil;  // Acquire GIL only for Python call

            // Create NumPy arrays that reference the existing C++ data without copying
            auto states_array = py::array_t<double>(
                {n_states},                                    // shape
                {sizeof(double)},                              // strides
                states,                                        // data pointer
                py::capsule(states, [](void *v) {})            // base object (empty capsule)
            );
            auto params_array = py::array_t<double>(
                {n_params},
                {sizeof(double)},
                params,
                py::capsule(params, [](void *v) {})
            );

            // Call the Python function
            result = CallbackStorage::forward_func(states_array, n_states, params_array, n_params);
        }  // GIL released here

        return result.cast<int>();
    } catch (const std::exception& e) {
        std::cerr << "Forward wrapper error: " << e.what() << std::endl;
        return -1;
    }
}

void observation_wrapper(double* states, int n_states, double* obs, int n_obs) {
    try {
        {
            py::gil_scoped_acquire gil;  // Acquire GIL only for Python call

            auto states_array = py::array_t<double>(
                {n_states},
                {sizeof(double)},
                states,
                py::capsule(states, [](void *v) {})
            );
            auto obs_array = py::array_t<double>(
                {n_obs},
                {sizeof(double)},
                obs,
                py::capsule(obs, [](void *v) {})
            );

            // Call the Python function
            CallbackStorage::observation_func(states_array, n_states, obs_array, n_obs);
        }  // GIL released here
    } catch (const std::exception& e) {
        std::cerr << "Observation wrapper error: " << e.what() << std::endl;
    }
}

PYBIND11_MODULE(roukf_py, m) {
    m.doc() = "ROUKF (Reduced-Order Unscented Kalman Filter) Python bindings";

    // Bind SigmaPointsGenerator::SIGMA_DISTRIBUTION enum
    py::enum_<SigmaPointsGenerator::SIGMA_DISTRIBUTION>(m, "SigmaDistribution")
        .value("SIMPLEX", SigmaPointsGenerator::SIMPLEX)
        .value("CANONIC", SigmaPointsGenerator::CANONIC)
        .value("STAR", SigmaPointsGenerator::STAR)
        .value("SIMPLEX_STAR", SigmaPointsGenerator::SIMPLEX_STAR)
        .export_values();

    // Bind the abstract base class
    py::class_<AbstractROUKF>(m, "AbstractROUKF")
        .def(py::init<>())
        .def("hasConverged", &AbstractROUKF::hasConverged)
        .def("getState", [](AbstractROUKF& self) {
            double* state;
            self.getState(&state);
            return py::array_t<double>({self.getStates()}, {sizeof(double)}, state, py::capsule(state, [](void *v) {}));
        })
        .def("setState", [](AbstractROUKF& self, py::array_t<double> array) {
            // Make sure the array is contiguous and in the correct format
            py::buffer_info buf = array.request();
            if (buf.ndim != 1) {
                throw std::runtime_error("Number of dimensions must be one");
            }
            
            // Create a copy of the data
            std::vector<double> data_copy(buf.shape[0]);
            std::memcpy(data_copy.data(), array.data(), buf.shape[0] * sizeof(double));
            
            // Pass the copied data to setState
            self.setState(data_copy.data());
        })
        .def("getError", [](AbstractROUKF& self) {
            double* error;
            self.getError(&error);
            return *error;
        })
        .def("getObsError", &AbstractROUKF::getObsError, py::arg("numObservation"))
        .def("getObservations", &AbstractROUKF::getObservations)
        .def("getStates", &AbstractROUKF::getStates)
        .def("getTolerance", &AbstractROUKF::getTolerance)
        .def("setTolerance", &AbstractROUKF::setTolerance, py::arg("tolerance"))
        .def("getMaxIterations", &AbstractROUKF::getMaxIterations)
        .def("setMaxIterations", &AbstractROUKF::setMaxIterations, py::arg("maxIterations"))
        .def("setParameters", [](AbstractROUKF& self, py::array_t<double> array) {
            
             // Make sure the array is contiguous and in the correct format
            py::buffer_info buf = array.request();
            if (buf.ndim != 1) {
                throw std::runtime_error("Number of dimensions must be one");
            }
            
            // Create a copy of the data
            std::vector<double> data_copy(buf.shape[0]);
            std::memcpy(data_copy.data(), array.data(), buf.shape[0] * sizeof(double));
            
            // Pass the copied data to setState
            self.setParameters(data_copy.data());
        })
        .def("getParameters", [](AbstractROUKF& self, int nParameters) {
            double* params;
            self.getParameters(&params);
            return py::array_t<double>({nParameters}, {sizeof(double)}, params, py::capsule(params, [](void *v) {}));
        });

    py::class_<ROUKF, AbstractROUKF>(m, "ROUKF")
        .def(py::init([](int n_observations, int n_states, int n_parameters,
                         py::array_t<double> states_uncertainty,
                         py::array_t<double> parameters_uncertainty,
                         SigmaPointsGenerator::SIGMA_DISTRIBUTION sigma_distribution) {
            return new ROUKF(n_observations, n_states, n_parameters,
                            states_uncertainty.mutable_data(),
                            parameters_uncertainty.mutable_data(),
                            sigma_distribution);
        }), R"pbdoc(
            Initialize a Reduced-Order Unscented Kalman Filter (ROUKF).

            Args:
                n_observations (int): Number of observations
                n_states (int): Number of states
                n_parameters (int): Number of parameters
                states_uncertainty (numpy.ndarray): Uncertainty for each observation, shape (n_observations,)
                parameters_uncertainty (numpy.ndarray): Uncertainty for each parameter, shape (n_parameters,)
                sigma_distribution (SigmaDistribution): Type of sigma points distribution

            Example:
                >>> roukf = ROUKF(
                ...     n_observations=3,
                ...     n_states=2,
                ...     n_parameters=2,
                ...     states_uncertainty=np.array([0.1, 0.1, 0.1]),
                ...     parameters_uncertainty=np.array([0.2, 0.2]),
                ...     sigma_distribution=SigmaDistribution.SIMPLEX
                ... )
        )pbdoc")
        .def("executeStep", [](ROUKF& self, py::array_t<double> observations,
                               py::function forward_func,
                               py::function observation_func) -> double {
            CallbackStorage::setCallbacks(forward_func, observation_func);

            double* obs_ptr = observations.mutable_data();

            return self.executeStep(
                obs_ptr,
                forward_wrapper,
                observation_wrapper
            );
        })
        .def("reset", [](ROUKF& self, int n_observations, int n_states, int n_parameters,
                         py::array_t<double> states_uncertainty,
                         py::array_t<double> parameters_uncertainty,
                         SigmaPointsGenerator::SIGMA_DISTRIBUTION sigma_distribution) {

            self.reset(n_observations, n_states, n_parameters,
                       states_uncertainty.mutable_data(),
                       parameters_uncertainty.mutable_data(),
                       sigma_distribution);
        });
}