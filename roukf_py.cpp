#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include "ROUKF.h"
#include "AbstractROUKF.h"
#include "SigmaPointsGenerator.h"
#include <iostream>

namespace py = pybind11;

// Thread-safe storage for Python callbacks
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

// Safe wrapper functions
int forward_wrapper(double* states, int n_states, double* params, int n_params) {
    std::cout << "forward_wrapper" << std::endl;
    
    try {
        std::cout << "forward_wrapper try" << std::endl;
        auto states_array = py::array_t<double>(n_states, states);
        auto params_array = py::array_t<double>(n_params, params);
        std::cout << "forward_wrapper arrays" << std::endl;
        py::object result = CallbackStorage::forward_func(states_array, n_states, params_array, n_params);
        std::cout << "forward_wrapper result" << std::endl;
        std::memcpy(states, states_array.data(), n_states * sizeof(double));
        std::memcpy(params, params_array.data(), n_params * sizeof(double));
        std::cout << "forward_wrapper memcpy" << std::endl;
        return result.cast<int>();
    } catch (const std::exception& e) {
        std::cerr << "Forward wrapper error: " << e.what() << std::endl;
        return -1;
    }
}

void observation_wrapper(double* states, int n_states, double* obs, int n_obs) {
    std::cout << "observation_wrapper" << std::endl;
    
    try {
        std::cout << "observation_wrapper try" << std::endl;
        auto states_array = py::array_t<double>(n_states, states);
        auto obs_array = py::array_t<double>(n_obs, obs);
        std::cout << "observation_wrapper arrays" << std::endl;
        CallbackStorage::observation_func(states_array, n_states, obs_array, n_obs);
        std::cout << "observation_wrapper callback" << std::endl;
        std::memcpy(states, states_array.data(), n_states * sizeof(double));
        std::memcpy(obs, obs_array.data(), n_obs * sizeof(double));
        std::cout << "observation_wrapper memcpy" << std::endl;
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
        .def("getState", [](AbstractROUKF& self) {
            double* state;
            self.getState(&state);
            py::array_t<double> result(self.getStates());
            std::memcpy(result.mutable_data(), state, self.getStates() * sizeof(double));
            return result;
        })
        .def("setState", [](AbstractROUKF& self, py::array_t<double> array) {
            self.setState(array.mutable_data());
        })
        .def("getError", [](AbstractROUKF& self) {
            double* error;
            self.getError(&error);
            return error;
        })
        .def("getObsError", &AbstractROUKF::getObsError)
        .def("getObservations", &AbstractROUKF::getObservations)
        .def("getStates", &AbstractROUKF::getStates)
        .def("getTolerance", &AbstractROUKF::getTolerance)
        .def("setTolerance", &AbstractROUKF::setTolerance)
        .def("getMaxIterations", &AbstractROUKF::getMaxIterations)
        .def("setMaxIterations", &AbstractROUKF::setMaxIterations)
        .def("hasConverged", &AbstractROUKF::hasConverged)
        .def("setParameters", [](AbstractROUKF& self, py::array_t<double> array) {
            self.setParameters(array.mutable_data());
        });

    // Bind the derived class with corrected reset method
    py::class_<ROUKF, AbstractROUKF>(m, "ROUKF")
        .def(py::init([](int n_observations, int n_states, int n_parameters,
                         py::array_t<double> states_uncertainty,
                         py::array_t<double> parameters_uncertainty,
                         SigmaPointsGenerator::SIGMA_DISTRIBUTION sigma_distribution) {
            return new ROUKF(n_observations, n_states, n_parameters,
                             states_uncertainty.mutable_data(),
                             parameters_uncertainty.mutable_data(),
                             sigma_distribution);
        }))
        .def("executeStep", [](ROUKF& self, py::array_t<double> observations,
                               py::function forward_func,
                               py::function observation_func) {
            std::cout << "executeStep" << std::endl;
            
            py::buffer_info obs_buf = observations.request();
            std::cout << "executeStep buffer info" << std::endl;
            
            // Store the Python callbacks
            CallbackStorage::setCallbacks(forward_func, observation_func);
            std::cout << "executeStep set callbacks" << std::endl;
            
            return self.executeStep(
                static_cast<double*>(obs_buf.ptr),
                forward_wrapper,
                observation_wrapper
            );
        })
        .def("executeStepParallel", [](ROUKF& self, py::array_t<double> observations,
                                      std::function<int(double*, int, double*, int)> forward_func,
                                      std::function<void(double*, int, double*, int)> observation_func,
                                      int seed) {
            py::buffer_info obs_buf = observations.request();
            auto forward_ptr = forward_func.target<int(*)(double*, int, double*, int)>();
            auto observation_ptr = observation_func.target<void(*)(double*, int, double*, int)>();
            if (!forward_ptr || !observation_ptr) {
                throw std::runtime_error("Function pointers cannot be null");
            }
            return self.executeStepParallel(
                static_cast<double*>(obs_buf.ptr),
                reinterpret_cast<forwardOp>(*forward_ptr),
                reinterpret_cast<observationOp>(*observation_ptr),
                seed,
                MPI_COMM_WORLD,
                MPI_COMM_WORLD
            );
        })
        .def("reset", [](ROUKF& self, int n_observations, int n_states, int n_parameters,
                         py::array_t<double> states_uncertainty,
                         py::array_t<double> parameters_uncertainty,
                         SigmaPointsGenerator::SIGMA_DISTRIBUTION sigma_distribution) {
            py::buffer_info states_buf = states_uncertainty.request();
            py::buffer_info params_buf = parameters_uncertainty.request();

            self.reset(n_observations, n_states, n_parameters,
                       states_uncertainty.mutable_data(),
                       parameters_uncertainty.mutable_data(),
                       sigma_distribution);
        });
}