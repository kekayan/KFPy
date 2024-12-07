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
    py::gil_scoped_acquire gil;
    std::cerr << "Entering forward_wrapper" << std::endl;
    
    try {
        // Create new arrays with copy
        auto states_array = py::array_t<double>(n_states);
        auto params_array = py::array_t<double>(n_params);
        
        // Copy data into arrays
        auto states_buf = states_array.request();
        auto params_buf = params_array.request();
        std::memcpy(states_buf.ptr, states, n_states * sizeof(double));
        std::memcpy(params_buf.ptr, params, n_params * sizeof(double));
        
        std::cerr << "Calling Python forward function" << std::endl;
        // Call Python function
        py::object result = CallbackStorage::forward_func(states_array, n_states, params_array, n_params);
        
        // Copy data back
        std::memcpy(states, states_buf.ptr, n_states * sizeof(double));
        std::memcpy(params, params_buf.ptr, n_params * sizeof(double));
        
        std::cerr << "Exiting forward_wrapper" << std::endl;
        return result.cast<int>();
    } catch (const std::exception& e) {
        std::cerr << "Forward wrapper error: " << e.what() << std::endl;
        return -1;
    }
}

void observation_wrapper(double* states, int n_states, double* obs, int n_obs) {
    py::gil_scoped_acquire gil;
    std::cerr << "Entering observation_wrapper" << std::endl;
    
    try {
        // Create new arrays with copy
        auto states_array = py::array_t<double>(n_states);
        auto obs_array = py::array_t<double>(n_obs);
        
        // Copy data into arrays
        auto states_buf = states_array.request();
        auto obs_buf = obs_array.request();
        std::memcpy(states_buf.ptr, states, n_states * sizeof(double));
        std::memcpy(obs_buf.ptr, obs, n_obs * sizeof(double));
        
        std::cerr << "Calling Python observation function" << std::endl;
        // Call Python function
        CallbackStorage::observation_func(states_array, n_states, obs_array, n_obs);
        
        // Copy data back
        std::memcpy(states, states_buf.ptr, n_states * sizeof(double));
        std::memcpy(obs, obs_buf.ptr, n_obs * sizeof(double));
        
        std::cerr << "Exiting observation_wrapper" << std::endl;
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
            std::cerr << "Entering getState" << std::endl;
            double* state;
            self.getState(&state);
            py::array_t<double> result(self.getStates());
            auto buf = result.request();
            double* ptr = static_cast<double*>(buf.ptr);
            std::memcpy(ptr, state, self.getStates() * sizeof(double));
            std::cerr << "Exiting getState" << std::endl;
            return result;
        })
        .def("setState", [](AbstractROUKF& self, py::array_t<double> array) {
            std::cerr << "Entering setState" << std::endl;
            py::buffer_info buf = array.request();
            double* ptr = static_cast<double*>(buf.ptr);
            self.setState(ptr);
            std::cerr << "Exiting setState" << std::endl;
        })
        .def("getError", [](AbstractROUKF& self) {
            std::cerr << "Entering getError" << std::endl;
            double* error;
            self.getError(&error);
            std::cerr << "Exiting getError" << std::endl;
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
            std::cerr << "Entering setParameters" << std::endl;
            py::buffer_info buf = array.request();
            double* ptr = static_cast<double*>(buf.ptr);
            self.setParameters(ptr);
            std::cerr << "Exiting setParameters" << std::endl;
        });

    // Bind the derived class with corrected reset method
    py::class_<ROUKF, AbstractROUKF>(m, "ROUKF")
        .def(py::init([](int n_observations, int n_states, int n_parameters,
                         py::array_t<double> states_uncertainty,
                         py::array_t<double> parameters_uncertainty,
                         SigmaPointsGenerator::SIGMA_DISTRIBUTION sigma_distribution) {
            std::cerr << "Entering ROUKF constructor" << std::endl;
            py::buffer_info states_buf = states_uncertainty.request();
            py::buffer_info params_buf = parameters_uncertainty.request();
            auto instance = new ROUKF(n_observations, n_states, n_parameters,
                                      static_cast<double*>(states_buf.ptr),
                                      static_cast<double*>(params_buf.ptr),
                                      sigma_distribution);
            std::cerr << "Exiting ROUKF constructor" << std::endl;
            return instance;
        }))
        .def("executeStep", [](ROUKF& self, py::array_t<double> observations,
                               py::function forward_func,
                               py::function observation_func) {
            std::cerr << "Entering executeStep" << std::endl;
            py::gil_scoped_release release;
            
            py::buffer_info obs_buf = observations.request();
            
            // Store the Python callbacks
            {
                py::gil_scoped_acquire gil;
                CallbackStorage::setCallbacks(forward_func, observation_func);
            }
            
            auto result = self.executeStep(
                static_cast<double*>(obs_buf.ptr),
                forward_wrapper,
                observation_wrapper
            );
            std::cerr << "Exiting executeStep" << std::endl;
            return result;
        }, "Execute a single step of the ROUKF algorithm",
           py::arg("observations"),
           py::arg("forward_func"),
           py::arg("observation_func"))
        .def("executeStepParallel", [](ROUKF& self, py::array_t<double> observations,
                                      std::function<int(double*, int, double*, int)> forward_func,
                                      std::function<void(double*, int, double*, int)> observation_func,
                                      int seed) {
            std::cerr << "Entering executeStepParallel" << std::endl;
            py::buffer_info obs_buf = observations.request();
            auto forward_ptr = forward_func.target<int(*)(double*, int, double*, int)>();
            auto observation_ptr = observation_func.target<void(*)(double*, int, double*, int)>();
            if (!forward_ptr || !observation_ptr) {
                throw std::runtime_error("Function pointers cannot be null");
            }
            auto result = self.executeStepParallel(
                static_cast<double*>(obs_buf.ptr),
                reinterpret_cast<forwardOp>(*forward_ptr),
                reinterpret_cast<observationOp>(*observation_ptr),
                seed,
                MPI_COMM_WORLD,
                MPI_COMM_WORLD
            );
            std::cerr << "Exiting executeStepParallel" << std::endl;
            return result;
        })
        .def("reset", [](ROUKF& self, int n_observations, int n_states, int n_parameters,
                         py::array_t<double> states_uncertainty,
                         py::array_t<double> parameters_uncertainty,
                         SigmaPointsGenerator::SIGMA_DISTRIBUTION sigma_distribution) {
            std::cerr << "Entering reset" << std::endl;
            py::buffer_info states_buf = states_uncertainty.request();
            py::buffer_info params_buf = parameters_uncertainty.request();

            self.reset(n_observations, n_states, n_parameters,
                       static_cast<double*>(states_buf.ptr),
                       static_cast<double*>(params_buf.ptr),
                       sigma_distribution);
            std::cerr << "Exiting reset" << std::endl;
        });
}