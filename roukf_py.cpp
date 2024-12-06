#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include "ROUKF.h"
#include "AbstractROUKF.h"
#include "SigmaPointsGenerator.h"
#include <mpi.h>

namespace py = pybind11;

// Static variables to hold the Python callables
static py::function py_forward_func;
static py::function py_observation_func;

// Wrapper functions that call the Python functions
int forward_wrapper(double* x, int n_states, double* out, int n_out) {
    // Create NumPy arrays without copying data
    py::array_t<double> x_array({n_states}, x);
    py::array_t<double> out_array({n_out}, out);

    // Call the Python function
    py::object result = py_forward_func(x_array, n_states, out_array, n_out);

    // Return the result as int
    return result.cast<int>();
}

void observation_wrapper(double* x, int n_states, double* z, int n_observations) {
    // Create NumPy arrays without copying data
    py::array_t<double> x_array({n_states}, x);
    py::array_t<double> z_array({n_observations}, z);

    // Call the Python function
    py_observation_func(x_array, n_states, z_array, n_observations);
}

PYBIND11_MODULE(roukf_py, m) {
    m.doc() = "ROUKF (Reduced-Order Unscented Kalman Filter) Python bindings";
    
    // Initialize MPI if needed
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized) {
        int argc = 0;
        char** argv = nullptr;
        MPI_Init(&argc, &argv);
    }

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
            auto buf = result.request();
            double* ptr = static_cast<double*>(buf.ptr);
            std::memcpy(ptr, state, self.getStates() * sizeof(double));
            return result;
        })
        .def("setState", [](AbstractROUKF& self, py::array_t<double> array) {
            py::buffer_info buf = array.request();
            double* ptr = static_cast<double*>(buf.ptr);
            self.setState(ptr);
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
            py::buffer_info buf = array.request();
            double* ptr = static_cast<double*>(buf.ptr);
            self.setParameters(ptr);
        });
        


    // Bind the derived class with corrected reset method
    py::class_<ROUKF, AbstractROUKF>(m, "ROUKF")
        .def(py::init([](int n_observations, int n_states, int n_parameters,
                         py::array_t<double> states_uncertainty,
                         py::array_t<double> parameters_uncertainty,
                         SigmaPointsGenerator::SIGMA_DISTRIBUTION sigma_distribution) {
            py::buffer_info states_buf = states_uncertainty.request();
            py::buffer_info params_buf = parameters_uncertainty.request();
            return new ROUKF(n_observations, n_states, n_parameters,
                             static_cast<double*>(states_buf.ptr),
                             static_cast<double*>(params_buf.ptr),
                             sigma_distribution);
        }))
        .def("executeStep", [](ROUKF& self, py::array_t<double> observations,
                               py::function forward_func,
                               py::function observation_func) {
            py::buffer_info obs_buf = observations.request();

            // Store the Python functions
            py_forward_func = forward_func;
            py_observation_func = observation_func;

            // Call executeStep with wrapper functions
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
                       static_cast<double*>(states_buf.ptr),
                       static_cast<double*>(params_buf.ptr),
                       sigma_distribution);
        });

    // Add cleanup handler
    atexit([]() {
        int finalized;
        MPI_Finalized(&finalized);
        if (!finalized) {
            MPI_Finalize();
        }
    });
}