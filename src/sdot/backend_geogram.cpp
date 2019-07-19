/* #define WITH_PROFILING */

#include <cmath>
#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

namespace py = pybind11;

#include <Eigen/Dense>
#include <Eigen/Sparse>

typedef Eigen::VectorXd VectorXd;
typedef Eigen::Vector3d Vector3d;
typedef Eigen::MatrixXd MatrixXd;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXd_RowMajor;
typedef Eigen::MatrixXi MatrixXi;
typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXi_RowMajor;
typedef Eigen::Triplet<double> Triplet;
typedef Eigen::SparseMatrix<double> SparseMatrix;

#include <geogram/basic/common.h>
#include <geogram/basic/command_line.h>
#include <geogram/basic/command_line_args.h>
#include <geogram/basic/file_system.h>
#include <geogram/mesh/mesh.h>
#include <geogram/mesh/mesh_io.h>
#include <geogram/mesh/mesh_tetrahedralize.h>
#include <geogram/NL/nl.h>
#include <geogram/NL/nl_matrix.h>
#include <exploragram/optimal_transport/optimal_transport.h>
#include <exploragram/optimal_transport/optimal_transport_2d.h>
#include <exploragram/optimal_transport/optimal_transport_3d.h>
#include <exploragram/optimal_transport/optimal_transport_on_surface.h>
#include <exploragram/optimal_transport/sampling.h>

#ifdef WITH_PROFILING
#include <chrono>
#endif

using namespace GEO;

static bool geogram_initialized = false;

void geogram_init(bool with_log) {
    if (geogram_initialized) {
        return;
    }

    geogram_initialized = true;

    GEO::initialize();
    CmdLine::import_arg_group("standard");
    CmdLine::import_arg_group("algo");

    if (! with_log) {
        Logger::instance()->set_minimal(true);
    }
}

typedef std::function<double(Vector3d const&)> Density_function;

// OTMType = 1 => 2D Surface / 2D cloud
// OTMType = 2 => 3D Surface / 3D cloud
// OTMType = 3 => 3D Tetrahedral mesh / 3D cloud
template < int OTMType >
class Density {
    static_assert(OTMType == 1 || OTMType == 2 || OTMType == 3, "OTMType must be 1, 2 or 3");

    public:
        Density (MatrixXd const& X, MatrixXi const& T,
                 Density_function const& values_function) {
            if (! geogram_initialized) {
                throw std::runtime_error("geogram must be initialized. Call geogram_init(bool) before.");
            }

            set_source_mesh(X, T);
            set_values(values_function);
        }

        void init_otm () {
            if (OTMType == 1) {
                M.vertices.set_dimension(3);
            } else if (OTMType == 2 || OTMType == 3) {
                M.vertices.set_dimension(4);
            }

            if (! weight.is_bound()) {
                weight.bind_if_is_defined(M.vertices.attributes(), "weight");
                if (! weight.is_bound()) {
                    /* set_density(M, 1.0, 1.0, "x"); */
                    weight.create_vector_attribute(M.vertices.attributes(), "weight", 1);
                }
            }

            if (OTMType == 1) {
                otm = new OptimalTransportMap2d(&M);
            } else if (OTMType == 2) {
                otm = new OptimalTransportMapOnSurface(&M);
            } else if (OTMType == 3) {
                otm = new OptimalTransportMap3d(&M);
            }

            otm->set_Newton(true);
            otm->set_verbose(false);
        }

        void set_source_mesh (MatrixXd_RowMajor const& X, MatrixXi_RowMajor const& T) {
            if (otm != nullptr) {
                delete otm;
                otm = nullptr;
            }

            vector<double> vertices(X.cols() * X.rows());
            std::copy(X.data(), X.data() + X.cols() * X.rows(),
                      vertices.begin());

            vector<index_t> triangles(T.cols() * T.rows());
            std::copy(T.data(), T.data() + T.cols() * T.rows(),
                      triangles.begin());

            this->vertices = X;
            this->triangles = T;

            M.facets.assign_triangle_mesh(3, vertices, triangles, false);

            // Tetrahedrization
            if ((OTMType == 3) && (M.cells.nb() == 0)) {
                Logger::out("I/O") << "Trying to tetrahedralize..." << std::endl;
                if(! mesh_tetrahedralize(M, true, false)) {
                    throw std::runtime_error("Error when tetrahedralizing");
                }
            }

            init_otm();
        }

        void set_values (Density_function const& values_function) {
            this->values = values_function;

            for (index_t i = 0; i < M.vertices.nb(); ++i) {
                double* p = M.vertices.point_ptr(i);
                Vector3d x(p[0], p[1], p[2]);
                weight[i] = values(x);
            }
        }

        double mass () const {
            if (OTMType == 1) {
                OptimalTransportMap2d* true_otm = static_cast<OptimalTransportMap2d*>(otm);
                return true_otm->total_mesh_mass();
            } else if (OTMType == 2) {
                OptimalTransportMapOnSurface* true_otm = static_cast<OptimalTransportMapOnSurface*>(otm);
                return true_otm->total_mesh_mass();
            }

            assert(OTMType == 3);
            OptimalTransportMap3d* true_otm = static_cast<OptimalTransportMap3d*>(otm);
            return true_otm->total_mesh_mass();
        }

        MatrixXd centroids (MatrixXd_RowMajor const& Y, VectorXd const& psi) {
            // Set Y
            Mesh M2;
            M2.vertices.assign_points(Y.data(), 3, Y.rows());
            otm->set_points(M2.vertices.nb(), M2.vertices.point_ptr(0), 3);

            int N = otm->nb_points(), d = otm->dimension();

            // Compute stuff
            double f = 0.0;
            std::vector<double> gk(N);
            std::fill(gk.begin(), gk.end(), 0.0);

            std::vector<double> pk(N);

            VectorXd P = -psi;
            std::vector<double> xk(P.data(), P.data() + P.size());

            otm->new_linear_system(N, pk.data());
            otm->eval_func_grad_Hessian(N, xk.data(), f, gk.data());

            // Compute centroids
            MatrixXd_RowMajor C(N, d);
            otm->compute_Laguerre_centroids(C.data());

            return C;
        }

        std::pair<VectorXd, SparseMatrix>
        kantorovich (MatrixXd_RowMajor const& Y, VectorXd const& nu, VectorXd const& psi) {
#ifdef WITH_PROFILING
            auto start_init = std::chrono::high_resolution_clock::now();
#endif
            Mesh M2;
            M2.vertices.assign_points(Y.data(), 3, Y.rows());
            otm->set_points(M2.vertices.nb(), M2.vertices.point_ptr(0), 3);

            index_t N = otm->nb_points();
            for (index_t i = 0; i < N; ++i) {
                /* otm->set_nu(i, nu(i) * mass() / N); */
                otm->set_nu(i, nu(i));
            }

            double f = 0.0;
            std::vector<double> gk(N);
            std::fill(gk.begin(), gk.end(), 0.0);

            std::vector<double> pk(N);

            // geogram definition of Power cell = || x - y_i ||^2 - \psi_i
            VectorXd P = -psi;
            std::vector<double> xk(P.data(), P.data() + P.size());
#ifdef WITH_PROFILING
            auto end_init = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float> diff_init_seconds = end_init - start_init;
            std::cerr << "geogram_kantorovich_initialization: " << diff_init_seconds.count() << std::endl;
#endif

#ifdef WITH_PROFILING
            auto start_compute = std::chrono::high_resolution_clock::now();
#endif
            // Solve H pk = -gk (pk = solve_graph_laplacian(H, -pk))
            otm->new_linear_system(N, pk.data());
            otm->eval_func_grad_Hessian(N, xk.data(), f, gk.data());
            /* otm->solve_linear_system(); */
#ifdef WITH_PROFILING
            auto end_compute = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float> diff_compute_seconds = end_compute - start_compute;
            std::cerr << "geogram_kantorovich_compute: " << diff_compute_seconds.count() << std::endl;
#endif

#ifdef WITH_PROFILING
            auto start_matrix = std::chrono::high_resolution_clock::now();
#endif
            // in the source: gk = A - nu
            VectorXd G = VectorXd::Map(gk.data(), gk.size());
            VectorXd A = G + nu;

            SparseMatrix DA(N, N);
            std::vector<Triplet> triplets;

            NLSparseMatrix* M = nlGetCurrentSparseMatrix();

            // rows
            // TODO: remove loops
            for (NLuint i = 0; i < M->m; ++i) {
                NLRowColumn row = M->row[i];
                for (NLuint c = 0; c < row.size; ++c) {
                    NLCoeff coeff = row.coeff[c];
                    if (i == coeff.index) {
                        continue;
                    }

                    triplets.emplace_back(i, coeff.index, -coeff.value);
                }
            }

            // diagonal
            for (NLuint i = 0; i < M->diag_size; ++i) {
                triplets.emplace_back(i, i, -M->diag[i]);
            }

            DA.setFromTriplets(triplets.begin(), triplets.end());
            DA.makeCompressed();
#ifdef WITH_PROFILING
            auto end_matrix = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float> diff_matrix_seconds = end_matrix - start_matrix;
            std::cerr << "geogram_kantorovich_matrix: " << diff_matrix_seconds.count() << std::endl;
#endif

            return {A, DA};
        }

        void save_diagram (std::string const& fname) const {
            Mesh M_diagram;
            otm->get_RVD(M_diagram);

            MeshIOFlags flags;

            if (OTMType == 1) {
            } else if (OTMType == 2) {
            } else if (OTMType == 3) {
                // fname must be a '.tet' or '.tet6' file (readable by vorpaview)
                flags.set_attribute(MESH_FACET_REGION);
                flags.set_attribute(MESH_CELL_REGION);
                flags.set_element(MESH_CELLS);
            }

            mesh_save(M_diagram, fname, flags);
        }

        VectorXd optimal_transport (MatrixXd_RowMajor const& Y, VectorXd const& nu, VectorXd const& psi0,
                                    double eps, bool verbose, int maxit) {
            // Target measure
            Mesh M2;
            M2.vertices.assign_points(Y.data(), 3, Y.rows());
            otm->set_points(M2.vertices.nb(), M2.vertices.point_ptr(0), 3);

            index_t N = otm->nb_points();
            for (index_t i = 0; i < N; ++i) {
                otm->set_nu(i, nu(i));
            }

            // Initial weights
            for (index_t i = 0; i < N; ++i) {
                otm->set_initial_weight(i, -psi0(i));
            }

            // Solve OT
            otm->set_Newton(true);
            otm->set_epsilon(eps);
            otm->set_verbose(verbose);

            otm->optimize_full_Newton(maxit);

            // Final weights
            VectorXd psi(N);
            for (index_t i = 0; i < N; ++i) {
                psi(i) = -otm->weight(i);
            }

            return psi;
        }

        ~Density () {
            /* std::cout << "dtor" << std::endl; */
            if (otm != nullptr) {
                delete otm;
                otm = nullptr;
            }
        }

    private:
        MatrixXd vertices;
        MatrixXi triangles;
        Attribute<double> weight;
        Density_function values;

        Mesh M;
        OptimalTransportMap* otm = nullptr;
};

using Density_3 = Density<3>; // 3D
using Density_2 = Density<2>; // Surface
using Density_1 = Density<1>; // 2D

PYBIND11_MODULE(backend_geogram, m) {
    m.doc() = "Restriced Power diagrams with geogram";

    m.def("geogram_init", &geogram_init);

    py::class_<Density_3>(m, "Density_3")
        .def(py::init<MatrixXd const&, MatrixXi const&, Density_function const&>())
        .def("kantorovich", &Density_3::kantorovich)
        .def("centroids", &Density_3::centroids)
        .def("save_diagram", &Density_3::save_diagram)
        .def("set_values", &Density_3::set_values)
        .def("set_source_mesh", &Density_3::set_source_mesh)
        .def("optimal_transport", &Density_3::optimal_transport)
        .def("mass", &Density_3::mass);

    py::class_<Density_2>(m, "Density_2")
        .def(py::init<MatrixXd const&, MatrixXi const&, Density_function const&>())
        .def("kantorovich", &Density_2::kantorovich)
        .def("centroids", &Density_2::centroids)
        .def("save_diagram", &Density_2::save_diagram)
        .def("set_values", &Density_2::set_values)
        .def("optimal_transport", &Density_2::optimal_transport)
        .def("mass", &Density_2::mass);

    py::class_<Density_1>(m, "Density_1")
        .def(py::init<MatrixXd const&, MatrixXi const&, Density_function const&>())
        .def("kantorovich", &Density_1::kantorovich)
        .def("centroids", &Density_1::centroids)
        .def("save_diagram", &Density_1::save_diagram)
        .def("set_values", &Density_1::set_values)
        .def("optimal_transport", &Density_1::optimal_transport)
        .def("mass", &Density_1::mass);
}

