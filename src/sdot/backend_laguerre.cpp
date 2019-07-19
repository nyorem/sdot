/* #define WITH_PROFILING */

#ifdef WITH_PROFILING
#include <chrono>
#endif

#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace py = pybind11;

#include "types.hpp"
#include "density.hpp"
#include "power_sphere_triangulation_intersection.hpp"

// CGAL
typedef CGAL::Polygon_3<Kernel_ad> Polygon_ad;

// Overloads non defined by CGAL
namespace CGAL {
    template < typename K >
    typename CGAL::Point_3<K>
    operator* (typename K::FT const& a, typename CGAL::Point_3<K> const& p) {
        return typename CGAL::Point_3<K>(a * p.x(), a * p.y(), a * p.z());
    }

    template < typename K >
    typename K::FT
    squared_length (typename CGAL::Point_3<K> const& p) {
        return p.x() * p.x() + p.y() * p.y() + p.z() * p.z();
    }
} // namespace CGAL

// Eigen
typedef Eigen::VectorXi VectorXi;
typedef Eigen::VectorXd VectorXd;
typedef Eigen::MatrixXd MatrixXd;
typedef Eigen::MatrixXi MatrixXi;
typedef Eigen::Triplet<double> Triplet;
typedef Eigen::SparseMatrix<double> SparseMatrix;
typedef Eigen::SparseVector<double> SparseVector;

// Identity (quadratic OT)
template < typename Kernel >
struct Identity {
    typedef typename Kernel::Point_3 Point;
    typedef typename Kernel::FT FT;

    static std::pair<Point, FT>
    transform (Point const& y, FT const& lambda) {
        return std::make_pair(y, lambda);
    }
};

namespace internal {
    // TODO: remove
    std::vector<Point> points_from_matrix (Eigen::MatrixXd const& p) {
        std::vector<Point> points;
        for (int i = 0; i < p.rows(); ++i) {
            points.emplace_back(p(i, 0), p(i, 1), p(i, 2));
        }
        return points;
    }

    // A, DA, Centroids, Dual Triangulation
    struct Results {
        VectorXd A;
        SparseMatrix DA;
        MatrixXd C;
        MatrixXi tri_map;
        VectorXi number_components;
        MatrixXi index_map;
        double f = 0.0;

        using tuple_type = std::tuple<VectorXd, SparseMatrix, MatrixXd, MatrixXi, VectorXi, MatrixXi, double>;

        Results (VectorXd const& A, SparseMatrix const& DA, MatrixXd const& C,
                 MatrixXi const& tri_map, double f) : A(A), DA(DA), C(C), tri_map(tri_map), f(f) {
        }

        Results (VectorXd const& A, SparseMatrix const& DA, MatrixXd const& C,
                 MatrixXi const& tri_map, VectorXi const& N, MatrixXi const& I, double f)
            : A(A), DA(DA), C(C), tri_map(tri_map), number_components(N), index_map(I), f(f) {
        }

        tuple_type to_tuple () const {
            return std::make_tuple(A, DA, C, tri_map,
                                   number_components, index_map,
                                   f);
        }
    };

    template < typename F, typename Density >
    Results
    compute (Surface_triangulation const& tri,
             Density const& density,
             std::vector<Point> const& points,
             std::vector<double> const& weights) {

        // BBOX points are ALWAYS at the end
        const int M = weights.size() - 8;

        // Convert to AD
        std::vector<std::pair<Weighted_point, Info>> prt;
        const int N = points.size();
        for (int i = 0; i < N; ++i) {
            Point point = points[i];
            FT weight = weights[i];

            AD yx(point.x(), N), yy(point.y(), N), yz(point.z(), N);
            Point_ad y(yx, yy, yz);
            AD ww(weight, N, i);

            Point_ad p; FT_ad w;
            std::tie(p, w) = F::transform(y, ww);

            Weighted_point_ad wpad(p, w);
            prt.emplace_back(Weighted_point(Point(p.x().value(), p.y().value(), p.z().value()),
                                            w.value()), Info(i, wpad));
        }

        RT rt(prt.begin(), prt.end());

        // Compute
        Eigen::VectorXd A = Eigen::VectorXd::Zero(M);
        std::vector<Triplet> triplets;

        // centroid_map: (v->info().index, comp) = index
        std::map<std::pair<int, int>, int> centroid_map;
        int NC = 0;

        // number_components_map: v->info().index -> number of components of \Lag_v(\psi)
        Eigen::VectorXi number_components_map = Eigen::VectorXi::Zero(M);

        FT_ad wass = 0.0; // Wassertstein 2 distance

#ifdef WITH_PROFILING
        float total_integrate = 0;
#endif

        std::map<std::pair<RT::Vertex_handle, int>, std::vector<Polygon_ad>> laguerre_map;
        auto computeArea = [&] (Triangle* const& t, RT::Vertex_handle const& v, int comp,
                                Polygon_ad const& poly, std::vector<int> const& boundary_edges) {
#ifdef WITH_PROFILING
            auto start_integrate = std::chrono::high_resolution_clock::now();
#endif
            laguerre_map[std::make_pair(v, comp)].push_back(poly);
            int i = v->info().index;

            if (centroid_map.count(std::make_pair(i, comp)) == 0) {
                centroid_map[std::make_pair(i, comp)] = NC++;
            }

            if (i != -1 && i < M) {
                number_components_map[i] = std::max(comp, number_components_map[i]);
            }

            // Computations
            // Area
            FT_ad curarea = MA::integrate(density[t], poly.vertices());

            // Derivatives
            if (i != -1 && i < M) {
                A[i] += curarea.value();
                for (SparseVector::InnerIterator vit(curarea.derivatives()); vit; ++vit) {
                    if (vit.row() < M) {
                        triplets.emplace_back(i, vit.row(), vit.value());
                    }
                }
            }

            // Distance
            typename Density::Scalar_function wasserstein = [&] (Point_ad const& x) {
                              Point xx(x.x().value(), x.y().value(), x.z().value());
                              return (xx - v->point().point()).squared_length();
                          };

            wass += MA::integrate(wasserstein, poly.vertices());
#ifdef WITH_PROFILING
            auto end_integrate = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float> diff_integrate_seconds = end_integrate - start_integrate;
            total_integrate += diff_integrate_seconds.count();
#endif
        };

#ifdef WITH_PROFILING
        auto start_inter = std::chrono::high_resolution_clock::now();
#endif
        auto dual_tri = MA::power_sphere_triangulation_intersection_ad<Kernel_ad>(tri, rt, computeArea);
#ifdef WITH_PROFILING
        auto end_inter = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> diff_inter_seconds = end_inter - start_inter;
        std::cerr << "reflector_inter: " << diff_inter_seconds.count() << std::endl;
        std::cerr << "reflector_integrate: " << total_integrate << std::endl;
#endif

        SparseMatrix DA(M, M);
        DA.setFromTriplets(triplets.begin(), triplets.end());
        DA.makeCompressed();

        // Dual triangulation
        MatrixXi dual(dual_tri.size(), 3);
        int nt = 0;
        for (const auto& t: dual_tri) {
            std::pair<int, int> i, j, k;
            std::tie(i, j, k) = t;
            dual(nt, 0) = i.first;
            dual(nt, 1) = j.first;
            dual(nt, 2) = k.first;

            nt++;
        }

        // Centroids
        MatrixXd C(N, 3);
        for (const auto& kv: laguerre_map) {
            Point_ad c = CGAL::centroid(kv.second);
            int i = kv.first.first->info().index;
            C(i, 0) = c.x().value();
            C(i, 1) = c.y().value();
            C(i, 2) = c.z().value();
        }

        return Results(A, DA, C, dual, wass.value());
    }
} // namespace internal

class Compute_area_derivatives {
    public:
        Compute_area_derivatives (std::string const& filename) {
            tri_ = new Surface_triangulation(filename);
        }

        Compute_area_derivatives (Eigen::MatrixXd const& X,
                                  Eigen::MatrixXi const& T) {
            std::vector<Point> points;
            for (int i = 0; i < X.rows(); ++i) {
                points.emplace_back(X(i, 0), X(i, 1), X(i, 2));
            }

            std::vector<std::tuple<int, int, int>> triangles;
            for (int i = 0; i < T.rows(); ++i) {
                triangles.emplace_back(T(i, 0), T(i, 1), T(i, 2));
            }

            tri_ = new Surface_triangulation(points, triangles);
        }

        void set_triangulation (std::string const& filename) {
            if (tri_ != nullptr) {
                delete tri_;
                tri_ = nullptr;
            }

            tri_ = new Surface_triangulation(filename);
        }

        double mass () {
            if (tri_ == nullptr || density == nullptr) {
                return 0.0;
            }

            return CGAL::to_double(MA::integrate(*tri_, density));
        }

        Eigen::MatrixXd boundary_points () const {
            auto points = tri_->boundary_points();
            Eigen::MatrixXd X(points.size(), 3);

            for (unsigned int i = 0; i < points.size(); ++i) {
                X(i, 0) = CGAL::to_double(points[i].x());
                X(i, 1) = CGAL::to_double(points[i].y());
                X(i, 2) = CGAL::to_double(points[i].z());
            }

            return X;
        }

        // Constant
        void with_density (double c) {
            if (density != nullptr)
                delete density;
            density = new Constant_density(c);
        }

        // From the list of values at the vertices of each triangle
        void with_density (Eigen::MatrixXd const& values) {
            if (density != nullptr)
                delete density;
            density = new Piecewise_linear_density(tri_, values);
        }

        // Linear interpolation between the Lebesgue measure on tri_ and density on tri_
        void with_interpolation () {
            if (interpolation != nullptr)
                delete interpolation;

            if (extended_density != nullptr)
                interpolation = new Interpolated_density(extended_density);
            else
                interpolation = new Interpolated_density(density);
        }

        void without_interpolation () {
            if (interpolation != nullptr)
                delete interpolation;
            interpolation = nullptr;
        }

        void set_interpolation_factor (double t) {
            if (interpolation != nullptr)
                interpolation->t = t;
        }

        void extend_density (std::string const& filename, double eps) {
            extended_tri_ = new Surface_triangulation(filename);
            extended_density = MA::extend_density(*density, tri_, extended_tri_, eps);
        }

        void restore_density () {
            extended_density = nullptr;
            extended_tri_ = nullptr;
        }

        // A, DA, C, T(, NumberComponents, Index)
        template < typename F >
        internal::Results::tuple_type
        compute_python (Eigen::MatrixXd const& p, Eigen::VectorXd const& w) {
            std::vector<Point> points = internal::points_from_matrix(p);
            std::vector<double> weights(w.data(), w.data() + w.rows() * w.cols());

            internal::Results res = compute<F>(points, weights);

            return res.to_tuple();
        }

        // Output the source density in a file: we output the value of the density
        // at the centroids of each triangle
        void output_density (std::string const& filename) {
            Surface_triangulation* T = extended_tri_;
            if (T == nullptr) {
                T = tri_;
            }

            if (interpolation != nullptr) {
                MA::output_density(filename, *T, *interpolation);
            } else {
                Density* dens = extended_density;
                if (dens == nullptr) {
                    dens = density;
                }

                MA::output_density(filename, *T, *dens);
            }
        }

        ~Compute_area_derivatives () {
            delete tri_;
            tri_ = nullptr;

            delete extended_tri_;
            extended_tri_ = nullptr;

            delete density;
            density = nullptr;

            delete extended_density;
            extended_density = nullptr;

            delete interpolation;
            interpolation = nullptr;
        }

    private:
        typedef MA::CDensity<Surface_triangulation, Kernel_ad> Density;
        typedef MA::Constant_density<Surface_triangulation, Kernel_ad> Constant_density;
        typedef MA::Piecewise_linear_density<Surface_triangulation, Kernel_ad> Piecewise_linear_density;
        typedef MA::Interpolated_density<Surface_triangulation, Kernel_ad> Interpolated_density;

        Density *density = nullptr, *extended_density = nullptr;
        Interpolated_density *interpolation = nullptr;

        // A, DA, Centroids, dual
        template < typename F >
        internal::Results
        compute (std::vector<Point> const& points,
                 std::vector<double> const& weights) {
            assert(density != nullptr);

            Surface_triangulation* T = extended_tri_;
            if (T == nullptr) {
                T = tri_;
            }

            if (interpolation != nullptr) {
                return internal::compute<F>(*T, *interpolation, points, weights);
            } else {
                Density* D = extended_density;
                if (D == nullptr) {
                    D = density;
                }

                return internal::compute<F>(*T, *D, points, weights);
            }
        }

        Surface_triangulation *tri_ = nullptr, *extended_tri_ = nullptr;
};

PYBIND11_MODULE(backend_laguerre, m) {
    m.doc() = "Laguerre diagram on triangulated surfaces plugin";

    py::class_<Compute_area_derivatives>(m, "MA")
        .def(py::init<std::string>())
        .def(py::init<Eigen::MatrixXd const&, Eigen::MatrixXi const&>())
        .def("set_triangulation", &Compute_area_derivatives::set_triangulation)
        .def("mass", &Compute_area_derivatives::mass)
        /* .def("boundary_points", &Compute_area_derivatives::boundary_points) */
        .def("with_density", (void (Compute_area_derivatives::*)(double)) &Compute_area_derivatives::with_density, "Choose a constant density")
        .def("with_density", (void (Compute_area_derivatives::*)(Eigen::MatrixXd const&)) &Compute_area_derivatives::with_density, "Construct a density from the list of values on a triangulation")
        /* .def("with_interpolation", &Compute_area_derivatives::with_interpolation) */
        /* .def("without_interpolation", &Compute_area_derivatives::without_interpolation) */
        /* .def("set_interpolation_factor", &Compute_area_derivatives::set_interpolation_factor) */
        /* .def("extend_density", &Compute_area_derivatives::extend_density) */
        /* .def("restore_density", &Compute_area_derivatives::restore_density) */
        .def("compute", &Compute_area_derivatives::compute_python<Identity<Kernel_ad>>)
        .def("output_density", &Compute_area_derivatives::output_density);
}

