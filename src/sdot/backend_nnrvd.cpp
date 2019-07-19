/* #define WITH_PROFILING */

#include <cmath>
#include <iostream>
#ifdef WITH_PROFILING
#include <chrono>
#endif

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include <Eigen/Dense>

typedef Eigen::VectorXd VectorXd;
typedef Eigen::VectorXi VectorXi;
typedef Eigen::Vector3d Vector3d;
typedef Eigen::Vector3i Vector3i;
typedef Eigen::MatrixXd MatrixXd;
typedef Eigen::MatrixXi MatrixXi;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXd_RowMajor;
typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXi_RowMajor;

#include <Eigen/Sparse>

typedef Eigen::Triplet<double> Triplet;
typedef Eigen::SparseVector<double> SparseVector;
typedef Eigen::SparseMatrix<double> SparseMatrix;

#include "AD.hpp"
typedef AD AScalar;

typedef Eigen::Matrix<AScalar, Eigen::Dynamic, 1> AVector;
typedef Eigen::Matrix<AScalar, Eigen::Dynamic, Eigen::Dynamic> AMatrix;
typedef Eigen::Matrix<AScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> AMatrix_RowMajor;

#include <Mesh/assembler.hpp>
#include <Mesh/builder.hpp>
#include <Mesh/wrapper.hpp>
#include <Mesh/measure.hpp>
#include <Mesh/import.hpp>
#include <Mesh/export.hpp>
#include <RVD/rvd.hpp>
#include <RVD/rdt.hpp>
#include <Tools/color.hpp>

using namespace Revoropt;

namespace details {
    template < bool AD = false >
    struct numeric_types {
        using scalar_type = double;
        using vector_type = VectorXd;
        using matrixxd_rm_type = Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    };

    template <>
    struct numeric_types<true> {
        using scalar_type = AScalar;
        using vector_type = AVector;
        using matrixxd_rm_type = Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    };

    template < typename T >
    double to_double (T x);

    template <>
    double to_double<double> (double x) {
        return x;
    }

    template <>
    double to_double<AD> (AD x) {
        return x.value();
    }

    template < typename Matrix, typename Vector >
    Matrix lift (Matrix const& Y, Vector const& w) {
        assert(Y.rows() == w.rows());

        Matrix lifted = Matrix::Zero(Y.rows(), 4);

        Vector M(w.rows());
        M.fill(w.minCoeff());
        Vector ww = w - M;

        lifted.block(0, 0, Y.rows(), 3) = Y;
        lifted.col(3) = ww.cwiseSqrt();

        return lifted;
    }

    // FaceSize = 3 (triangles)
    template < int VertexDim = 3, bool AD = false >
    ROMeshAssembler<3, VertexDim, typename numeric_types<AD>::scalar_type>
    create_mesh (typename numeric_types<AD>::matrixxd_rm_type const& X,
                 MatrixXi_RowMajor const& T) {
        // raw arrays are internally handled in row major order
        const int FaceSize = 3;
        using Mesh = ROMeshAssembler<FaceSize, VertexDim, typename numeric_types<AD>::scalar_type>;

        return Mesh(X.data(), X.rows(),
                    (const unsigned int*) T.data(), T.rows());
    }

    // Orthogonal projection on the plane defined by the triangle (a, b, c)
    template < typename Vector >
    Vector orthogonal_projection (Vector const& a, Vector const& b, Vector const& c,
                                  Vector const& p) {
        Vector n = (b - a).cross(c - a);
        n /= n.norm();
        double lambda = n.dot(p);
        Vector x = p - lambda * n;
        return x;
    }

    AMatrix_RowMajor to_ad (MatrixXd_RowMajor const& m) {
        const int N = m.rows(), M = m.cols();
        AMatrix_RowMajor Am(N, M);

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < M; ++j) {
                Am(i, j) = AScalar(m(i, j), N, i); // one variable per line
            }
        }

        return Am;
    }
} // namespace details

// Density on a 2D triangulation embedded in 3D.
template < bool AD >
class CDensity_2 {
    public:
        static const int FaceSize = 3; // triangulation

        using scalar_type = typename details::numeric_types<AD>::scalar_type;

        using vector3d_type = Eigen::Matrix<scalar_type, 3, 1>;
        using vectorxd_type = Eigen::Matrix<scalar_type, Eigen::Dynamic, 1>;
        using matrixxd_type = Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic>;
        using matrixxd_rm_type = Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

        using Mesh = ROMeshAssembler<FaceSize, 3, scalar_type>;
        using LiftedMesh = ROMeshAssembler<FaceSize, 4, scalar_type>;

    public:
        CDensity_2 (MatrixXd_RowMajor const& X, MatrixXi_RowMajor const& T,
                    VectorXd const& mu) {
            VectorXd w0 = VectorXd::Zero(X.rows());
            MatrixXd_RowMajor LX = details::lift(X, w0);

            initialize(X, mu, LX);
            triangles_ = T;

            mesh_ = details::create_mesh<3, AD>(vertices_, triangles_);
            lifted_mesh_ = details::create_mesh<4, AD>(lifted_vertices_, triangles_);

            compute_linear_interpolation();
        }

        void set_values (VectorXd const& mu) {
            assert(mu.rows() == values_.rows());

            values_ = mu;
            compute_linear_interpolation();
        }

        // Without AD
        template < bool B = AD, typename std::enable_if<!B>::type* = nullptr >
        void initialize (MatrixXd_RowMajor const& X, VectorXd const& mu,
                         MatrixXd_RowMajor const& LX) {
            vertices_ = X;
            values_ = mu;
            lifted_vertices_ = LX;
        }

        // With AD
        template < bool B = AD, typename std::enable_if<B>::type* = nullptr >
        void initialize (MatrixXd_RowMajor const& X, VectorXd const& mu,
                         MatrixXd_RowMajor const& LX) {
            vertices_ = details::to_ad(X);
            values_ = mu;
            lifted_vertices_ = details::to_ad(LX);
        }

        scalar_type eval (unsigned int it, vectorxd_type const& p) const {
            Eigen::Vector4d const& F = coeffs_[it];
            return F(0) * p(0) + F(1) * p(1) + F(2) * p(2) + F(3);
        }

        // Quadrature of order 1 (i.e. exact for an affine density over a triangle)
        scalar_type integrate_triangle (unsigned int it,
                                        vectorxd_type const& a,
                                        vectorxd_type const& b,
                                        vectorxd_type const& c) const {
            scalar_type area = triangle_area<3, scalar_type>(a.data(), b.data(), c.data());
            return area * eval(it, (a + b + c) / 3);
        }

        // Quadrature of order 1 (i.e. exact for an affine density over a triangle)
        scalar_type integrate_segment (unsigned int it,
                                       vectorxd_type const& a,
                                       vectorxd_type const& b) const {
            scalar_type len = (b - a).norm();
            return len * eval(it, 0.5 * (a + b));
        }

        Mesh& mesh () {
            return mesh_;
        }

        LiftedMesh& lifted_mesh () {
            return lifted_mesh_;
        }

        Vector3i triangle (unsigned int it) const {
            return triangles_.row(it);
        }

        vector3d_type vertex (unsigned int iv) const {
            return vertices_.row(iv);
        }

    private:
        matrixxd_rm_type vertices_;
        matrixxd_rm_type lifted_vertices_;
        MatrixXi_RowMajor triangles_;
        VectorXd values_;

        Mesh mesh_;
        LiftedMesh lifted_mesh_;

        std::vector<Eigen::Vector4d> coeffs_;

        void compute_linear_interpolation () {
            coeffs_.resize(triangles_.rows());

            for (int it = 0; it < triangles_.rows(); ++it) {
                int ia = triangles_(it, 0),
                    ib = triangles_(it, 1),
                    ic = triangles_(it, 2);

                double va = values_(ia),
                       vb = values_(ib),
                       vc = values_(ic);

                vector3d_type a = vertices_.row(ia),
                              b = vertices_.row(ib),
                              c = vertices_.row(ic),
                              v = a + (b - a).cross(c - a);

                Eigen::Matrix4d A;
                A << details::to_double(a.x()), details::to_double(a.y()), details::to_double(a.z()), 1,
                     details::to_double(b.x()), details::to_double(b.y()), details::to_double(b.z()), 1,
                     details::to_double(c.x()), details::to_double(c.y()), details::to_double(c.z()), 1,
                     details::to_double(v.x()), details::to_double(v.y()), details::to_double(v.z()), 1;

                Eigen::Vector4d V;
                V << va, vb, vc, 0;

                // A * F = V
                Eigen::Vector4d F = A.partialPivLu().solve(V);

                coeffs_[it] = F;
            }
        }
};

using Density_2 = CDensity_2<false>; // without automatic differentiation (quadratic cost)
using AD_Density_2 = CDensity_2<true>; // with automatic differentiation (reflector, refractor cost)

namespace details {
    // Quadrature of order 2 i.e. exact for a degree 2 polynomial over a triangle
    template < bool AD = false, typename Function >
    typename numeric_types<AD>::scalar_type
    integrate_triangle_2 (Function f,
                          typename numeric_types<AD>::vector_type const& a,
                          typename numeric_types<AD>::vector_type const& b,
                          typename numeric_types<AD>::vector_type const& c) {
        typedef typename numeric_types<AD>::scalar_type Scalar;

        Scalar res(0.0);
        Scalar _1_6 = 1.0f / 6.0f, _2_3 = 2.0f / 3.0f;

        res += f(_1_6 * a + _2_3 * b + _1_6 * c);
        res += f(_2_3 * a + _1_6 * b + _1_6 * c);
        res += f(_1_6 * a + _2_3 * b + _1_6 * c);

        Scalar A = triangle_area<3, Scalar>(a.data(), b.data(), c.data());
        res *= _1_6 * 2 * A;

        return res;
    }

    // Quadrature of order 3 i.e. exact for a degree 3 polynomial over a triangle
    template < bool AD, typename Function >
    typename numeric_types<AD>::scalar_type
    integrate_triangle_3 (Function f,
                          typename numeric_types<AD>::vector_type const& a,
                          typename numeric_types<AD>::vector_type const& b,
                          typename numeric_types<AD>::vector_type const& c) {
        typedef typename numeric_types<AD>::scalar_type Scalar;
        typedef typename numeric_types<AD>::vector_type Vector;

        Scalar _1_2 = 1.0f / 2.0f, _1_6 = 1.0f / 6.0f, _2_3 = 2.0f / 3.0f,
               _1_30 = 1.0f / 30.0f, _9_30 = 9.0f / 30.0f;

        Vector u = b - a, v = c - a;
        Scalar r1 = _1_30 * f(a + _1_2 * u + _1_2 * v);
        Scalar r2 = _1_30 * f(a + _1_2 * u);
        Scalar r3 = _1_30 * f(a + _1_2 * v);
        Scalar r4 = _9_30 * f(a + _1_6 * u + _2_3 * v);
        Scalar r5 = _9_30 * f(a + _1_6 * v + _2_3 * u);
        Scalar r6 = _9_30 * f(a + _1_6 * u + _1_6 * v);
        Scalar A = triangle_area<3, Scalar>(a.data(), b.data(), c.data());

        return A * (r1 + r2 + r3 + r4 + r5 + r6);
    }

    template <typename Triangulation, bool AD = false>
    class BarycenterAction : public Action<Triangulation> {
        /* Typedefs */
        typedef typename Triangulation::Scalar Scalar ;
        enum { Dim = Triangulation::VertexDim } ;
        typedef MatrixXd_RowMajor Matrix;

      public:
        typedef Eigen::Matrix<Scalar, Dim, 1> Vector;
        typedef Eigen::Matrix<Scalar, 3, 1> Vector3d;

        BarycenterAction(Scalar* target, // nu
                         std::map<std::pair<int, int>, Vector3d>* target_boundaries,
                         unsigned int size, // N_Y
                         CDensity_2<AD>* density, // mu
                         const Matrix *sites) : // Y
            barycenters_(target), barycenters_boundaries_(target_boundaries), density_(density), sites_(sites) {
          std::fill(target, target+Dim*size, 0) ;
          areas_.assign(size, 0) ;
          w2 = 0.0;
        }

        /* Action for to use on the RVD */
        void operator() (unsigned int site,
                         unsigned int triangle,
                         const RVDPolygon<Triangulation>& polygon) {
          // size of the polygon
          unsigned int size = polygon.size();
          if (size < 3) return;

          // TODO: assert raised (invalid sizes when resizing array)
          /* Vector y_i = sites_->row(site).cast<Scalar>(); */

          Vector y_i;
          auto y_i_tmp = sites_->row(site);
          for (int i = 0; i < 3; ++i) {
              y_i(i) = y_i_tmp(i);
          }

          // triangulate the polygon
          const Vector& base_vertex = polygon[0].vertex;
          for (unsigned int i = 1; i < size - 1; ++i) {
            const Vector& v1 = polygon[ i        ].vertex;
            const Vector& v2 = polygon[(i+1)%size].vertex;

            Scalar area = density_->integrate_triangle(triangle,
                                                       base_vertex, v1, v2);

            // Compute the W^2 distance
            w2 += integrate_triangle_3<AD>([&] (Vector const& x) {
                                              return (x - y_i).squaredNorm() * density_->eval(triangle, x);
                                          }, base_vertex, v1, v2);

            // Compute barycenter and area (normalized in finish)
            Eigen::Map<Vector> barycenter(barycenters_ + Dim*site);
            barycenter += area * (base_vertex + v1 + v2) / 3;
            areas_[site] += area;
          }

          // Derivatives
          for (unsigned int v = 0; v < size; ++v) {
              const RVDVertex<Triangulation>& p1 = polygon[ v        ].vertex;
              const RVDVertex<Triangulation>& p2 = polygon[(v+1)%size].vertex;

              const Vector& v1 = polygon[ v        ].vertex ;
              const Vector& v2 = polygon[(v+1)%size].vertex ;

              if (p1.config() == RVDVertex<Triangulation>::EDGE_VERTEX ||
                  p1.config() == RVDVertex<Triangulation>::FACE_VERTEX) {
                  cells_[site].push_back(p1);
              }

              if (p2.config() == RVDVertex<Triangulation>::EDGE_VERTEX ||
                  p2.config() == RVDVertex<Triangulation>::FACE_VERTEX) {
                  cells_[site].push_back(p2);
              }

              // Lag_{ij} where i = site
              const RVDEdge<Triangulation>& e = polygon[v];
              if (e.config == RVDEdge<Triangulation>::BISECTOR_EDGE) {
                  unsigned int j = e.combinatorics;
                  boundaries_[std::make_pair(site, j)].emplace_back(triangle, v1, v2);

                  // Barycenter of Lag_{ij} (non-normalized)
                  Scalar len = density_->integrate_segment(triangle, v1, v2);
                  Vector3d bary = len * (v1.topRows(3) + v2.topRows(3)) / 2;
                  auto key = std::make_pair(site, j);

                  if (barycenters_boundaries_->count(key) == 0) {
                      (*barycenters_boundaries_)[key] = Vector3d::Zero();
                  }
                  (*barycenters_boundaries_)[key] += bary;
              }

              // Triangulation
              if (e.config == RVDEdge<Triangulation>::BISECTOR_EDGE &&
                  e.vertex.config() == RVDVertex<Triangulation>::FACE_VERTEX) {
                  triangulation_.emplace_back(e.vertex.combinatorics()[3],
                                              e.vertex.combinatorics()[4],
                                              e.vertex.combinatorics()[5]);
              }
          }
        }

        void finish() {
          for (unsigned int i = 0; i < areas_.size(); ++i) {
            Eigen::Map<Vector> barycenter(barycenters_ + Dim*i) ;
            barycenter /= areas_[i] ;
          }
        }

        Scalar squared_distance () const {
            return w2;
        }

        std::vector<Scalar> const& areas () const {
            return areas_;
        }

        std::vector<std::tuple<int, int, int>> const& triangulation () const {
            return triangulation_;
        }

        std::map<std::pair<int, int>,
                 std::vector<std::tuple<unsigned int, Vector, Vector>>> const& boundaries () const {
             return boundaries_;
         }

        std::map<int, std::vector<Vector>> const& cells () const {
            return cells_;
        }

      private:
        Scalar w2;
        Scalar* barycenters_ ;
        std::map<std::pair<int, int>, Vector3d>* barycenters_boundaries_ ;
        std::vector<Scalar> areas_ ;
        CDensity_2<AD>* density_;
        const Matrix* sites_;

        std::map<std::pair<int, int>,
                 std::vector<std::tuple<unsigned int, Vector, Vector>>> boundaries_;
        std::vector<std::tuple<int, int, int>> triangulation_;
        std::map<int, std::vector<Vector>> cells_;
    };
} // namespace details

struct Results {
    double w2; // W^2 squared distance
    VectorXd A; // areas of the Laguerre cells
    SparseMatrix DA; // gradient of the areas of the Laguerre cells
    MatrixXd C; // centroids of the Laguerre cells
    MatrixXi T; // dual of the Laguerre diagram
    std::vector<MatrixXd> TT; // List (with duplicates) of the Laguerre vertices
    std::map<std::pair<int, int>, Vector3d> C_boundary; // centroids of the boundaries of the Laguerre cells

    typedef std::tuple<double,
                       VectorXd,
                       SparseMatrix,
                       MatrixXd,
                       MatrixXi,
                       std::vector<MatrixXd>,
                       std::map<std::pair<int, int>, Vector3d>> tuple_type;

    tuple_type to_tuple () const {
        return std::make_tuple(w2, A, DA, C, T, TT, C_boundary);
    }
};

// Export Laguerre cells as a OBJ/MTL files.
// FIXME: exported mesh is in 4D (but MeshLab can still open it
// TODO: AD version
void export_laguerre_cells (Density_2 mu,
                            MatrixXd_RowMajor const& sites, VectorXd const& w,
                            std::string const& basename) {
    const int FaceSize = 3;
    const int VertexDim = 4;
    typedef ROMesh<FaceSize, VertexDim> RVDMesh;

    const int nsites = w.rows();

    // Lift 3D Power diagram into a 4D Voronoi diagram
    MatrixXd_RowMajor lifted_sites = details::lift(sites, w);

    // Compute intersection diagram
    RVD<RVDMesh> rvd;
    rvd.set_sites(lifted_sites.data(), nsites);
    rvd.set_mesh(&mu.lifted_mesh());

    // action to pass to the RVD computer. We will store it to export it.
    RVDStore<RVDMesh> store ;
    rvd.compute(store);

    // export the RVD with a material file to distinguish the cells
    std::vector<double> colormap(3*nsites) ;
    generate_bright_colormap(nsites, colormap.data()) ;

    export_colored_obj(&store.rvd_mesh,
                       store.face_sites.data(),
                       colormap.data(),
                       basename
                      );
}

// Without AD = quadratic cost
Results::tuple_type monge_ampere (Density_2 mu,
                                  MatrixXd_RowMajor const& sites, VectorXd const& w) {
    // --------------------------------
    // Computation of Laguerre diagram
    // --------------------------------
#ifdef WITH_PROFILING
    auto start_diagram = std::chrono::high_resolution_clock::now();
#endif

    const int FaceSize = 3;
    const int VertexDim = 4;
    typedef ROMesh<FaceSize, VertexDim> RVDMesh;

    typedef Density_2::vector3d_type Vector3d;

    const int nsites = w.rows();

    // Lift 3D Power diagram into a 4D Voronoi diagram
    MatrixXd_RowMajor lifted_sites = details::lift(sites, w);

    // Compute intersection diagram
    RVD<RVDMesh> rvd;

    rvd.set_sites(lifted_sites.data(), nsites);
    rvd.set_mesh(&mu.lifted_mesh());

    std::vector<double> barycenters(VertexDim * nsites) ;
    std::map<std::pair<int, int>, Vector3d> barycenters_boundaries;
    details::BarycenterAction<RVDMesh> action(barycenters.data(), &barycenters_boundaries,
                                              nsites, &mu, &sites);

    rvd.compute(action);
    action.finish();

#ifdef WITH_PROFILING
    auto end_diagram = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> diff_diagram_seconds = end_diagram - start_diagram;
    std::cerr << "Diagram computation: " << diff_diagram_seconds.count() << std::endl;
#endif

    // --------------------------------
    // Integration
    // --------------------------------
#ifdef WITH_PROFILING
    auto start_integration = std::chrono::high_resolution_clock::now();
#endif

    // Areas
    std::vector<double> const& areas = action.areas();
    VectorXd A = Eigen::Map<const VectorXd>(areas.data(), sites.rows());

    // Derivatives
    std::vector<Triplet> triplets;
    double diagonal[nsites];
    std::fill(diagonal, diagonal + nsites, 0.0);
    for (const auto& kv: action.boundaries()) {
        int i = -1, j = -1;
        std::tie(i, j) = kv.first;

        VectorXd yi = sites.row(i), yj = sites.row(j);
        Vector3d diff = (yj - yi).topRows(3);

        double v = 0.0;

        for (const auto& s: kv.second) {
            int it = -1;
            VectorXd src, dst;
            std::tie(it, src, dst) = s;

            Vector3i t = mu.triangle(it);
            Vector3d a = mu.vertex(t(0)),
                     b = mu.vertex(t(1)),
                     c = mu.vertex(t(2));

            double tmp = mu.integrate_segment(it, src.topRows(3), dst.topRows(3));
            /* tmp /= 2 * diff.norm(); */
            tmp /= 2 * details::orthogonal_projection(a, b, c, diff).norm();
            v += tmp;
        }

        triplets.emplace_back(i, j, v);
        diagonal[i] += v;
    }

    for (int i = 0; i < nsites; ++i) {
        triplets.emplace_back(i, i, -diagonal[i]);
    }

    SparseMatrix DA(nsites, nsites);
    DA.setFromTriplets(triplets.begin(), triplets.end());
    DA.makeCompressed();

    // Centroids
    MatrixXd C(nsites, 3);
    for (int i = 0; i < nsites; ++i) {
        C(i, 0) = barycenters[i * VertexDim + 0];
        C(i, 1) = barycenters[i * VertexDim + 1];
        C(i, 2) = barycenters[i * VertexDim + 2];
    }
    /* MatrixXd C = Eigen::Map<MatrixXd>(barycenters.data(), nsites, 3); */

    // Dual triangulation
    std::vector<std::tuple<int, int, int>> const& tris = action.triangulation();
    MatrixXi T(tris.size(), 3);
    for (int i = 0; i < T.rows(); ++i) {
        T(i, 0) = std::get<0>(tris[i]);
        T(i, 1) = std::get<1>(tris[i]);
        T(i, 2) = std::get<2>(tris[i]);
    }

    // Vertices of the Laguerre cells
    auto cells = action.cells();
    std::vector<MatrixXd> TT;
    for (const auto& kv: cells) {
        int site = kv.first;
        MatrixXd TTi(kv.second.size(), 3);

        for (int j = 0; j < kv.second.size(); ++j) {
            const details::BarycenterAction<RVDMesh>::Vector v = kv.second[j];
            TTi(j, 0) = v(0);
            TTi(j, 1) = v(1);
            TTi(j, 2) = v(2);
        }

        TT.push_back(TTi);
    }

    Results res = {action.squared_distance(), A, DA, C, T, TT, barycenters_boundaries};

#ifdef WITH_PROFILING
    auto end_integration = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> diff_integration_seconds = end_integration - start_integration;
    std::cerr << "Integration: " << diff_integration_seconds.count() << std::endl;
#endif

    return res.to_tuple();
}

// With AD = "any" cost
Results::tuple_type monge_ampere_ad (AD_Density_2 mu,
                                     MatrixXd_RowMajor const& sites, VectorXd const& w) {
    typedef typename CDensity_2<true>::scalar_type scalar_type;

    const int FaceSize = 3;
    const int VertexDim = 4;
    typedef ROMesh<FaceSize, VertexDim, scalar_type> RVDMesh;

    typedef AD_Density_2::vector3d_type Vector3d;

    const int nsites = w.rows();

    // Lift 3D Power diagram into a 4D Voronoi diagram
    MatrixXd_RowMajor lifted_sites_tmp = details::lift(sites, w);
    AMatrix_RowMajor lifted_sites = details::to_ad(lifted_sites_tmp);

    // Compute intersection diagram
    RVD<RVDMesh> rvd;

    rvd.set_sites(lifted_sites.data(), nsites);
    rvd.set_mesh(&mu.lifted_mesh());

    std::vector<scalar_type> barycenters(VertexDim * nsites) ;
    std::map<std::pair<int, int>, Vector3d> barycenters_boundaries;
    details::BarycenterAction<RVDMesh, true> action(barycenters.data(), &barycenters_boundaries,
                                                    nsites, &mu, &sites);

    rvd.compute(action);
    action.finish();

    // Areas
    VectorXd A = VectorXd::Zero(sites.rows());
    std::vector<scalar_type> const& areas = action.areas();
    for (int i = 0; i < A.rows(); ++i) {
        A(i) = areas[i].value();
    }

    // Derivatives
    std::vector<Triplet> triplets;
    double diagonal[nsites];
    std::fill(diagonal, diagonal + nsites, 0.0);
    for (int i = 0; i < nsites; ++i) {
        scalar_type curarea = areas[i];
        for (SparseVector::InnerIterator vit(curarea.derivatives()); vit; ++vit) {
            triplets.emplace_back(i, vit.row(), vit.value());
            diagonal[i] += vit.value();
        }
    }

    for (int i = 0; i < nsites; ++i) {
        triplets.emplace_back(i, i, -diagonal[i]);
    }

    SparseMatrix DA(nsites, nsites);
    DA.setFromTriplets(triplets.begin(), triplets.end());
    DA.makeCompressed();

    // Centroids
    MatrixXd C(nsites, 3);
    for (int i = 0; i < nsites; ++i) {
        C(i, 0) = barycenters[i * VertexDim + 0].value();
        C(i, 1) = barycenters[i * VertexDim + 1].value();
        C(i, 2) = barycenters[i * VertexDim + 2].value();
    }

    // Dual triangulation
    std::vector<std::tuple<int, int, int>> const& tris = action.triangulation();
    MatrixXi T(tris.size(), 3);
    for (int i = 0; i < T.rows(); ++i) {
        T(i, 0) = std::get<0>(tris[i]);
        T(i, 1) = std::get<1>(tris[i]);
        T(i, 2) = std::get<2>(tris[i]);
    }

    Results res = {details::to_double(action.squared_distance()), A, DA, C, T};

    return res.to_tuple();
}

PYBIND11_MODULE(backend_nnrvd, m) {
    m.doc() = "Restriced Power diagrams on surfaces with NNRVD";

    py::class_<Density_2>(m, "Density_2")
        .def(py::init<MatrixXd_RowMajor const&, MatrixXi_RowMajor const&, VectorXd const&>())
        .def("set_values", &Density_2::set_values)
        .def("eval", &Density_2::eval);

    py::class_<AD_Density_2>(m, "AD_Density_2")
        .def(py::init<MatrixXd_RowMajor const&, MatrixXi_RowMajor const&, VectorXd const&>());

    m.def("monge_ampere", &monge_ampere, "Compute areas of Laguerre cells and its derivatives");
    m.def("monge_ampere_ad", &monge_ampere_ad, "Compute areas of Laguerre cells and its derivatives");

    m.def("export_laguerre_cells", &export_laguerre_cells);
}

