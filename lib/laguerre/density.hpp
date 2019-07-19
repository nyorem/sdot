#ifndef _DENSITY_HPP_
#define _DENSITY_HPP_

#include <cmath>
#include <functional>
#include <Eigen/Dense> // used in Piecewise_linear_density
#include "gaussian_quadrature.hpp"
#include "utils.hpp"

namespace MA {
    template < typename K >
    using CScalar_function = std::function<typename K::FT(typename CGAL::Point_3<K> const&)>;

    // Density defined on a triangulation (piecewise on each triangle)
    template < typename T, typename Kernel_ >
    struct CDensity {
        typedef typename T::Face_handle Face_handle_T;

        typedef Kernel_ Kernel;
        typedef typename Kernel::FT FT;
        typedef typename Kernel::Point_3 Point;

        typedef CScalar_function<Kernel> Scalar_function;

        virtual Scalar_function operator[] (Face_handle_T const& t) const = 0;

        virtual ~CDensity () {
        }
    };

    // Constant density
    template < typename T, typename Kernel >
    struct Constant_density : public CDensity<T, Kernel> {
        typedef CDensity<T, Kernel> Base;
        typedef typename Base::FT FT;
        typedef typename Base::Scalar_function Scalar_function;
        typedef typename Base::Point Point;
        typedef typename Base::Face_handle_T Face_handle_T;
        typedef typename T::Point Point_T;

        FT c;

        Constant_density (FT c = 1.0f) : c(c) {
        }

        Scalar_function operator[] (Face_handle_T const& t) const override {
            return [&] (Point const& p) {
                return c;
            };
        }
    };

    // Piecewise linear density
    template < typename T, typename Kernel >
    struct Piecewise_linear_density : public CDensity<T, Kernel> {
        typedef CDensity<T, Kernel> Base;
        typedef typename Base::FT FT;
        typedef typename Base::Scalar_function Scalar_function;
        typedef typename Base::Point Point;
        typedef typename Base::Face_handle_T Face_handle_T;
        typedef typename T::Point Point_T;

        // coeffs[t] = (a, b, c, d) s.t. F(p) = a*x + b*y + c*z + d
        std::map<Face_handle_T, Eigen::Vector4d> coeffs;
        std::map<std::tuple<Point_T, Point_T, Point_T>, Face_handle_T> face_handle_map;

        Piecewise_linear_density () {
        }

        // Construction from the list of values at the vertices of each triangle
        Piecewise_linear_density (T* tri, Eigen::MatrixXd const& values) {
            for (auto tit = tri->triangles_begin(); tit != tri->triangles_end(); ++tit) {
                Face_handle_T t = &(*tit);
                int tid = t->tid;
                double va = values(tid, 0), vb = values(tid, 1), vc = values(tid, 2);
                set_values(t, va, vb, vc);
                face_handle_map[std::make_tuple(t->vertex(0), t->vertex(1), t->vertex(2))] = t;
            }
        }

        void set_values (Face_handle_T const& t, double va, double vb, double vc) {
            typename T::Point a = t->vertex(0),
                              b = t->vertex(1),
                              c = t->vertex(2),
                              v = a + CGAL::cross_product(b - a, c - a);

            Eigen::Matrix4d A;
            A << a.x(), a.y(), a.z(), 1,
                 b.x(), b.y(), b.z(), 1,
                 c.x(), c.y(), c.z(), 1,
                 v.x(), v.y(), v.z(), 1;

            Eigen::Vector4d V;
            V << va, vb, vc, 0;

            // A * F = V
            Eigen::Vector4d F = A.partialPivLu().solve(V);

            coeffs[t] = F;
        }

        Scalar_function operator[] (Face_handle_T const& t) const override {
            return [&] (Point const& p) {
                Eigen::Vector4d const& F = coeffs.at(t);
                return F(0) * p.x() + F(1) * p.y() + F(2) * p.z() + F(3);
            };
        }
    };

    // Interpolation between the Lebesgue density (Constant_density with C = 1)
    // and an other density.
    // t = 1: Lebesgue / t = 0: other measure
    // TODO: not used
    template < typename T, typename Kernel >
    struct Interpolated_density : CDensity<T, Kernel> {
        typedef CDensity<T, Kernel> Base;
        typedef typename Base::FT FT;
        typedef typename Base::Scalar_function Scalar_function;
        typedef typename Base::Point Point;
        typedef typename Base::Face_handle_T Face_handle_T;

        FT t = 1;
        Base* mu;

        Interpolated_density (Base* mu) : mu(mu) {
        }

        Scalar_function operator[] (Face_handle_T const& triangle) const override {
            return [&] (Point const& p) {
                return t * 1.0f + (1 - t) * (*mu)[triangle](p);
            };
        }
    };

    // TODO: not used
    template < typename T, typename Kernel >
    Piecewise_linear_density<T, Kernel>*
    extend_density (CDensity<T, Kernel> const& density,
                    T* base_domain,
                    T* extended_domain,
                    double eps = 1e-6) {
        typename CGAL::Cartesian_converter<typename T::Kernel, Kernel> Converter;

        Piecewise_linear_density<T, Kernel>* res = new Piecewise_linear_density<T, Kernel>();
        for (auto tit = extended_domain->triangles_begin();
             tit != extended_domain->triangles_end();
             ++tit) {
            typename T::Face_handle t = &(*tit);

            typename T::Point a = t->vertex(0),
                              b = t->vertex(1),
                              c = t->vertex(2);

            double va = base_domain->is_inside(a) ? CGAL::to_double(density[t](Converter(a))) : eps,
                   vb = base_domain->is_inside(b) ? CGAL::to_double(density[t](Converter(b))) : eps,
                   vc = base_domain->is_inside(c) ? CGAL::to_double(density[t](Converter(c))) : eps;

            res->set_values(t, va, vb, vc);
        }

        return res;
    }

    // Integrate an affine function on a 3D triangle.
    // PRE-CONDITION: f is affine
    template < typename K >
    typename K::FT
    integrate (CScalar_function<K> const& f,
               typename CGAL::Point_3<K> const& a,
               typename CGAL::Point_3<K> const& b,
               typename CGAL::Point_3<K> const& c) {
        return ::integrate_1(f, a, b, c);
    }

    // Integrate an affine function over a 3D polygon given by its ordered vertices.
    // PRE-CONDITION: f is affine
    template < typename K >
    typename K::FT
    integrate (CScalar_function<K> const& f,
               std::vector<typename K::Point_3> const& vertices) {
        typedef typename K::FT FT;

        FT res = 0;

        // Subdivide in triangles
        for (std::size_t i = 1; i < vertices.size() - 2; ++i) {
            res += MA::integrate(f, vertices[0], vertices[i], vertices[i + 1]);
        }

        return res;
    }

    // Integrate an affine on a triangulated surface
    // PRE-CONDITION: f is affine
    template < typename Kernel, typename T >
    typename Kernel::FT
    integrate (T& tri, MA::CDensity<T, Kernel>* f) {
        typename Kernel::FT res = 0;
        typename CGAL::Cartesian_converter<typename T::Kernel, Kernel> Converter;

        for (auto tit = tri.triangles_begin(); tit != tri.triangles_end(); ++tit) {
            typename T::Face_handle t = &(*tit);
            res += MA::integrate((*f)[t], Converter(t->vertex(0)),
                                          Converter(t->vertex(1)),
                                          Converter(t->vertex(2)));
        }

        return res;
    }

    // Output a density on a file (only used for debug purposes)
    // Evaluate the density at the centroids of the triangles: x y z v
    template < typename T, typename Density >
    void output_density (std::string const& filename, T& tri, Density const& density) {
        typename CGAL::Cartesian_converter<typename T::Kernel, typename Density::Kernel> Converter;
        std::ofstream out(filename.c_str());
        for (auto tit = tri.triangles_begin(); tit != tri.triangles_end(); ++tit) {
            typename T::Face_handle t = &(*tit);
            auto c = Converter(CGAL::centroid(*t));
            out << c.x() << " " << c.y() << " " << c.z() << " " << density[t](c) << std::endl;
        }
    }
} // namespace MA

#endif

