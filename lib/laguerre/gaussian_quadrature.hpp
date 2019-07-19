#ifndef _GAUSSIAN_QUADRATURE_H_
#define _GAUSSIAN_QUADRATURE_H_

#include <cmath>

// 2D

// Evaluate the integral of a scalar function F using a
// Gaussian quadrature with N points.
// The integration is exact for polynomials of degree 2N - 1 or less.
// F: R -> R
template < typename F >
double gaussian_quadrature (F const& f,
                            double const points[],
                            double const weights[],
                            double const N,
                            double const a = -1, double const b = 1) {
    double integral = 0;

    for (int  i = 0; i < N; ++i) {
        double x = points[i],
               xx = a + 0.5 * (b - a) * (1 + x);
        integral += weights[i] * f(xx);
    }

    integral *= 0.5 * (b - a);

    return integral;
}

// Parametrization of a segment
// phi : [0, 1] -> s
/* template < typename Point, typename Segment > */
/* std::function<double(Point)> */
/* phi (Segment const &s) { */
/*     return [=] (double t) { return (1 - t) * (s.source() - CGAL::ORIGIN) + */
/*         t * (s.target() - CGAL::ORIGIN); */
/*     }; */
/* } */

// Gaussian quadrature with 2 points.
// The integral is exact for polynomials of degree 3 or less.
// F: R -> R
template < typename F >
double gaussian_quadrature_order_2 (F const& f,
                                    double const a = -1, double const b = 1) {
    double points[2] = { -1 / std::sqrt(3), 1 / std::sqrt(3) },
           weights[2] = { 1, 1 };

    return gaussian_quadrature(f, points, weights, 2, a, b);
}

// Gaussian quadrature with 3 points.
// The integral is exact for polynomials of degree 5 or less.
// F: R -> R
template < typename F >
double gaussian_quadrature_order_3 (F const& f,
                                    double const a = -1,
                                    double const b = 1) {
    double points[3] = { -std::sqrt(3) / std::sqrt(5), 0, std::sqrt(3) / std::sqrt(5) },
           weights[3] = { double(5) / 9, double(8) / 9, double(5) / 9 };

    return gaussian_quadrature(f, points, weights, 3, a, b);
}

// Gaussian quadrature on a segment with 2 points for a vector field.
// Field: Point -> Point
template < typename Field >
double gaussian_quadrature_field_order_2 (Field const& X,
                                          typename Field::Segment const& s,
                                          typename Field::Vector const& n) {
    // phi : [0, 1] -> s
    auto phi = [&] (double t) { return (1 - t) * (s.source() - CGAL::ORIGIN) +
                                        t * (s.target() - CGAL::ORIGIN);
                              };

    double integral = gaussian_quadrature_order_2([&](double t) {
                                    typename Field::Point p = CGAL::ORIGIN + phi(t);
                                    return (X(p) - CGAL::ORIGIN) * n;
                                }, 0, 1);

    integral *= std::sqrt(s.squared_length());

    return integral;
}

// Gaussian quadrature on a segment with 3 points for a vector field.
// Field: Point -> Point
template < typename Field >
double gaussian_quadrature_field_order_3 (Field const& X,
                                          typename Field::Segment const& s,
                                          typename Field::Vector const& n) {
    // phi : [0, 1] -> s
    auto phi = [&] (double t) { return (1 - t) * (s.source() - CGAL::ORIGIN) +
                                        t * (s.target() - CGAL::ORIGIN);
                              };

    double integral = gaussian_quadrature_order_3([&](double t) {
                                    typename Field::Point p = CGAL::ORIGIN + phi(t);
                                    return (X(p) - CGAL::ORIGIN) * n;
                                }, 0, 1);

    integral *= std::sqrt(s.squared_length());

    return integral;
}

// Gaussian quadrature on a segment with 2 points for the divergence of a vector field.
// Field: Point -> Point
template < typename Field >
double gaussian_quadrature_div_order_2 (Field const& X,
                                        typename Field::Segment const& s) {
    // phi : [0, 1] -> s
    auto phi = [&] (double t) { return (1 - t) * (s.source() - CGAL::ORIGIN) +
                                        t * (s.target() - CGAL::ORIGIN);
                              };

    double integral = gaussian_quadrature_order_2([&] (double t) {
                                                return X.div(CGAL::ORIGIN + phi(t));
                                            }, 0, 1);

    integral *= std::sqrt(s.squared_length());

    return integral;
}

// Gaussian quadrature on a segment with 3 points for the divergence of a vector field.
// Field: Point -> Point
template < typename Field >
double gaussian_quadrature_div_order_3 (Field const& X,
                                        typename Field::Segment const& s) {
    // phi : [0, 1] -> s
    auto phi = [&] (double t) { return (1 - t) * (s.source() - CGAL::ORIGIN) +
                                        t * (s.target() - CGAL::ORIGIN);
                              };

    double integral = gaussian_quadrature_order_3([&] (double t) {
                                                return X.div(CGAL::ORIGIN + phi(t));
                                            }, 0, 1);

    integral *= std::sqrt(s.squared_length());

    return integral;
}

// Gaussian quadrature on a segment with 2 points for a scalar function.
// F: Point -> R
template < typename F >
double gaussian_quadrature_scalar_order_2 (F const& f,
                                           typename F::Segment const& s) {
    // phi : [0, 1] -> s
    auto phi = [&] (double t) { return (1 - t) * (s.source() - CGAL::ORIGIN) +
                                        t * (s.target() - CGAL::ORIGIN);
                              };

    double integral = gaussian_quadrature_order_2([&] (double t) {
                                                return f(CGAL::ORIGIN + phi(t));
                                            }, 0, 1);

    integral *= std::sqrt(s.squared_length());

    return integral;
}

// Gaussian quadrature on a segment with 3 points for a scalar function.
// F: Point -> R
template < typename F >
double gaussian_quadrature_scalar_order_3 (F const& f,
                                           typename F::Segment const& s) {
    // phi : [0, 1] -> s
    auto phi = [&] (double t) { return (1 - t) * (s.source() - CGAL::ORIGIN) +
                                        t * (s.target() - CGAL::ORIGIN);
                              };

    double integral = gaussian_quadrature_order_3([&] (double t) {
                                                return f(CGAL::ORIGIN + phi(t));
                                            }, 0, 1);

    integral *= std::sqrt(s.squared_length());

    return integral;
}

// Gaussian quadrature of degree 1, 2 or 3 for a 2D density over a 2D triangle
template < typename K >
typename K::FT
integrate (std::function<typename K::FT(typename K::Point_2 const&)> const& f,
           typename CGAL::Triangle_2<K> const& t,
           unsigned int order = 2) {
    assert(order >= 1 && order <= 3);

    typedef typename K::FT FT;
    typedef typename K::Point_2 Point;

    // Nodal functions
    auto N1 = [] (FT ksi, FT eta) {
        return 1 - ksi - eta;
    };

    auto N2 = [] (FT ksi, FT eta) {
        return ksi;
    };

    auto N3 = [] (FT ksi, FT eta) {
        return eta;
    };

    // Change of variable
    auto P = [&] (FT ksi, FT eta) {
        return t.vertex(0).x() * N1(ksi, eta) +
            t.vertex(1).x() * N2(ksi, eta) +
            t.vertex(2).x() * N3(ksi, eta);
    };

    auto Q = [&] (FT ksi, FT eta) {
        return t.vertex(0).y() * N1(ksi, eta) +
            t.vertex(1).y() * N2(ksi, eta) +
            t.vertex(2).y() * N3(ksi, eta);
    };

    auto F = [&] (Point const& p) {
        return f(Point(P(p.x(), p.y()), Q(p.x(), p.y())));
    };

    // Integration
    std::vector<FT> weights;
    std::vector<Point> points;
    if (order == 1) {
        weights = { FT (1.0f / 2.0f) };
        points = { Point(1.0f / 3.0f, 1.0f / 3.0f) };
    } else if (order == 2) {
        weights = { FT(1.0f / 6.0f), FT(1.0f / 6.0f), FT(1.0f / 6.0f) };
        points = { Point(1.0f / 6.0f, 1.0f / 6.0f)
                 , Point(2.0f / 3.0f, 1.0f / 6.0f)
                 , Point(1.0f / 6.0f, 2.0f / 3.0f)
                 };
    } else if (order == 3) {
        weights = { FT(-27.0f / 96.0f), FT(25.0f / 96.0f), FT(25.0f / 96.0f), FT(25.0f / 96.0f) };
        points = { Point(1.0f / 3.0f, 1.0f / 3.0f)
                 , Point(1.0f / 5.0f, 1.0f / 5.0f)
                 , Point(1.0f / 5.0f, 3.0f / 5.0f)
                 , Point(3.0f / 5.0f, 1.0f / 5.0f)
                 };
    }

    FT res = 0;
    for (std::size_t i = 0; i < weights.size(); ++i) {
        res += weights[i] * F(points[i]);
    }

    return 2 * t.area() * res;
}

// Gaussian quadrature of degree 1, 2 or 3 for a 2D density over a 2D triangle
template < typename K >
typename K::FT
integrate (std::function<typename K::FT(typename K::Point_2 const&)> const& f,
           typename CGAL::Point_2<K> const& a,
           typename CGAL::Point_2<K> const& b,
           typename CGAL::Point_2<K> const& c,
           unsigned int order = 2) {
    typedef typename CGAL::Triangle_2<K> Triangle;

    return integrate(f, Triangle(a, b, c), order);
}

// 3D

// Gaussian quadrature of order 1, 2 or 3 for a 3D density defined on a 3D triangle.
template < typename K >
typename K::FT
integrate (std::function<typename K::FT(typename K::Point_3 const&)> const& f,
           typename CGAL::Triangle_3<K> const& t,
           unsigned int order = 2) {
    assert(order >= 1 && order <= 3);

    typedef typename K::FT FT;
    typedef typename K::Point_2 Point_2;
    typedef typename K::Vector_3 Vector;
    typedef typename K::Point_3 Point;

    Vector v0 = t.vertex(0) - CGAL::ORIGIN,
           v1 = t.vertex(1) - CGAL::ORIGIN,
           v2 = t.vertex(2) - CGAL::ORIGIN;

    // Integration
    std::vector<FT> weights;
    std::vector<Point_2> points;
    if (order == 1) {
        weights = { FT (1.0f / 2.0f) };
        points = { Point_2(1.0f / 3.0f, 1.0f / 3.0f) };
    } else if (order == 2) {
        weights = { FT(1.0f / 6.0f), FT(1.0f / 6.0f), FT(1.0f / 6.0f) };
        points = { Point_2(1.0f / 6.0f, 1.0f / 6.0f)
                 , Point_2(2.0f / 3.0f, 1.0f / 6.0f)
                 , Point_2(1.0f / 6.0f, 2.0f / 3.0f)
                 };
    } else if (order == 3) {
        weights = { FT(-27.0f / 96.0f), FT(25.0f / 96.0f), FT(25.0f / 96.0f), FT(25.0f / 96.0f) };
        points = { Point_2(1.0f / 3.0f, 1.0f / 3.0f)
                 , Point_2(1.0f / 5.0f, 1.0f / 5.0f)
                 , Point_2(1.0f / 5.0f, 3.0f / 5.0f)
                 , Point_2(3.0f / 5.0f, 1.0f / 5.0f)
                 };
    }

    FT res = 0;
    for (std::size_t i = 0; i < weights.size(); ++i) {
        FT alpha = points[i].x(), beta = points[i].y(), gamma = 1 - alpha - beta;
        Point p = CGAL::ORIGIN + (gamma * v0 + alpha * v1 + beta * v2);
        res += weights[i] * f(p);
    }

    return 2 * sqrt(t.squared_area()) * res;
}

// Gaussian quadrature of order 1, 2 or 3 for a 3D density defined on a 3D triangle.
template < typename K >
typename K::FT
integrate (std::function<typename K::FT(typename K::Point_3 const&)> const& f,
           typename CGAL::Point_3<K> const& a,
           typename CGAL::Point_3<K> const& b,
           typename CGAL::Point_3<K> const& c,
           unsigned int order = 2) {
    assert(order >= 1 && order <= 3);

    typedef typename K::FT FT;
    typedef typename K::Point_2 Point_2;
    typedef typename K::Vector_3 Vector;
    typedef typename K::Point_3 Point;

    Vector v0 = a - CGAL::ORIGIN,
           v1 = b - CGAL::ORIGIN,
           v2 = c - CGAL::ORIGIN;

    // Integration
    std::vector<FT> weights;
    std::vector<Point_2> points;
    if (order == 1) {
        weights = { FT (1.0f / 2.0f) };
        points = { Point_2(1.0f / 3.0f, 1.0f / 3.0f) };
    } else if (order == 2) {
        weights = { FT(1.0f / 6.0f), FT(1.0f / 6.0f), FT(1.0f / 6.0f) };
        points = { Point_2(1.0f / 6.0f, 1.0f / 6.0f)
                 , Point_2(2.0f / 3.0f, 1.0f / 6.0f)
                 , Point_2(1.0f / 6.0f, 2.0f / 3.0f)
                 };
    } else if (order == 3) {
        weights = { FT(-27.0f / 96.0f), FT(25.0f / 96.0f), FT(25.0f / 96.0f), FT(25.0f / 96.0f) };
        points = { Point_2(1.0f / 3.0f, 1.0f / 3.0f)
                 , Point_2(1.0f / 5.0f, 1.0f / 5.0f)
                 , Point_2(1.0f / 5.0f, 3.0f / 5.0f)
                 , Point_2(3.0f / 5.0f, 1.0f / 5.0f)
                 };
    }

    FT res = 0;
    for (std::size_t i = 0; i < weights.size(); ++i) {
        FT alpha = points[i].x(), beta = points[i].y(), gamma = 1 - alpha - beta;
        Point p = CGAL::ORIGIN + (gamma * v0 + alpha * v1 + beta * v2);
        res += weights[i] * f(p);
    }

    FT squared_double_area = CGAL::cross_product(b - a, c - a).squared_length();

    if (squared_double_area == 0.0) {
        return 0.0;
    } else {
        return 0.5 * srqt(squared_double_area) * res;
    }
}

// Gaussian quadrature of order 1 for a 3D density defined on a 3D triangle.
template < typename K >
typename K::FT
integrate_1 (std::function<typename K::FT(typename K::Point_3 const&)> const& f,
             typename CGAL::Point_3<K> const& a,
             typename CGAL::Point_3<K> const& b,
             typename CGAL::Point_3<K> const& c) {
    typedef typename K::FT FT;
    typedef typename K::Vector_3 Vector;
    typedef typename K::Point_3 Point;

    Vector v0 = a - CGAL::ORIGIN,
           v1 = b - CGAL::ORIGIN,
           v2 = c - CGAL::ORIGIN;

    // Integration
    Point centroid = CGAL::ORIGIN + (v0 + v1 + v2) / 3;
    FT squared_double_area = CGAL::cross_product(b - a, c - a).squared_length();

    if (squared_double_area == 0.0) {
        return 0.0;
    } else {
        return 0.5 * sqrt(squared_double_area) * f(centroid);
    }
}

#endif

