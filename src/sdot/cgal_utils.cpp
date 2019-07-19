#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;

#include <Eigen/Dense>

typedef Eigen::MatrixXd MatrixXd;
typedef Eigen::VectorXd VectorXd;

#include "conforming_lloyd_2.hpp"
#include "conforming_lloyd_3.hpp"

Eigen::MatrixXd
conforming_lloyd_2(Eigen::MatrixXd const& c,
                   Eigen::MatrixXd const& Y,
                   Eigen::VectorXd const& w,
                   Eigen::MatrixXd const& poly) {
    std::vector<Point_2> centroids;
    std::vector<Point_2> vertices;
    std::vector<double> weights;

    for (int i = 0; i < c.rows(); ++i) {
        centroids.emplace_back(c(i, 0), c(i, 1));
    }

    for (int i = 0; i < Y.rows(); ++i) {
        vertices.emplace_back(Y(i, 0), Y(i, 1));
    }

    for (int i = 0; i < w.rows(); ++i) {
        weights.push_back(w(i));
    }

    return MA::conforming_lloyd_2(centroids, vertices, weights, poly);
}

Eigen::MatrixXd
conforming_lloyd_3 (Eigen::MatrixXd const& c,
                    Eigen::MatrixXd const& Y,
                    Eigen::VectorXd const& w,
                    const Eigen::MatrixXd &X_tri,
                    const Eigen::MatrixXi &E_tri) {
    std::vector<Point_3> centroids;
    std::vector<Point_3> vertices;
    std::vector<double> weights;

    for (int i = 0; i < c.rows(); ++i) {
        centroids.emplace_back(c(i, 0), c(i, 1), c(i, 2));
    }

    for (int i = 0; i < Y.rows(); ++i) {
        vertices.emplace_back(Y(i, 0), Y(i, 1), Y(i, 2));
    }

    for (int i = 0; i < w.rows(); ++i) {
        weights.push_back(w(i));
    }

    return MA::conforming_lloyd_3(centroids, vertices, weights, X_tri, E_tri);
}

PYBIND11_MODULE(cgal_utils, m) {
    m.doc() = "CGAL utilities";

    m.def("conforming_lloyd_2", &conforming_lloyd_2);
    m.def("conforming_lloyd_3", &conforming_lloyd_3);
}

