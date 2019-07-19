#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include <Eigen/Dense>

#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_triangle_primitive.h>
#include <CGAL/Simple_cartesian.h>
#include <vector>

typedef CGAL::Simple_cartesian<double> K;
typedef K::Point_3 Point_3;
typedef K::Ray_3 Ray_3;
typedef K::Vector_3 Vector_3;
typedef K::Triangle_3 Triangle_3;

typedef std::vector<Triangle_3>::iterator Iterator;
typedef CGAL::AABB_triangle_primitive<K, Iterator> Primitive;
typedef CGAL::AABB_traits<K, Primitive> AABB_triangle_traits;
typedef CGAL::AABB_tree<AABB_triangle_traits> AABBTree;

class TriangleSoup : public AABBTree {
    public:
        TriangleSoup (const Eigen::MatrixXd &points,
                      const Eigen::MatrixXi &triangles) {
            for (int i = 0; i < triangles.rows(); ++i) {
                Point_3 pa(points(triangles(i, 0), 0),
                           points(triangles(i, 0), 1),
                           points(triangles(i, 0), 2));

                Point_3 pb(points(triangles(i, 1), 0),
                           points(triangles(i, 1), 1),
                           points(triangles(i, 1), 2));

                Point_3 pc(points(triangles(i, 2), 0),
                           points(triangles(i, 2), 1),
                           points(triangles(i, 2), 2));

                _triangles.emplace_back(pa, pb, pc);
            }

            insert(_triangles.begin(),_triangles.end());
        }

        std::pair<Eigen::MatrixXd, Eigen::VectorXi>
        intersection_with_rays (Eigen::MatrixXd const& rays) const {
            Eigen::MatrixXd ipoints(rays.rows(), 3);
            Eigen::VectorXi iids(rays.rows());

            for (int i = 0; i < rays.rows(); ++i) {
                Ray_3 p(Point_3(rays(i, 0), rays(i, 1), rays(i, 2)),
                        Vector_3(rays(i, 3), rays(i, 4), rays(i, 5)));
                auto oid = any_intersection(p);
                Point_3 point;
                if (oid && CGAL::assign(point, oid->first)) {
                    ipoints(i, 0) = point.x();
                    ipoints(i, 1) = point.y();
                    ipoints(i, 2) = point.z();
                    iids(i) = oid->second - _triangles.begin();
                }
            }

            return {ipoints, iids};
        }

        Eigen::VectorXd
        squared_distances (Eigen::MatrixXd const& points) const {
            Eigen::VectorXd distances(points.rows());

            for (int i = 0; i < points.rows(); ++i) {
                Point_3 p(points(i, 0), points(i, 1), points(i, 2));
                distances(i) = squared_distance(p);
            }

            return distances;
        }

    private:
        std::vector<Triangle_3> _triangles;
};

PYBIND11_MODULE(aabb_tree, m) {
    m.doc() = "AABB Tree";

    py::class_<TriangleSoup>(m, "TriangleSoup")
        .def(py::init<Eigen::MatrixXd const&, Eigen::MatrixXi const&>())
        .def("intersection_with_rays", &TriangleSoup::intersection_with_rays)
        .def("squared_distances", &TriangleSoup::squared_distances);
}

