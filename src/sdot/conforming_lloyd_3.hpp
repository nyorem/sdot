#ifndef _CONFORMING_LLOYG_3_HPP_
#define _CONFORMING_LLOYG_3_HPP_

#include <CGAL/Regular_triangulation_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>

typedef CGAL::Weighted_point_3<Kernel> Weighted_point_3;
typedef CGAL::Regular_triangulation_vertex_base_3<Kernel> Vbase_3;
typedef CGAL::Triangulation_vertex_base_with_info_3<int, Kernel, Vbase_3> Vb_3;
typedef CGAL::Regular_triangulation_cell_base_3<Kernel> Cb_3;
typedef CGAL::Triangulation_data_structure_3<Vb_3, Cb_3> Tds_3;
typedef CGAL::Regular_triangulation_3<Kernel, Tds_3> RT_3;
typedef CGAL::Point_3<Kernel> Point_3;
typedef CGAL::Segment_3<Kernel> Segment_3;
typedef CGAL::Plane_3<Kernel> Plane_3;

#include "predicates_sphere.hpp" // for do_power_face_segment_intersect and plane_line_intersect

typedef MA::Power_sphere_intersection_traits<Kernel> Intersection_traits;

namespace MA {
    namespace details {
        RT_3
        make_regular_triangulation (std::vector<Point_3> const& points,
                                    std::vector<double> const& weights) {
            int N = points.size();
            std::vector<std::pair<Weighted_point_3, size_t>> prt;
            for (int i = 0; i < N; ++i) {
                prt.emplace_back(Weighted_point_3(points[i], weights[i]), i);
            }

            RT_3 rt(prt.begin(), prt.end());
            rt.infinite_vertex()->info() = -1;
            return rt;
        }

        Point_3 construct_vertex_inter_face (RT_3::Edge const& f, Segment_3 const& s) {
            Plane_3 p = CGAL::radical_plane(f.first->vertex(f.second)->point(),
                                          f.first->vertex(f.third)->point());

            return plane_line_intersection<Kernel>::compute(p, s.supporting_line());
        }

        void compute_adjacencies_with_triangulation (std::vector<Point_3> const& X,
                                                     std::vector<double> const& weights,
                                                     Eigen::MatrixXd const& X_tri,
                                                     Eigen::MatrixXi const& E_tri,
                                                     std::vector<std::vector<Segment_3>> &adjedges,
                                                     std::vector<std::vector<size_t>> &adjverts) {
            auto rt = details::make_regular_triangulation(X, weights);

            int Nv = X.size();
            adjedges.assign(Nv, std::vector<Segment_3>());
            adjverts.assign(Nv, std::vector<size_t>());

            for (int e = 0; e < E_tri.rows(); ++e) {
                int p = E_tri(e, 0), pnext = E_tri(e, 1);

                Point_3 source(X_tri(p, 0), X_tri(p, 1), X_tri(p, 2)),
                      target(X_tri(pnext, 0), X_tri(pnext, 1), X_tri(pnext, 2));
                Segment_3 s(source, target);

                auto u = rt.nearest_power_vertex(source);
                auto v = rt.nearest_power_vertex(target);

                adjverts[u->info()].push_back(p);
                Point_3 pointprev = source;

                auto uprev = u;
                while (u != v) {
                    std::vector<RT_3::Edge> edges_rt;
                    rt.finite_incident_edges(u, std::back_inserter(edges_rt));

                    for (const auto& f: edges_rt) {
                        if (rt.is_infinite(f))
                            continue;

                        // we do not want to go back to the previous vertex!
                        auto unext = f.first->vertex(f.third);
                        if (unext == uprev)
                            continue;

                        // check whether dual edge intersects with the constraint
                        if (do_power_face_segment_intersect<Intersection_traits>(rt, f, s)) {
                            Point_3 point = construct_vertex_inter_face(f, s);

                            adjedges[u->info()].push_back(Segment_3(pointprev, point));
                            pointprev = point;
                            uprev = u;
                            u = unext;

                            break;
                        }
                    }
                }

                adjverts[v->info()].push_back(pnext);
                adjedges[v->info()].emplace_back(pointprev, target);
            }
        }
    } // namespace details

    Eigen::MatrixXd
    conforming_lloyd_3 (const std::vector<Point_3> &centroids,
                        const std::vector<Point_3> &X,
                        const std::vector<double> &w,
                        const Eigen::MatrixXd &X_tri,
                        const Eigen::MatrixXi &E_tri) {
        Eigen::MatrixXd c(centroids.size(), 3);
        for (int i = 0; i < c.rows(); ++i) {
            c(i, 0) = centroids[i].x();
            c(i, 1) = centroids[i].y();
            c(i, 2) = centroids[i].z();
        }

        std::vector<std::vector<Segment_3>> adjedges;
        std::vector<std::vector<size_t>> adjverts;
        details::compute_adjacencies_with_triangulation(X, w, X_tri, E_tri, adjedges, adjverts);

        size_t N = X.size();
        for (size_t i = 0; i < N; ++i) {
            if (adjverts[i].size() != 0) {
                c.row(i) = X_tri.row(adjverts[i][0]);
            }

            if (adjedges[i].size() != 0) {
                double mindist = 1e10;
                Eigen::VectorXd proj;
                for (size_t j = 0; j < adjedges[i].size(); ++j) {
                    Eigen::Vector3d source(adjedges[i][j].source().x(),
                                           adjedges[i][j].source().y(),
                                           adjedges[i][j].source().z());
                    Eigen::Vector3d dest(adjedges[i][j].target().x(),
                                          adjedges[i][j].target().y(),
                                          adjedges[i][j].target().z());
                    auto p = projection_on_segment(source, dest, c.row(i));
                    double dp = (p - c.row(i).transpose()).squaredNorm();
                    if (mindist > dp) {
                        mindist = dp;
                        proj = p;
                    }
                }
                c.row(i) = proj;
            }
        }

        return c;
    }
} // namespace MA

#endif

