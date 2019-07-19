#ifndef _CONFORMING_LLOYD_2_HPP_
#define _CONFORMING_LLOYD_2_HPP_

// Adapated from: https://github.com/mrgt/PyMongeAmpere/blob/master/MongeAmpere.cpp#L667

#include <CGAL/Regular_triangulation_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>

// Regular triangulation
typedef CGAL::Regular_triangulation_vertex_base_2<Kernel> RTVbase;
typedef CGAL::Triangulation_vertex_base_with_info_2 <size_t, Kernel, RTVbase> RTVb;
typedef CGAL::Regular_triangulation_face_base_2<Kernel> RTCb;
typedef CGAL::Triangulation_data_structure_2<RTVb,RTCb> RTTds;
typedef CGAL::Regular_triangulation_2<Kernel, RTTds> RT_2;
typedef RT_2::Weighted_point Weighted_point_2;

typedef CGAL::Point_2<Kernel> Point_2;
typedef CGAL::Segment_2<Kernel> Segment_2;

template <class K>
bool
object_contains_point(const CGAL::Object &oi, CGAL::Point_2<K> &intp) {
    if(const CGAL::Point_2<K>* r = CGAL::object_cast< CGAL::Point_2<K> >(&oi))
    {
        intp = *r;
        return true;
    }
    else if(const CGAL::Segment_2<K>* s = CGAL::object_cast< CGAL::Segment_2<K> >(&oi))
    {
        intp = CGAL::midpoint(s->source(), s->target());
        return true;
    }
    return false;
}

template <class K>
bool
edge_dual_and_segment_isect(const CGAL::Object &o,
                            const CGAL::Segment_2<K> &s,
                            CGAL::Point_2<K> &intp) {
    if (const CGAL::Segment_2<K> *os = CGAL::object_cast< CGAL::Segment_2<K> >(&o))
        return object_contains_point(CGAL::intersection(*os, s), intp);
    if (const CGAL::Line_2<K> *ol = CGAL::object_cast<CGAL::Line_2<K> >(&o))
        return object_contains_point(CGAL::intersection(*ol, s), intp);
    if (const CGAL::Ray_2<K> *orr = CGAL::object_cast< CGAL::Ray_2<K> >(&o))
        return object_contains_point(CGAL::intersection(*orr, s), intp);
    return false;
}

namespace MA {
    namespace details {
        RT_2
        make_regular_triangulation (std::vector<Point_2> const& points,
                                    std::vector<double> const& weights) {
            int N = points.size();
            std::vector<std::pair<Weighted_point_2, int>> prt;
            for (int i = 0; i < N; ++i) {
                prt.emplace_back(Weighted_point_2(points[i], weights[i]), i);
            }

            RT_2 rt(prt.begin(), prt.end());
            rt.infinite_vertex()->info() = -1;
            return rt;
        }
    } // namespace details
} // namespace MA

void compute_adjacencies_with_polygon (std::vector<Point_2> const& X,
                                       std::vector<double> const& weights,
                                       Eigen::MatrixXd const& polygon,
                                       std::vector<std::vector<Segment_2>> &adjedges,
                                       std::vector<std::vector<size_t>> &adjverts) {
    auto rt = MA::details::make_regular_triangulation(X,weights);

    int Np = polygon.rows();
    int Nv = X.size();
    adjedges.assign(Nv, std::vector<Segment_2>());
    adjverts.assign(Nv, std::vector<size_t>());

    for (int p = 0; p < Np; ++p) {
        int pnext = (p + 1) % Np;
        Point_2 source(polygon(p,0), polygon(p,1));
        Point_2 target(polygon(pnext,0), polygon(pnext,1));

        auto u = rt.nearest_power_vertex(source);
        auto v = rt.nearest_power_vertex(target);

        adjverts[u->info()].push_back(p);
        Point_2 pointprev = source;

        auto  uprev = u;
        while (u != v) {
            // find next vertex intersecting with  segment
            auto c = rt.incident_edges(u), done(c);
            do {
                if (rt.is_infinite(c))
                    continue;

                // we do not want to go back to the previous vertex!
                auto unext = (c->first)->vertex(rt.ccw(c->second));
                if (unext == uprev)
                    continue;

                // check whether dual edge (which can be a ray, a line
                // or a segment) intersects with the constraint
                Point_2 point;
                if (!edge_dual_and_segment_isect(rt.dual(c),
                                                 Segment_2(source,target),
                                                 point))
                    continue;

                adjedges[u->info()].push_back(Segment_2(pointprev,point));
                pointprev = point;
                uprev = u;
                u = unext;

                break;
            } while(++c != done);
        }

        adjverts[v->info()].push_back(pnext);
        adjedges[v->info()].push_back(Segment_2(pointprev, target));
    }
}

// Return projection of p on [v,w]
Eigen::VectorXd projection_on_segment(Eigen::VectorXd const& v,
                                      Eigen::VectorXd const& w,
                                      Eigen::VectorXd const& p) {
    double l2 = (v-w).squaredNorm();
    if (l2 <= 1e-10)
        return v;

    // Consider the line extending the segment, parameterized as v + t
    // (w - v).  We find projection of point p onto the line.  It falls
    // where t = [(p-v) . (w-v)] / |w-v|^2
    double t = (p - v).dot(w - v) / l2;
    t = std::min(std::max(t,0.0), 1.0);

    return v + t * (w - v);
}

namespace MA {
    // poly is an array of the boundary vertices
    Eigen::MatrixXd
    conforming_lloyd_2(const std::vector<Point_2> &centroids,
                       const std::vector<Point_2> &X,
                       const std::vector<double> &w,
                       const Eigen::MatrixXd &poly) {
        Eigen::MatrixXd c(centroids.size(), 2);
        for (int i = 0; i < c.rows(); ++i) {
            c(i, 0) = centroids[i].x();
            c(i, 1) = centroids[i].y();
        }

        std::vector<std::vector<Segment_2>> adjedges;
        std::vector<std::vector<size_t>> adjverts;
        compute_adjacencies_with_polygon(X, w, poly, adjedges, adjverts);

        size_t N = centroids.size();
        for (size_t i = 0; i < N; ++i) {
            if (adjverts[i].size() != 0) {
                c.row(i) = poly.row(adjverts[i][0]);
            }

            if (adjedges[i].size() != 0) {
                double mindist = 1e10;
                Eigen::VectorXd proj;
                for (size_t j = 0; j < adjedges[i].size(); ++j) {
                    Eigen::Vector2d source (adjedges[i][j].source().x(),
                                            adjedges[i][j].source().y());
                    Eigen::Vector2d dest (adjedges[i][j].target().x(),
                                          adjedges[i][j].target().y());
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

