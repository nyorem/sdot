#ifndef _PREDICATES_SPHERE_HPP_
#define _PREDICATES_SPHERE_HPP_

#include <CGAL/Regular_triangulation_3.h>
#include <cassert>

#include "utils.hpp"

// TODO:
// - handle non bounded power faces

namespace CGAL {
    // Version of radical_plane with Weighted points
    // see Cartesian_kernel/include/CGAL/Cartesian/function_objects.h
    template < typename Kernel >
    typename Kernel::Plane_3
    radical_plane (const typename CGAL::Weighted_point_3<Kernel>&p,
                   const typename CGAL::Weighted_point_3<Kernel>&q) {
        typedef typename CGAL::Plane_3<Kernel> Plane_3;
        typedef typename Kernel::FT FT;

        const FT a = 2 * (q.point().x() - p.point().x());
        const FT b = 2 * (q.point().y() - p.point().y());
        const FT c = 2 * (q.point().z() - p.point().z());
        const FT d = CGAL::square(p.point().x()) +
            CGAL::square(p.point().y()) +
            CGAL::square(p.point().z()) - p.weight() -
            CGAL::square(q.point().x()) -
            CGAL::square(q.point().y()) -
            CGAL::square(q.point().z()) + q.weight();

        return Plane_3(a, b, c, d);
    }

    // || p - x ||_{pow} - || p - y ||_{pow}
    template < typename Kernel >
    CGAL::Comparison_result
    compare_power_distance (const typename Kernel::Point_3& p,
                            const typename CGAL::Weighted_point_3<Kernel>&x,
                            const typename CGAL::Weighted_point_3<Kernel>&y) {
        typename Kernel::Compare_power_distance_3 Compare_power_distance_3;

        return Compare_power_distance_3(p, x, y);
    }
} // namespace CGAL

namespace MA {
    // Common
    // A null facet.
    template < typename T >
    typename T::Facet null_facet () {
        return std::make_pair(typename T::Cell_handle(0), 0);
    }

    // A null edge.
    template < typename T >
    typename T::Edge null_edge () {
        return typename T::Edge(typename T::Cell_handle(0), 0, 0);
    }

    // Intersection betweeen two objects (of type A and B) and whose result is of type Res.
    template < typename A, typename B, typename Res >
    struct a_b_intersection {
        static Res compute (const A &a, const B &b) {
            auto isect = CGAL::intersection(a, b);

            assert(isect);

            Res* res = boost::get<Res>(&*isect);
            assert(res);
            return *res;
        }
    };

    // Intersection point between a plane and a line.
    template < typename K >
    using plane_line_intersection = a_b_intersection<CGAL::Plane_3<K>, CGAL::Line_3<K>,
                                                     CGAL::Point_3<K>>;

    // Intersection line between two planes.
    template < typename K >
    using plane_plane_intersection = a_b_intersection<CGAL::Plane_3<K>, CGAL::Plane_3<K>,
                                                      CGAL::Line_3<K>>;

    // Find the two neighbouring cells of a facet.
    template < typename T >
    std::pair<typename T::Cell_handle, typename T::Cell_handle>
    cells_from_facet (typename T::Facet const& f) {
        return std::make_pair(f.first, f.first->neighbor(f.second));
    }

    // Finite edges that form the boundary of the power face 'e' (dual of an edge)
    template < typename RT >
    std::vector<typename RT::Cell_handle>
    boundary_edges (RT const& rt, typename RT::Edge const& e) {
        typename RT::Cell_circulator c0 = rt.incident_cells(e), c = c0;
        std::vector<typename RT::Cell_handle> cells;

        do {
            if (rt.is_infinite(c))
                continue;

            cells.push_back(c);
        } while (++c != c0);

        return cells;
    }

    // Compute the finite boundary of the dual facet associated to the edge e.
    template < typename RT >
    std::vector<typename RT::Bare_point>
    dual_facet (RT const& rt,
                typename RT::Edge const& e) {
        std::vector<typename RT::Cell_handle> cells = boundary_edges(rt, e);
        std::vector<typename RT::Bare_point> points;

        std::transform(cells.begin(), cells.end(), std::back_inserter(points),
                       [&] (typename RT::Cell_handle const& c) {
                           return rt.dual(c);
                       });

        return points;
    }

    // Intersection between a power face (= dual of an edge) and a segment.
    template < typename Traits, typename RT >
    bool do_power_face_segment_intersect (RT const& rt,
                                          typename RT::Edge const& f,
                                          typename RT::Segment const& s) {
        typedef typename RT::Weighted_point Weighted_point;
        typedef typename RT::Bare_point Point;
        typedef typename RT::Plane Plane;

        typename Traits::Power_face_segment_intersect Power_face_segment_intersect;
        typename Traits::template Point_inside_power_face<RT> Point_inside_power_face(rt);

        Weighted_point u = f.first->vertex(f.second)->point(),
                       v = f.first->vertex(f.third)->point();

        if (Power_face_segment_intersect(u, v, s.source(), s.target())) {
            Plane p = CGAL::radical_plane(u, v);

            auto mx = CGAL::intersection(p, s);
            Point* x = boost::get<Point>(&*mx);
            assert(x);

            return Point_inside_power_face(f, *x);
        }

        return false;
    }

    // Intersection between a power edge (= dual of a facet) and a triangle.
    // TODO: segfaults
    template < typename Traits, typename RT, typename Face_handle_T >
    bool do_power_edge_triangle_intersect (typename RT::Facet const& e,
                                           Face_handle_T const& t) {
        typename Traits::Power_line_triangle_intersect Power_line_triangle_intersect;
        typename Traits::Plane_cut_power_edge Plane_cut_power_edge;

        typedef typename RT::Weighted_point Weighted_point;

        Weighted_point u = e.first->vertex((e.second + 1) % 4)->point(),
                       v = e.first->vertex((e.second + 2) % 4)->point(),
                       w = e.first->vertex((e.second + 3) % 4)->point();

        Weighted_point up = e.first->vertex(e.second)->point(),
                       um = e.first->neighbor(e.second)->vertex(e.second)->point();

        return Power_line_triangle_intersect(u, v, w, t->vertex(0), t->vertex(1), t->vertex(2))
            && Plane_cut_power_edge(u, v, w, up, um, t->vertex(0), t->vertex(1), t->vertex(2));
    }

    // Intersection between a triangle and a power edge (= dual of a facet).
    // TODO: replace with do_power_edge_triangle_intersect
    template < typename RT
             , typename Face_handle_T
             >
    bool do_triangle_power_edge_intersect (RT const& rt,
                                           Face_handle_T const& t,
                                           typename RT::Facet const& e) {
        typedef typename CGAL::Kernel_traits<typename RT::Point>::Kernel Kernel;
        typedef typename Kernel::Segment_3 Segment;
        typedef typename Kernel::Ray_3 Ray;
        typedef typename Kernel::Line_3 Line;

        CGAL::Object obj = rt.dual(e);

        if (const Segment* s = CGAL::object_cast<Segment>(&obj)) {
            return CGAL::do_intersect(*s, *t);
        } else if (const Ray* r = CGAL::object_cast<Ray>(&obj)) {
            return CGAL::do_intersect(*r, *t);
        }

        const Line* l = CGAL::object_cast<Line>(&obj);
        return CGAL::do_intersect(*l, *t);
    }

    // Intersection between the boundary of a power face and a triangle.
    template < typename Traits, typename T, typename RT >
    bool
    do_boundary_power_face_triangle_intersect (RT const& rt,
                                               typename RT::Edge const& f,
                                               typename T::Face_handle const& t,
                                               std::vector<typename RT::Facet>& epows) {
        typename RT::Facet_circulator fc = rt.incident_facets(f), fc0 = fc;

        do {
            typename RT::Facet e = *fc;

            if (rt.is_infinite(e))
                continue;

            if (do_triangle_power_edge_intersect(rt, t, e)) {
            /* if (do_power_edge_triangle_intersect<Traits, RT>(e, t)) { */
                epows.push_back(e);
            }
        } while (++fc != fc0);

        return ! epows.empty();
    }

    // Intersection between a triangle and a power face (= dual of an edge).
    // We can optionally give a starting edge.
    template < typename Traits
             , typename RT
             , typename Face_handle_T
             , typename Edge_T
             >
    bool do_triangle_power_face_intersect (RT const& rt,
                                           typename RT::Edge const& f,
                                           Face_handle_T const& t,
                                           std::vector<Edge_T>& res,
                                           Maybe<Edge_T> const& start = boost::none) {
        typedef typename RT::Segment Segment;

        Edge_T e0 = t->edge(0), e1 = t->edge(1), e2 = t->edge(2);

        bool result0 = do_power_face_segment_intersect<Traits>(rt, f, Segment(t->vertex(0), t->vertex(1))),
             result1 = do_power_face_segment_intersect<Traits>(rt, f, Segment(t->vertex(1), t->vertex(2))),
             result2 = do_power_face_segment_intersect<Traits>(rt, f, Segment(t->vertex(2), t->vertex(0)));

        if (result0 || result1 || result2) {
            if (result0) {
                if (!start || start.value() != e0)
                    res.push_back(e0);
            }

            if (result1) {
                if (!start || start.value() != e1)
                    res.push_back(e1);
            }

            if (result2) {
                if (!start || start.value() != e2)
                    res.push_back(e2);
            }
        }

        return ! res.empty();
    }

    // Traits class for intersecting a 3D power diagram and a 2D triangulation of S^2.
    class Power_sphere_intersection_traits_base {
        public:
            // Constructions
            struct Construct_dual {
                    template <class K>
                    typename CGAL::Plane_3<K>
                    operator() (const typename CGAL::Point_3<K>& p,
                                const typename CGAL::Point_3<K>& q) const {
                        return CGAL::bisector(p, q);
                    }

                    template <class K>
                    typename K::Plane_3
                    operator() (const typename CGAL::Weighted_point_3<K> &p,
                                const typename CGAL::Weighted_point_3<K> &q) const {
                        return CGAL::radical_plane(p, q);
                    }

                    template <class K>
                    typename CGAL::Line_3<K>
                    operator() (const typename CGAL::Point_3<K>& p,
                                const typename CGAL::Point_3<K>& q,
                                const typename CGAL::Point_3<K>& r) const {
                        typename CGAL::Plane_3<K> p1 = (*this)(p, q),
                                                  p2 = (*this)(q, r);

                        return plane_plane_intersection<K>::compute(p1, p2);
                    }

                    template <class K>
                    typename K::Line_3
                    operator() (const typename CGAL::Weighted_point_3<K> &p,
                                const typename CGAL::Weighted_point_3<K> &q,
                                const typename CGAL::Weighted_point_3<K> &r) const {
                        typename CGAL::Plane_3<K> p1 = (*this)(p, q),
                                                  p2 = (*this)(q, r);

                        return plane_plane_intersection<K>::compute(p1, p2);
                    }

                    template <class K>
                    typename K::Point_3
                    operator() (const typename CGAL::Weighted_point_3<K> &p,
                                const typename CGAL::Weighted_point_3<K> &q,
                                const typename CGAL::Weighted_point_3<K> &r,
                                const typename CGAL::Weighted_point_3<K> &s) const {
                        return CGAL::weighted_circumcenter(p, q, r, s);
                    }

                    template <class K>
                    typename CGAL::Point_3<K>
                    operator() (const typename CGAL::Point_3<K> &p,
                                const typename CGAL::Point_3<K> &q,
                                const typename CGAL::Point_3<K> &r,
                                const typename CGAL::Point_3<K> &s) const {
                        return CGAL::circumcenter(p, q, r, s);
                    }
            };

            // Predicates
            // Intersection between the line defined by a power edge (dual of three points) and a triangle (3 points).
            // TODO: check
            struct Power_line_triangle_intersect {
                typedef bool result_type;

                template < typename Weighted_point, typename Point >
                result_type
                operator() (Weighted_point const& u, Weighted_point const& v, Weighted_point const& w,
                            Point const& a, Point const& b, Point const& c) const {
                    Weighted_point bisectors[3] = {u, v, w};

                    for (int i = 0; i < 3; ++i) {
                        Weighted_point const& u1 = bisectors[i], u2 = bisectors[(i + 1) % 3];

                        CGAL::Comparison_result resa = CGAL::compare_power_distance(a, u1, u2),
                                                resb = CGAL::compare_power_distance(b, u1, u2),
                                                resc = CGAL::compare_power_distance(c, u1, u2);

                        if (resa == resb && resb == resc && resc == resa)
                            return false;
                    }

                    return true;
                }
            };

            // Test if a plane (3 points) cuts a power edge (dual of 5 points)
            struct Plane_cut_power_edge {
                typedef bool result_type;

                template < typename Weighted_point, typename Point >
                result_type
                operator() (Weighted_point const& u, Weighted_point const& v, Weighted_point const& w,
                            Weighted_point const& up, Weighted_point const& um,
                            Point const& a, Point const& b, Point const& c) const {
                    typedef typename CGAL::Kernel_traits<Point>::Kernel K;
                    typedef typename CGAL::Plane_3<K> Plane;

                    Point e1 = CGAL::weighted_circumcenter(u, v, w, up),
                          e2 = CGAL::weighted_circumcenter(u, v, w, um);

                    Plane p(a, b, c);

                    return p.oriented_side(e1) != p.oriented_side(e2);
                }
            };

            // Intersection between a power face (dual of 2 points) and a segment
            struct Power_face_segment_intersect {
                typedef bool result_type;

                template < typename Weighted_point, typename Point >
                result_type
                operator() (Weighted_point const& u, Weighted_point const& v,
                            Point const& a, Point const& b) const {
                    return CGAL::compare_power_distance(a, u, v) != CGAL::compare_power_distance(b, u, v);
                }
            };

            // Test if a point belongs to a power face (dual of an edge)
            template < typename RT >
            struct Point_inside_power_face {
                RT const& rt;

                Point_inside_power_face (RT const& rt) : rt(rt) {
                }

                typedef bool result_type;
                typedef typename RT::Edge Edge;
                typedef typename RT::Vertex_handle Vertex_handle;
                typedef typename RT::Cell_circulator Cell_circulator;
                typedef typename RT::Weighted_point Weighted_point;

                template < typename Point >
                result_type
                operator() (Edge const& e, Point const& x) const {
                    Cell_circulator c0 = rt.incident_cells(e), c = c0;

                    Vertex_handle u = e.first->vertex(e.second),
                                  v = e.first->vertex(e.third);

                    do {
                        if (rt.is_infinite(c))
                            continue;

                        int i1 = -1, i2 = -1;
                        for (int i = 0; i < 4; ++i) {
                            if (c->vertex(i) != u && c->vertex(i) != v) {
                                if (i1 == -1) {
                                    i1 = i;
                                } else if (i2 == -1) {
                                    i2 = i;
                                } else {
                                    break;
                                }
                            }
                        }

                        assert(i1 != -1 && i1 != -1);

                        Weighted_point u1 = c->vertex(i1)->point(),
                                       u2 = c->vertex(i2)->point();

                        if (CGAL::compare_power_distance(x, v->point(), u1) != CGAL::SMALLER ||
                            CGAL::compare_power_distance(x, v->point(), u2) != CGAL::SMALLER)
                            return false;
                    } while (++c != c0);

                    return true;
                }
            };
    };

    // Filtered traits class
    template < typename K >
    class Power_sphere_intersection_traits {
        public:
            typedef K Kernel;

            typedef typename K::Exact_kernel_rt EK;
            typedef typename K::Approximate_kernel Approximate_kernel;

            typedef Power_sphere_intersection_traits_base Exact_traits;
            typedef Power_sphere_intersection_traits_base Filtering_traits;

            typedef typename K::C2E C2E;
            typedef typename K::C2F C2F;

            // Constructions
            typedef Power_sphere_intersection_traits_base::Construct_dual Construct_dual;

            // Filtered predicates
            // Power_line_triangle_intersect
            typedef CGAL::Filtered_predicate<
                typename Exact_traits::Power_line_triangle_intersect,
                typename Filtering_traits::Power_line_triangle_intersect,
                C2E, C2F > Power_line_triangle_intersect;

            // Plane_cut_power_edge
            typedef CGAL::Filtered_predicate<
                typename Exact_traits::Plane_cut_power_edge,
                typename Filtering_traits::Plane_cut_power_edge,
                C2E, C2F > Plane_cut_power_edge;

            // Power_face_segment_intersect
            typedef CGAL::Filtered_predicate<
                typename Exact_traits::Power_face_segment_intersect,
                typename Filtering_traits::Power_face_segment_intersect,
                C2E, C2F > Power_face_segment_intersect;

            // Point_inside_power_face
            template < typename RT >
            using Point_inside_power_face = Filtering_traits::Point_inside_power_face<RT>;
    };

} // namespace MA

#endif

