#ifndef _POWER_SPHERE_TRIANGULATION_INTERSECTION_HPP_
#define _POWER_SPHERE_TRIANGULATION_INTERSECTION_HPP_

#include <CGAL/Regular_triangulation_3.h>
#include <CGAL/Origin.h>
#include <CGAL/utility.h>
#include <algorithm>
#include <cassert>
#include <map>
#include <stack>

#include "predicates_sphere.hpp"
#include "utils.hpp"
#include "utils_ad.hpp"

#ifndef NDEBUG
#include <chrono>
#endif

namespace MA {
    namespace internal {
        template < typename Point_3 >
        struct Point3Compare {
            const double epsilon = 1e-8;

            bool operator() (Point_3 const& p, Point_3 const& q) const {
                if (p.x() < q.x() - epsilon)
                    return true;
                if (p.x() > q.x() + epsilon)
                    return false;

                if (p.y() < q.y() - epsilon)
                    return true;
                if (p.y() > q.y() + epsilon)
                    return false;

                if (p.z()< q.z() - epsilon)
                    return true;

                return false;
            }
        };
    } // namespace internal

    using Vertex_with_comp = std::pair<int, int>;
    using DualTriangulation = std::vector<std::tuple<Vertex_with_comp, Vertex_with_comp, Vertex_with_comp>>;

    // Intersection class (traits) that contains all necessary types
    template < typename T_, typename RT_, typename Traits >
    struct Tri_intersector {
        typedef T_ T;
        typedef RT_ RT;

        typedef typename CGAL::Kernel_traits<typename RT::Point>::Kernel K;
        typedef typename CGAL::Point_3<K> Point;
        typedef typename CGAL::Segment_3<K> Segment;
        typedef typename CGAL::Ray_3<K> Ray;
        typedef typename CGAL::Line_3<K> Line;
        typedef typename CGAL::Plane_3<K> Plane;
        typedef typename K::Weighted_point_3 Weighted_point;

        typedef typename T::Edge Edge_T;
        typedef typename T::Face_handle Face_handle_T;

        typedef typename RT::Vertex_handle Vertex_handle_RT;
        typedef typename RT::Facet Edge_RT; // an edge is the dual of a triangle
        typedef typename RT::Edge Face_RT; // a face is the dual of an edge
        typedef typename RT::Cell_handle Cell_handle_RT;

        // There are 3 types of vertices:
        // - the ones that are the intersection of a power edge and a triangle
        // - the ones that are the intersection of a power face and an edge of a triangle
        // - the ones that belongs to the triangulation
        enum VertexType { VERTEX_INTER_EDGE, VERTEX_INTER_FACE, VERTEX_T };
        typedef std::pair<Edge_RT, Face_handle_T> Vertex_INTER_EDGE;
        typedef std::pair<Face_RT, Edge_T> Vertex_INTER_FACE;
        typedef int Vertex_T;

        struct Pgon_vertex {
            VertexType type;
            Vertex_INTER_EDGE v_inter_edge;
            Vertex_INTER_FACE v_inter_face;
            Vertex_T v_t;

            int idx = -1;

            bool operator== (const Pgon_vertex &other) const {
                if (type != other.type)
                    return false;
                else if (type == VERTEX_INTER_EDGE)
                    return v_inter_edge == other.v_inter_edge;
                else if (type == VERTEX_INTER_FACE)
                    return v_inter_face == other.v_inter_face;

                // type == VERTEX_T
                return v_t == other.v_t;
            }

            bool is_vertex_inter_edge () const {
                return type == VERTEX_INTER_EDGE;
            }

            bool is_vertex_inter_face () const {
                return type == VERTEX_INTER_FACE;
            }

            bool is_vertex_t () const {
                return type == VERTEX_T;
            }
        };

        static Pgon_vertex make_vertex_inter_edge (Edge_RT const& a, Face_handle_T const& b) {
            Pgon_vertex v;

            v.type = VERTEX_INTER_EDGE;
            v.v_inter_edge = std::make_pair(a, b);
            v.v_inter_face = std::make_pair(null_edge<RT>(), Edge_T());
            v.v_t = -1;

            return v;
        }

        static Pgon_vertex make_vertex_inter_face (Face_RT const& a, Edge_T const& b) {
            Pgon_vertex v;

            v.type = VERTEX_INTER_FACE;
            v.v_inter_edge = std::make_pair(null_facet<RT>(), nullptr);
            v.v_inter_face = std::make_pair(a, b);
            v.v_t = -1;

            return v;
        }

        static Pgon_vertex make_vertex_t (Vertex_T a) {
            Pgon_vertex v;

            v.type = VERTEX_T;
            v.v_inter_edge = std::make_pair(null_facet<RT>(), nullptr);
            v.v_inter_face = std::make_pair(null_edge<RT>(), Edge_T());
            v.v_t = a;

            return v;
        }

        // An edge is composed of two vertices, delimits two power cells
        // (dual of two vertices) and is contained in a triangle.
        struct Pgon_edge {
            std::pair<Pgon_vertex, Pgon_vertex> endpoints;

            Pgon_vertex first () const {
                return endpoints.first;
            }

            Pgon_vertex& first () {
                return endpoints.first;
            }

            Pgon_vertex second () const {
                return endpoints.second;
            }

            Pgon_vertex& second () {
                return endpoints.second;
            }
        };

        static Pgon_edge make_edge (Pgon_vertex const& u, Pgon_vertex const& v) {
            Pgon_edge e;
            e.endpoints = std::make_pair(u, v);

            return e;
        }

        // Intersection vertex = one triangle of T, one power edge of P (three vertices)
        struct Vertex_inter_edge {
            std::tuple<Face_handle_T, Vertex_handle_RT, Vertex_handle_RT, Vertex_handle_RT> data;
            Face_handle_T tri;
            Edge_RT edge;

            Vertex_inter_edge (Face_handle_T t, Edge_RT e) : tri(t), edge(e) {
                int i = e.second;
                Vertex_handle_RT u = e.first->vertex((i + 1) % 4),
                                 v = e.first->vertex((i + 2) % 4),
                                 w = e.first->vertex((i + 3) % 4);

                sort_3(u, v, w);

                data = std::make_tuple(t, u, v, w);
            }

            bool operator== (Vertex_inter_edge const& other) const {
                return data == other.data;
            }

            bool operator< (Vertex_inter_edge const& other) const {
                return data < other.data;
            }

            Pgon_vertex to_pgon_vertex () const {
                Pgon_vertex v = make_vertex_inter_edge(edge, tri);
                return v;
            }
        };

        // Intersection vertex = one edge of T, one 2-face of P (i.e. two vertices)
        struct Vertex_inter_face {
            std::tuple<int, int, Vertex_handle_RT, Vertex_handle_RT> data;
            Edge_T edge;
            Face_RT face;

            Vertex_inter_face (Edge_T e, Face_RT f) : edge(e), face(f) {
                int x = e.source(), y = e.target();
                Vertex_handle_RT u = f.first->vertex(f.second),
                                 v = f.first->vertex(f.third);

                if (x > y)
                    std::swap(x, y);

                if (u > v)
                    std::swap(u, v);

                data = std::make_tuple(x, y, u, v);
            }

            bool operator== (Vertex_inter_face const& other) const {
                return data == other.data;
            }

            bool operator< (Vertex_inter_face const& other) const {
                return data < other.data;
            }

            Pgon_vertex to_pgon_vertex () const {
                Pgon_vertex v = make_vertex_inter_face(face, edge);
                return v;
            }
        };

        // Graphs
        // Triangulation
        struct EdgeGraph_T {
            std::pair<Pgon_vertex, Pgon_vertex> endpoints;
            std::pair<Face_handle_T, Face_handle_T> t;
            Vertex_handle_RT cell;

            EdgeGraph_T (Pgon_vertex const &u, Pgon_vertex const& v,
                         Face_handle_T const& t1, Face_handle_T const& t2,
                         Vertex_handle_RT const& c) : endpoints(u, v), t(t1, t2), cell(c) {
            }

            Pgon_edge to_pgon_edge () const {
                Pgon_edge e;
                e.endpoints = endpoints;
                return e;
            }
        };

        struct Graph_T {
            std::vector<EdgeGraph_T> edges;
            std::vector<Vertex_inter_face> vertices;
        };

        // Power diagram
        struct EdgeGraph_P {
            std::pair<Pgon_vertex, Pgon_vertex> endpoints;
            std::pair<Vertex_handle_RT, Vertex_handle_RT> cells;
            Face_handle_T t;

            EdgeGraph_P (Pgon_vertex const& u, Pgon_vertex const& v,
                         Vertex_handle_RT const& c1, Vertex_handle_RT const& c2,
                         Face_handle_T const& t) : endpoints(u, v), cells(c1, c2), t(t) {
            }

            Pgon_edge to_pgon_edge () const {
                Pgon_edge e;
                e.endpoints = endpoints;
                return e;
            }
        };

        // An edge between two Vertex_inter_edge that cuts two power cells.
        struct BigEdgeGraph_P {
            std::tuple<Face_handle_T, Face_handle_T, Vertex_handle_RT, Vertex_handle_RT> data;

            BigEdgeGraph_P (Face_handle_T t1, Face_handle_T t2,
                            Vertex_handle_RT c1, Vertex_handle_RT c2) {
                if (t1 > t2)
                    std::swap(t1, t2);

                if (c1 > c2)
                    std::swap(c1, c2);

                data = std::make_tuple(t1, t2, c1, c2);
            }

            bool operator== (BigEdgeGraph_P const& other) const {
                return data == other.data;
            }

            bool operator< (BigEdgeGraph_P const& other) const {
                return data < other.data;
            }
        };

        struct Graph_P {
            // Walk in the triangulation starting from a Vertex_inter_face
            std::pair<std::vector<EdgeGraph_P>, Pgon_vertex>
            intersect (RT const& rt,
                       Face_handle_T const& t0,
                       Vertex_inter_face const& v0) {
                Edge_T startE = v0.edge;
                Face_RT f = v0.face;

                Face_handle_T t = t0;
                Vertex_inter_face v = v0;

                std::vector<EdgeGraph_P> edges_p;

                while (t != nullptr) {
                    std::vector<Edge_T> restmp;
                    if (do_triangle_power_face_intersect<Traits>(rt, f, t, restmp, boost::make_optional(startE))) {
                        assert(restmp.size() == 1);
                        Edge_T start = restmp.front();

                        // Egde between v and (f inter start)
                        Vertex_inter_face vv(start, f);
                        edges_p.emplace_back(v.to_pgon_vertex(), vv.to_pgon_vertex(),
                                             f.first->vertex(f.second),
                                             f.first->vertex(f.third),
                                             t);
                        visited_vertices.insert(vv);

                        v = vv;
                        t = t->other(start);
                        startE = start;

                        if (t == t0)
                            break;
                    } else {
                        // Find the power edge intersecting t
                        std::vector<Edge_RT> epows_;
                        if (! do_boundary_power_face_triangle_intersect<Traits, T>(rt, f, t, epows_))
                            assert(false);
                        assert(epows_.size() == 1);
                        Edge_RT epow = epows_.front();
                        epows.push_back(epow);

                        // Edge between v and (epow inter t)
                        Vertex_inter_edge vv(t, epow);
                        edges_p.emplace_back(v.to_pgon_vertex(), vv.to_pgon_vertex(),
                                             f.first->vertex(f.second),
                                             f.first->vertex(f.third),
                                             t);

                        return {edges_p, vv.to_pgon_vertex()};
                    }
                }

                return {edges_p, v.to_pgon_vertex()};
            }

            // Walk in the triangulation starting from a Vertex_inter_edge
            std::pair<std::vector<EdgeGraph_P>, Pgon_vertex>
            intersect (RT const& rt, Face_RT const& f,
                       Vertex_inter_edge const& v0) {
                Face_handle_T t0 = v0.tri;

                std::vector<Edge_T> restmp;
                if (do_triangle_power_face_intersect<Traits>(rt, f, t0, restmp)) {
                    assert(restmp.size() == 1);

                    Edge_T start = restmp.front();

                    // Edge between v0 and (f inter start)
                    Vertex_inter_face v(start, f);
                    EdgeGraph_P e0(v0.to_pgon_vertex(), v.to_pgon_vertex(),
                                   f.first->vertex(f.second),
                                   f.first->vertex(f.third),
                                   t0);
                    visited_vertices.insert(v);

                    auto res = intersect(rt, t0->other(start), v);

                    // Add the first edge
                    res.first.insert(res.first.begin(), e0);

                    return res;
                }

                // Find the other power edge of f (different from v0.edge) interesting t0
                typename RT::Facet_circulator fc0 = rt.incident_facets(f, v0.edge), fc = fc0;
                fc++; // Skip first edge (= v0.edge)
                Edge_RT epow;

                do {
                    epow = *fc;
                    if (do_triangle_power_edge_intersect(rt, t0, epow)) {
                        break;
                    }
                } while (++fc != fc0);

                Vertex_inter_edge v(t0, epow);
                epows.push_back(epow);

                std::pair<std::vector<EdgeGraph_P>, Pgon_vertex> res;

                res.first.emplace_back(v0.to_pgon_vertex(), v.to_pgon_vertex(),
                                       f.first->vertex(f.second),
                                       f.first->vertex(f.third),
                                       t0);

                res.second = v.to_pgon_vertex();

                return res;
            }

            // Insert (if needed) the initial vertex.
            // v0.type == VERTEX_INTER_FACE
            bool insert_if_not_visited (Vertex_inter_face const& v0) {
                if (! visited_vertices.count(v0)) {
                    visited_vertices.insert(v0);
                    return false;
                }

                return true;
            }

            bool insert_if_not_visited (std::vector<EdgeGraph_P> const& edges_p,
                                        Pgon_vertex const& v0,
                                        Pgon_vertex const& vlast,
                                        Face_RT const& f) {
                Face_handle_T t0 = nullptr, tlast = nullptr;

                switch (v0.type) {
                    case VERTEX_INTER_FACE:
                        if (v0.v_inter_face.second.faces.first != nullptr) {
                            t0 = v0.v_inter_face.second.faces.first;
                        } else {
                            t0 = v0.v_inter_face.second.faces.second;
                        }
                        break;

                    case VERTEX_INTER_EDGE:
                        t0 = v0.v_inter_edge.second;
                        break;

                    default:
                        break;
                }

                switch (vlast.type) {
                    case VERTEX_INTER_FACE:
                        if (vlast.v_inter_face.second.faces.first != nullptr) {
                            tlast = vlast.v_inter_face.second.faces.first;
                        } else {
                            tlast = vlast.v_inter_face.second.faces.second;
                        }
                        break;

                    case VERTEX_INTER_EDGE:
                        tlast = vlast.v_inter_edge.second;
                        break;

                    default:
                        break;
                }

                BigEdgeGraph_P e_p(t0, tlast,
                                   f.first->vertex(f.second),
                                   f.first->vertex(f.third));

                if (! visited_edges.count(e_p)) {
                    edges.insert(edges.end(), edges_p.begin(), edges_p.end());
                    visited_edges.insert(e_p);
                    return true;
                }

                return false;
            }

            // Dual triangulation of the Laguerre diagram.
            DualTriangulation
            build_dual () const {
                DualTriangulation dual_tri;

                for (const auto& epow: epows) {
                    Vertex_handle_RT u = epow.first->vertex((epow.second + 1) % 4),
                                     v = epow.first->vertex((epow.second + 2) % 4),
                                     w = epow.first->vertex((epow.second + 3) % 4);

                    dual_tri.emplace_back(std::make_pair(u->info().index, 0),
                                          std::make_pair(v->info().index, 0),
                                          std::make_pair(w->info().index, 0));
                }

                return dual_tri;
            }

            std::vector<EdgeGraph_P> edges;

            std::set<Vertex_inter_face> visited_vertices;
            std::set<BigEdgeGraph_P> visited_edges;

            std::vector<Edge_RT> epows;
        };
    };

    // Policy class that defines all constructions
    template < typename T, typename RT, typename K, bool WithAD = false >
    struct Constructions;

    // Without AD
    template < typename T, typename RT, typename K >
    struct Constructions<T, RT, K, false> {
        typedef K Kernel_in;
        typedef K Kernel_out;

        // Output types (Kernel_out can be different from Kernel_in)
        typedef typename Kernel_out::Point_3 Point;
        typedef typename Kernel_out::Segment_3 Segment;
        typedef typename Kernel_out::Plane_3 Plane;
        typedef typename Kernel_out::Line_3 Line;

        typedef Power_sphere_intersection_traits<Kernel_in> Intersection_traits;
        typedef Tri_intersector<T, RT, Intersection_traits> Tri_isector;
        typedef typename Tri_isector::Pgon_vertex Pgon_vertex;
        typedef typename Tri_isector::Pgon_edge Pgon_edge;

        typedef typename Tri_isector::Edge_RT Edge_RT;
        typedef typename Tri_isector::Face_RT Face_RT;

        // vertex_to_point
        Point construct (T const& tri, RT const& rt, Pgon_vertex const& v) {
            if (v.type == Tri_isector::VERTEX_INTER_EDGE) {
                Edge_RT e = v.v_inter_edge.first;

                Line l(rt.dual(e.first),
                       rt.dual(e.first->neighbor(e.second)));
                Plane p = v.v_inter_edge.second->supporting_plane();

                return plane_line_intersection<K>::compute(p, l);
            } else if (v.type == Tri_isector::VERTEX_INTER_FACE) {
                Face_RT f = v.v_inter_face.first;

                Plane p = CGAL::radical_plane(f.first->vertex(f.second)->point(),
                                              f.first->vertex(f.third)->point());

                Line l = tri.segment(v.v_inter_face.second).supporting_line();

                return plane_line_intersection<K>::compute(p, l);
            }

            // v.type == VERTEX_T
            return tri.point(v.v_t);
        }

        // edge_to_segment
        Segment construct (T const& tri, RT const& rt, Pgon_edge const& e) {
            return Segment(construct(tri, rt, e.first()), construct(tri, rt, e.second()));
        }
    };

    // With AD
    // K = K_ad
    template < typename T, typename RT, typename K >
    struct Constructions<T, RT, K, true> {
        typedef typename T::Kernel Kernel_in;
        typedef K Kernel_out;

        typedef typename CGAL::Cartesian_converter<Kernel_in, K> To_ad;

        // Input types (without AD)
        typedef typename Kernel_in::Segment_3 Segment_in;
        typedef typename Kernel_in::Ray_3 Ray_in;

        // Output types (Kernel_out can be different from Kernel_in)
        typedef typename Kernel_out::Point_3 Point;
        typedef typename Kernel_out::Segment_3 Segment;
        typedef typename Kernel_out::Plane_3 Plane;
        typedef typename Kernel_out::Line_3 Line;
        typedef typename Kernel_out::Weighted_point_3 Weighted_point;

        typedef Power_sphere_intersection_traits<Kernel_in> Intersection_traits;
        typedef Tri_intersector<T, RT, Intersection_traits> Tri_isector;
        typedef typename Tri_isector::Pgon_vertex Pgon_vertex;
        typedef typename Tri_isector::Pgon_edge Pgon_edge;

        typedef typename Tri_isector::Edge_RT Edge_RT;
        typedef typename Tri_isector::Face_RT Face_RT;
        typedef typename Tri_isector::Face_handle_T Face_handle_T;
        typedef typename Tri_isector::Edge_T Edge_T;

        // Caches
        // Vertex_inter_edge
        std::map<Face_handle_T, Plane> triangle_plane_cache;
        std::map<Edge_RT, Line> edge_line_cache;

        // Vertex_inter_face
        std::map<Face_RT, Plane> face_plane_cache;
        std::map<Edge_T, Line> triangle_edge_line_cache;

        // vertex_to_point
        Point construct (T const& tri, RT const& rt, Pgon_vertex const& v) {
            To_ad to_ad;

            if (v.type == Tri_isector::VERTEX_INTER_EDGE) {
                // TODO: e -> 2 tetraedres => stocker les weighted_circumcenter dans Cell_with_info dans la fonction dual
                Edge_RT e = v.v_inter_edge.first;
                if (edge_line_cache.count(e) == 0) {
                    Line ll(CGAL::dual<Point>(rt, e.first),
                            CGAL::dual<Point>(rt, e.first->neighbor(e.second)));
                    edge_line_cache[e] = ll;
                }
                Line l = edge_line_cache[e];

                Face_handle_T t = v.v_inter_edge.second;
                if (triangle_plane_cache.count(t) == 0)
                    triangle_plane_cache[t] = to_ad(t->supporting_plane());
                Plane p = triangle_plane_cache[t];

                return plane_line_intersection<K>::compute(p, l);
            } else if (v.type == Tri_isector::VERTEX_INTER_FACE) {
                Face_RT f = v.v_inter_face.first;
                if (face_plane_cache.count(f) == 0) {
                    Plane pp = CGAL::radical_plane(f.first->vertex(f.second)->info().ad_point,
                                                   f.first->vertex(f.third)->info().ad_point);
                    face_plane_cache[f] = pp;
                }
                Plane p = face_plane_cache[f];


                Edge_T et = v.v_inter_face.second;
                if (triangle_edge_line_cache.count(et) == 0)
                    triangle_edge_line_cache[et] = to_ad(tri.segment(et).supporting_line());
                Line l = triangle_edge_line_cache[et];

                return plane_line_intersection<K>::compute(p, l);
            }

            // v.type == VERTEX_T
            return to_ad(tri.point(v.v_t));
        }

        // edge_to_segment
        Segment construct (T const& tri, RT const& rt, Pgon_edge const& e) {
            return Segment(construct(tri, rt, e.first()), construct(tri, rt, e.second()));
        }
    };

    // Incident facets to a power edge (= dual of a facet)
    // which belongs to a power face (= dual of an edge).
    template < typename RT >
    std::vector<typename RT::Edge>
    get_incident_facets (RT const& rt,
                         typename RT::Facet const& e,
                         typename RT::Edge const& f) {
        typedef typename RT::Vertex_handle Vertex_handle;
        typedef typename RT::Edge Edge;

        Vertex_handle x1 = f.first->vertex(f.second),
                      x2 = f.first->vertex(f.third);

        int i = e.second;
        int j1 = -1, j2 = -1, j3 = -1;
        for (int j = 0; j <= 3; ++j) {
            if (j == i)
                continue;

            Vertex_handle x = e.first->vertex(j);
            if (x == x1)
                j1 = j;
            else if (x == x2)
                j2 = j;
            else
                j3 = j;
        }

        assert(j1 >= 0 && j2 >= 0 && j3 >= 0);

        std::vector<Edge> res;
        res.emplace_back(e.first, j1, j3);
        res.emplace_back(e.first, j2, j3);

        return res;
    }

    // Find the power edges of the face f that intersect the triangulation.
    template < typename Cons, typename T, typename RT, typename F >
    DualTriangulation
    power_sphere_triangulation_intersection_raw (T const& tri, RT const& rt,
                                                 F out) {
        typedef typename CGAL::Kernel_traits<typename T::Point>::Kernel K;
        typedef Power_sphere_intersection_traits<K> Traits;

        typedef Tri_intersector<T, RT, Traits> Tri_isector;

        typedef typename Tri_isector::Edge_T Edge_T;
        typedef typename Tri_isector::Face_handle_T Face_handle_T;

        typedef typename Tri_isector::Vertex_handle_RT Vertex_handle_RT;
        typedef typename Tri_isector::Face_RT Face_RT;

        typedef typename Tri_isector::Pgon_vertex Pgon_vertex;
        typedef typename Tri_isector::Pgon_edge Pgon_edge;

        typedef typename Cons::Kernel_out::Segment_3 Segment;
        typedef typename Cons::Kernel_out::Point_3 Point;

        DualTriangulation dual_tri;
        if (tri.number_of_vertices() == 0 || rt.number_of_vertices() == 0)
            return dual_tri;

        // 1. Triangulation traversal
        typedef typename Tri_isector::Graph_T Graph_T;

#ifndef NDEBUG
        auto startT = std::chrono::system_clock::now();
#endif

        Graph_T G_t;
        std::stack<Edge_T> edgesT;
        std::set<Edge_T> visited_edgesT;

        Edge_T e0 = *tri.edges_cbegin();
        edgesT.push(e0);
        visited_edgesT.insert(e0);

        while (! edgesT.empty()) {
            Edge_T e = edgesT.top(); edgesT.pop();

            // Add intermediate vertices / edges
            Vertex_handle_RT u = rt.nearest_power_vertex(tri.point(e.source())),
                             v = rt.nearest_power_vertex(tri.point(e.target()));
            Vertex_handle_RT uprev = u;

            Pgon_vertex vprev = Tri_isector::make_vertex_t(e.source());

            while (u != v) {
                std::vector<Face_RT> edges_rt;
                rt.finite_incident_edges(u, std::back_inserter(edges_rt));
                for (const auto& f: edges_rt) {
                    Vertex_handle_RT unext = f.first->vertex(f.third);

                    // We don't want to go back
                    if (unext == uprev) {
                        continue;
                    }

                    /* auto poly = dual_facet(rt, f); */
                    if (do_power_face_segment_intersect<Traits>(rt, f, tri.segment(e))) {
                        typename Tri_isector::Vertex_inter_face vi(e, f);
                        Pgon_vertex vvi = vi.to_pgon_vertex();

                        G_t.vertices.push_back(vi);
                        G_t.edges.emplace_back(vprev, vvi, e.face(0), e.face(1), u);

                        vprev = vvi;

                        uprev = u;
                        u = unext;
                        break;
                    }
                }
            }

            // Last edge
            Pgon_vertex vlast = Tri_isector::make_vertex_t(e.target());

            G_t.edges.emplace_back(vprev, vlast, e.face(0), e.face(1), v);

            // Push non visited incident edges to e.source() and e.target()
            for (const auto& et: tri.incident_edges(e.source())) {
                if (! visited_edgesT.count(et)) {
                    edgesT.push(et);
                    visited_edgesT.insert(et);
                }
            }

            for (const auto& et: tri.incident_edges(e.target())) {
                if (! visited_edgesT.count(et)) {
                    edgesT.push(et);
                    visited_edgesT.insert(et);
                }
            }
        }

#ifndef NDEBUG
        auto endT = std::chrono::system_clock::now();
        std::chrono::duration<double> durationT = endT - startT;
        /* std::cout << "Graph_T traversal = "<< durationT.count() << std::endl; */
#endif

        // 2. Power diagram traversal
        typedef typename Tri_isector::Vertex_inter_edge Vertex_inter_edge;
        typename Tri_isector::Graph_P G_p;

#ifndef NDEBUG
        auto startP = std::chrono::system_clock::now();
#endif

        for (const auto& v0: G_t.vertices) {
            // v0.type == VERTEX_INTER_FACE
            if (G_p.insert_if_not_visited(v0))
                continue;

            Face_RT f = v0.face;
            Edge_T et = v0.edge;

            std::stack<std::pair<Vertex_inter_edge, Face_RT>> Q;
            Face_handle_T faces_t[2] = {et.faces.first, et.faces.second};
            for (int i = 0; i < 2 ; ++i) {
                Face_handle_T t0 = faces_t[i];
                if (t0 == nullptr) {
                    continue;
                }

                auto last = G_p.intersect(rt, t0, v0);
                auto vlast = last.second;

                if (! G_p.insert_if_not_visited(last.first, v0.to_pgon_vertex(), vlast, f)) {
                    continue;
                }

                if (last.second.type == Tri_isector::VERTEX_INTER_EDGE) {
                    Vertex_inter_edge w(last.second.v_inter_edge.second, last.second.v_inter_edge.first);
                    for (const auto& fn: get_incident_facets(rt, w.edge, f)) {
                        Q.emplace(w, fn);
                    }
                }
            }

            while (! Q.empty()) {
                auto v = Q.top(); Q.pop();
                Face_RT f = v.second;

                auto last = G_p.intersect(rt, f, v.first);
                auto vlast = last.second;

                if (! G_p.insert_if_not_visited(last.first, v.first.to_pgon_vertex(), vlast, f)) {
                    continue;
                }

                if (last.second.type == Tri_isector::VERTEX_INTER_EDGE) {
                    Vertex_inter_edge w(last.second.v_inter_edge.second, last.second.v_inter_edge.first);
                    for (const auto& fn: get_incident_facets(rt, w.edge, f)) {
                        Q.emplace(w, fn);
                    }
                }
            }
        }

#ifndef NDEBUG
        auto endP = std::chrono::system_clock::now();
        std::chrono::duration<double> durationP = endP - startP;
        /* std::cout << "Graph_P traversal = "<< durationP.count() << std::endl; */
#endif

        // 3. We associate to each (triangle, power cell), the list of its edges
#ifndef NDEBUG
        auto startA = std::chrono::system_clock::now();
#endif

        // (triangle t, vertex v) -> list of edges (e, is_boundary)
        std::map<std::pair<Face_handle_T, Vertex_handle_RT>,
                 std::vector<std::pair<Pgon_edge, bool>>> triangles_map;

        for (const auto& e: G_t.edges) {
            Pgon_edge ee = e.to_pgon_edge();
            bool is_boundary = e.t.first == nullptr || e.t.second == nullptr;
            triangles_map[std::make_pair(e.t.first, e.cell)].emplace_back(ee, is_boundary);
            triangles_map[std::make_pair(e.t.second, e.cell)].emplace_back(ee, is_boundary);
        }

        for (const auto& e: G_p.edges) {
            Pgon_edge ee = e.to_pgon_edge();
            triangles_map[std::make_pair(e.t, e.cells.first)].emplace_back(ee, true);
            triangles_map[std::make_pair(e.t, e.cells.second)].emplace_back(ee, true);
        }

        Cons cons;

#ifndef NDEBUG
        auto endA = std::chrono::system_clock::now();
        std::chrono::duration<double> durationA = endA - startA;

        /* std::cout << "Association time = " << durationA.count() << std::endl; */

        // Finally, we decompose each Laguerre cell into a collection of 3D polygons
        auto startLoop = std::chrono::system_clock::now();
#endif

        typedef CGAL::Polygon_3<typename Cons::Kernel_out> Polygon;

        // Indices: Point -> index
        std::map<Point, int, internal::Point3Compare<Point>> point_index_map;
        int cur_idx = 0;

        for (auto& kv: triangles_map) {
            if (kv.first.first == nullptr)
                continue;

            for (auto& e: kv.second) {
                Segment s = cons.construct(tri, rt, e.first);

                // Handle indices
                if (point_index_map.count(s.source()) == 0) {
                    point_index_map[s.source()] = cur_idx++;
                }
                if (point_index_map.count(s.target()) == 0) {
                    point_index_map[s.target()] = cur_idx++;
                }

                int idx_source = point_index_map[s.source()],
                    idx_target = point_index_map[s.target()];

                // Update indices in Pgon_vertex
                e.first.first().idx  = idx_source;
                e.first.second().idx = idx_target;
            }
        }

        for (const auto& kv: triangles_map) {
            if (kv.first.first == nullptr)
                continue;

            // Add, sort edges and (eventually) close the polygon
#ifndef NDEBUG
            auto startO = std::chrono::system_clock::now();
#endif

            Polygon poly;
            std::vector<int> boundary_edges;
            int i = 0;
            for (const auto& e: kv.second) {
                Segment s = cons.construct(tri, rt, e.first);

                poly.push_back(s, e.first.first().idx, e.first.second().idx);
                if (e.second) {
                    boundary_edges.push_back(i);
                }
                i++;
            }

            poly.order();

#ifndef NDEBUG
            auto endO = std::chrono::system_clock::now();
            std::chrono::duration<double> durationO = endO - startO;
            /* std::cout << "Order time = " << durationO.count() << std::endl; */
#endif

#ifndef NDEBUG
            auto startOut = std::chrono::system_clock::now();
#endif

            out(kv.first.first, kv.first.second, 0 /* comp */, poly, boundary_edges);

#ifndef NDEBUG
            auto endOut = std::chrono::system_clock::now();
            std::chrono::duration<double> durationOut = endOut - startOut;
            /* std::cout << "Out time = " << durationOut.count() << std::endl; */
#endif
        }

#ifndef NDEBUG
        auto endLoop = std::chrono::system_clock::now();
        std::chrono::duration<double> durationLoop = endLoop - startLoop;
        /* std::cout << "Loop time = " << durationLoop.count() << std::endl; */

        /* std::cout << std::endl; */
#endif

        // MTW case
        return G_p.build_dual();
    }

    // Without AD
    template < typename T, typename RT, typename F >
    DualTriangulation
    power_sphere_triangulation_intersection (T const& tri, RT const& rt, F out) {
        typedef MA::Constructions<T, RT, typename T::Kernel, false> Constructions;

        return power_sphere_triangulation_intersection_raw<Constructions>(tri, rt, out);
    }

    // With AD
    template < typename Kernel_ad, typename T, typename RT, typename F >
    DualTriangulation
    power_sphere_triangulation_intersection_ad (T const& tri, RT const& rt, F out) {
        typedef MA::Constructions<T, RT, Kernel_ad, true> Constructions;

        return power_sphere_triangulation_intersection_raw<Constructions>(tri, rt, out);
    }
} // namespace MA

#endif

