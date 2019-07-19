#ifndef _SPHERE_TRIANGULATION_2_H_
#define _SPHERE_TRIANGULATION_2_H_

#include <algorithm>
#include <array>
#include <boost/optional.hpp>
#include <cassert>
#include <fstream>
#include <iterator>
#include <map>
#include <unordered_set>
#include <vector>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_triangle_primitive.h>
#include <CGAL/Origin.h>
#include <CGAL/squared_distance_3.h>

// For loading the triangulation from an OFF file
#include <CGAL/Polyhedron_3.h>
#include <CGAL/IO/Polyhedron_iostream.h>

#include "unordered_pair.hpp"

// Forward declarations
template < typename K >
class Triangle_with_neighbours;

template < typename K >
struct CEdge_with_info;

template < typename K >
class Surface_triangulation_2;

// An edge has two endpoints and belongs to two triangles.
template < typename K >
struct CEdge_with_info {
    typedef Triangle_with_neighbours<K> Face;
    typedef Face* Face_handle;
    typedef CGAL::Point_3<K> Point;
    typedef CGAL::Segment_3<K> Segment;

    std::unordered_pair<int> endpoints;
    std::unordered_pair<Face_handle> faces;

    CEdge_with_info () = default;

    CEdge_with_info (int x, int y,
                     Face_handle const& t1, Face_handle const& t2) : endpoints(x, y), faces(t1, t2){
    }

    int source () const {
        return endpoints.first;
    }

    int target () const {
        return endpoints.second;
    }

    Face_handle face (int i) const {
        if (i % 2 == 0)
            return faces.first;

        // i % 2 == 1
        return faces.second;
    }

    bool operator== (CEdge_with_info<K> const& other) const {
        return endpoints == other.endpoints;
    }

    bool operator!= (CEdge_with_info<K> const& other) const {
        return endpoints != other.endpoints;
    }

    bool operator< (CEdge_with_info<K> const& other) const {
        return endpoints < other.endpoints;
    }
};

// A triangle with its 3 neighbours.
template < typename K >
class Triangle_with_neighbours : public CGAL::Triangle_3<K> {
    public:
        typedef K Kernel;
        typedef CGAL::Triangle_3<Kernel> Base;
        typedef CGAL::Point_3<Kernel> Point;
        typedef CGAL::Vector_3<Kernel> Vector;
        typedef CGAL::Segment_3<Kernel> Segment;
        typedef CGAL::Plane_3<Kernel> Plane;
        typedef CGAL::Ray_3<Kernel> Ray;

        typedef Triangle_with_neighbours<K>* Face_handle;
        typedef CEdge_with_info<K> Edge;

        using Base::vertex;

        // Indices of the three vertices of the triangle
        int indices[3] = {-1};
        // Index of the triangle
        int tid = -1;

    private:
        typedef std::unordered_pair<Point> Point_pair;

    public:
        Triangle_with_neighbours () = default;

        Triangle_with_neighbours (Point const& p, Point const& q, Point const& r) : CGAL::Triangle_3<K>(p, q, r) {
            neighbours_ = {nullptr};
        }

        Edge edge (int i) {
            assert(0 <= i && i <= 2);

            Point p1 = vertex(i % 3), p2 = vertex((i + 1) % 3);
            Point_pair pp(p1, p2);
            int idx_neighbour = neighbour_has_edge(pp);
            Face_handle t1 = this, t2 = (idx_neighbour == -1) ? nullptr : neighbour(idx_neighbour);

            return Edge(indices[i % 3], indices[(i + 1) % 3], t1, t2);
        }

        bool has_vertex (Point const& p) const {
            return vertex(0) == p || vertex(1) == p || vertex(2) == p;
        }

        Face_handle neighbour (int i) const {
            assert(0 <= i && i <= 2);

            return neighbours_[i];
        }

        Face_handle other (Edge const& e) const {
            if (e.faces.first == this)
                return e.faces.second;

            assert(e.faces.second == this);
            return e.faces.first;
        }

        void set_neighbour (int i, Face_handle t) {
            assert(0 <= i && i <= 2);
            neighbours_[i] = t;
        }

        void set_neighbours (Face_handle t0, Face_handle t1, Face_handle t2) {
            set_neighbour(0, t0);
            set_neighbour(1, t1);
            set_neighbour(2, t2);
        }

        // source: http://math.stackexchange.com/questions/544946/determine-if-projection-of-3d-point-onto-plane-is-within-a-triangle
        bool is_inside (Point const& p) const {
            // Barycentric coordinates
            Vector u = vertex(1) - vertex(0),
                   v = vertex(2) - vertex(0),
                   n = CGAL::cross_product(u, v),
                   w = p - vertex(0);

            double gamma = CGAL::cross_product(u, w) * n / n.squared_length(),
                   beta = CGAL::cross_product(w, v) * n / n.squared_length(),
                   alpha = 1 - gamma - beta;

            return (0 <= alpha && alpha <= 1 && 0 <= beta && beta <= 1 && 0 <= gamma && gamma <= 1);
        }

        // Unit outward normal
        Vector normal () const {
            Vector n = CGAL::cross_product(vertex(1) - vertex(0),
                                           vertex(2) - vertex(0));
            n = n / CGAL::sqrt(n.squared_length());

            return n;
        }

        void compute_normal () {
            n_ = normal();
        }

        // Should not be used
        bool operator== (Triangle_with_neighbours const& other) = delete;

    private:
        std::array<Triangle_with_neighbours*, 3> neighbours_;
        Vector n_; // Unit outward normal

        // Utility functions
        bool has_edge (Point_pair const& e) const {
            Point p0 = vertex(0), p1 = vertex(1), p2 = vertex(2);
            std::array<Point_pair, 3> edges;
            edges[0] = std::make_unordered_pair(p0, p1);
            edges[1] = std::make_unordered_pair(p1, p2);
            edges[2] = std::make_unordered_pair(p2, p0);

            return e == edges[0] || e == edges[1] || e == edges[2];
        }

        // Find the neighbour triangle with a given edge
        int neighbour_has_edge (Point_pair const& e) const {
            if (neighbours_[0] != nullptr && neighbours_[0]->has_edge(e))
                return 0;
            else if (neighbours_[1] != nullptr && neighbours_[1]->has_edge(e))
                return 1;
            else if (neighbours_[2] != nullptr && neighbours_[2]->has_edge(e))
                return 2;
            else
                return -1;
        }
};

// Custom primitive for AABB tree intersection.
// Iterator::value_type = Triangle_with_neighbours
template < typename K, typename Iterator >
struct Triangle_with_neigbours_primitive {
    public:
        typedef typename K::Point_3 Point;
        typedef Triangle_with_neighbours<K>* Id;
        typedef typename K::Triangle_3 Datum;

    private:
        Id t_;

    public:
        Triangle_with_neigbours_primitive () {}

        Triangle_with_neigbours_primitive (Iterator it) :
            t_(&(*it)) {
        }

        const Id& id () const {
            return t_;
        }

        Datum datum () const {
            return Datum(t_->vertex(0), t_->vertex(1), t_->vertex(2));
        }

        Point reference_point () const {
            return t_->vertex(0);
        }
};

// Triangulation of the 3D unit sphere.
template < typename K >
class Surface_triangulation_2 {
    public:
        typedef K Kernel;
        typedef typename Kernel::FT FT;
        typedef CGAL::Point_3<Kernel> Point;
        typedef CGAL::Segment_3<Kernel> Segment;
        typedef CGAL::Plane_3<Kernel> Plane;
        typedef CGAL::Ray_3<Kernel> Ray;
        typedef CGAL::Line_3<Kernel> Line;
        typedef CGAL::Vector_3<Kernel> Vector;

        typedef Point Vertex;
        typedef Vertex* Vertex_handle;

        typedef CEdge_with_info<K> Edge;

        typedef Triangle_with_neighbours<Kernel> Face;
        typedef Face Triangle;
        typedef Face* Face_handle;

        typedef typename std::vector<Point>::const_iterator Vertices_const_iterator;
        typedef typename std::vector<Edge>::const_iterator Edges_const_iterator;
        typedef typename std::vector<Triangle>::iterator Triangles_iterator;
        typedef typename std::vector<Triangle>::const_iterator Triangles_const_iterator;

        typedef Triangle_with_neigbours_primitive<K, Triangles_iterator> Primitive;
        typedef typename CGAL::AABB_traits<K, Primitive> AABB_traits;
        typedef typename CGAL::AABB_tree<AABB_traits> Tree;

    private:
        typedef std::pair<boost::variant<Point, Segment>, Face_handle> Intersection_return_type;

    public:
        Surface_triangulation_2 (std::string const& filename) {
            // Construct the triangulation by loading an OFF file
            from_off(filename);

            // Initialize data structures
            init_structures();
        }

        Surface_triangulation_2 (std::vector<Point> const& points,
                                std::vector<std::tuple<int, int, int>> const& triangles) {
            // The triangulation is given by the list of its vertices and the indices
            // corresponding to the triangles
            for (const auto& ind: triangles) {
                push_back(Triangle(points[std::get<0>(ind)],
                                   points[std::get<1>(ind)],
                                   points[std::get<2>(ind)]));
            }

            // Initialize data structures
            init_structures();
        }

        void init_structures () {
            // Compute adjacency relation
            compute_adjacencies();
            // Compute the edges
            add_edges();

            // Normals
            /* for (auto& t: triangles_) */
            /*     t.compute_normal(); */

            // Construct AABB tree
            tree_.insert(triangles_.begin(), triangles_.end());
        }

        void clear () {
            // Adjacencies
            for (auto& t : triangles_) {
                t.set_neighbour(nullptr, nullptr, nullptr);
            }

            edges_.clear();
            tree_.clear();
        }

        Vertices_const_iterator vertices_cbegin () const {
            return vertices_.cbegin();
        }

        Vertices_const_iterator vertices_cend () const {
            return vertices_.cend();
        }

        Edges_const_iterator edges_cbegin () const {
            return edges_.cbegin();
        }

        Edges_const_iterator edges_cend () const {
            return edges_.cend();
        }

        Triangles_iterator triangles_begin () {
            return triangles_.begin();
        }

        Triangles_iterator triangles_end () {
            return triangles_.end();
        }

        Triangles_const_iterator triangles_cbegin () const {
            return triangles_.cbegin();
        }

        Triangles_const_iterator triangles_cend () const {
            return triangles_.cend();
        }

        std::size_t number_of_vertices () const {
            return vertices_.size();
        }

        std::size_t number_of_triangles () const {
            return triangles_.size();
        }

        std::size_t number_of_edges () const {
            return edges_.size();
        }

        // Insertion of a triangle
        void push_back (Triangle t) {
            for (int i = 0; i < 3; ++i) {
                auto resfind = std::find(vertices_.begin(), vertices_.end(), t.vertex(i));
                if (resfind == vertices_.end()) {
                    vertices_index_[t.vertex(i)] = index_vertices_;
                    vertices_.push_back(t.vertex(i));
                    t.indices[i] = index_vertices_;
                    index_vertices_++;
                } else {
                    t.indices[i] = vertices_index_[t.vertex(i)];
                }
            }

            t.tid = index_triangles_;
            index_triangles_++;
            triangles_.push_back(t);
        }

        // Insertion of triangles
        template < typename TriangleIterator >
        void insert (TriangleIterator tbegin, TriangleIterator tbeyond) {
            typedef typename std::iterator_traits<TriangleIterator>::value_type IteratorValueType;
            static_assert(std::is_same<IteratorValueType, Triangle>::value,
                          "insert: value_type must be Triangle");

            for (TriangleIterator tit = tbegin; tit != tbeyond; ++tit)
                push_back(*tit);
        }

        // Conversion
        int index (Point const& p) const {
            return vertices_index_.at(p);
        }

        Point point (int i) const {
            return vertices_[i];
        }

        Segment segment (Edge const& e) const {
            return Segment(vertices_[e.source()],
                           vertices_[e.target()]);
        }

        Triangle triangle (int i) const {
            assert(0 <= i && i < triangles_.size());
            return triangles_[i];
        }

        // Total area
        FT total_area () const {
            FT res = 0;

            for (const auto& t: triangles_)
                res += CGAL::sqrt(t.squared_area());

            return res;
        }

        // Total edge length
        FT total_edge_length () const {
            FT res = 0;

            for (const auto& e: edges_)
                res += CGAL::sqrt(segment(e).squared_length());

            return res;
        }

        // Test if the triangulation is valid: each triangle must have 3 neighbours
        bool is_valid () const {
            for (const auto& t : triangles_) {
                if (t.neighbour(0) == nullptr || t.neighbour(1) == nullptr || t.neighbour(2) == nullptr)
                    return false;
            }

            return true;
        }

        // Points on the boundary
        std::vector<Point> boundary_points () const {
            std::vector<Point> res;
            std::set<int> visited_points;

            for (auto eit = edges_cbegin(); eit != edges_cend(); ++eit) {
                if (eit->face(0) == nullptr || eit->face(1) == nullptr) {
                    if (visited_points.count(eit->source() == 0)) {
                        visited_points.insert(eit->source());
                        res.push_back(point(eit->source()));
                    }

                    if (visited_points.count(eit->target()) == 0) {
                        visited_points.insert(eit->target());
                        res.push_back(point(eit->target()));
                    }
                }
            }

            return res;
        }

        // Triangles incident to a given vertex.
        std::vector<Face> incident_triangles (int v) const {
            std::vector<Face> tris;

            for (const auto& t : triangles_) {
                if (t.indices[0] == v || t.indices[1] == v || t.indices[2] == v) {
                    tris.push_back(t);
                }
            }

            return tris;
        }

        // Edges incident to a given vertex.
        std::vector<Edge> incident_edges (int v) const {
            std::vector<Edge> edges;

            for (auto eit = edges_cbegin(); eit != edges_cend(); ++eit) {
                if (eit->source() == v || eit->target() == v) {
                    edges.push_back(*eit);
                }
            }

            return edges;
        }

        // Test if a point is on the triangulation
        // TODO: change name -> has_on
        bool is_inside (Point const& p) const {
            for (const auto& t : triangles_) {
                if (t.is_inside(p))
                    return true;
            }

            return false;
        }

        bool has_on (Point const& p) const {
            return is_inside(p);
        }

        // Convert to an OFF file.
        void to_off (std::string const& path) const {
            std::ofstream os(path.c_str());
            // Header
            os << "OFF\n" << vertices_.size() << " " << triangles_.size() << " 0" << std::endl;

            // Vertices
            for (const auto& v : vertices_)
                os << v << std::endl;

            // Faces
            for (const auto& t : triangles_) {
                os << "3 ";
                os << vertices_index_.at(t.vertex(0)) << " ";
                os << vertices_index_.at(t.vertex(1)) << " ";
                os << vertices_index_.at(t.vertex(2)) << std::endl;
            }
        }

        FT squared_distance (Point const& p) const {
            return tree_.squared_distance(p);
        }

    private:
        Point origin_ = CGAL::ORIGIN;

        std::vector<Triangle> triangles_;
        int index_triangles_ = 0;

        std::vector<Point> vertices_;
        std::map<Point, int> vertices_index_;

        std::vector<Edge> edges_;

        Tree tree_;

        int index_vertices_ = 0;

        // Compute the adjacencies relations
        void compute_adjacencies () {
            for (auto& t : triangles_) {
                const Point& p0 = t.vertex(0), p1 = t.vertex(1), p2 = t.vertex(2);

                auto t1 = std::find_if(triangles_.begin(), triangles_.end(), [&] (const Triangle& tt) {
                              return tt != t && tt.has_vertex(p0) && tt.has_vertex(p1);
                          });

                auto t2 = std::find_if(triangles_.begin(), triangles_.end(), [&] (const Triangle& tt) {
                              return tt != t && tt.has_vertex(p1) && tt.has_vertex(p2);
                          });

                auto t3 = std::find_if(triangles_.begin(), triangles_.end(), [&] (const Triangle& tt) {
                              return tt != t && tt.has_vertex(p2) && tt.has_vertex(p0);
                          });

                Face_handle tt1 = (t1 == triangles_.end()) ? nullptr : &*t1,
                            tt2 = (t2 == triangles_.end()) ? nullptr : &*t2,
                            tt3 = (t3 == triangles_.end()) ? nullptr : &*t3;

                t.set_neighbours(tt1, tt2, tt3);
            }
        }

        // Compute the edges of the triangulation
        void add_edges () {
            std::set<std::unordered_pair<int>> visited_edges;

            for (auto& t: triangles_) {
                for (int i = 0; i <= 2; ++i) {
                    Edge e = t.edge(i);

                    if (! visited_edges.count(e.endpoints)) {
                        edges_.push_back(e);
                        visited_edges.insert(e.endpoints);
                    }
                }
            }
        }

        // From an OFF file
        void from_off (std::string const& filename) {
            typedef typename CGAL::Polyhedron_3<K> Polyhedron_3;
            Polyhedron_3 P;

            std::ifstream file(filename);
            if (! file.good())
                assert(false);
            file >> P;

            for (typename Polyhedron_3::Facet_iterator fit = P.facets_begin();
                 fit != P.facets_end();
                 ++fit) {
                assert(fit->is_triangle());

                typename Polyhedron_3::Halfedge_handle h = fit->halfedge();

                push_back(Triangle(h->vertex()->point(),
                                   h->next()->vertex()->point(),
                                   h->next()->next()->vertex()->point()));
            }
        }
};

#endif

