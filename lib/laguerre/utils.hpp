#ifndef _UTILS_HPP_
#define _UTILS_HPP_

#include <algorithm>
#include <boost/optional.hpp>
#include <cmath>
#include <Eigen/Dense>

// Haskell-ish alias for boost::optional
template < typename T >
using Maybe = boost::optional<T>;

// Sort 3 values in increasing order.
template < typename T >
void sort_3 (T& a, T& b, T& c) {
    if (a > b)
        std::swap(a, b);
    if (a > c)
        std::swap(a, c);
    if (b > c)
        std::swap(b, c);
}

namespace CGAL {
    /* template < typename Point > */
    /* bool is_close (Point const& p, Point const& q, const double epsilon = 1e-8) { */
    /*     return (p - q).squared_length() <= epsilon * epsilon; */
    /* } */

    // A 3D polygon.
    template < typename K_ >
    struct Polygon_3 {
        typedef K_ K;
        typedef typename K::FT FT;
        typedef typename CGAL::Plane_3<K> Plane;
        typedef typename CGAL::Point_3<K> Point;
        typedef typename CGAL::Vector_3<K> Vector;
        typedef typename CGAL::Segment_3<K> Segment;

        typedef std::pair<Point, int> Point_with_index;
        typedef std::pair<Point_with_index, Point_with_index> Segment_with_index;

        typedef typename std::vector<Segment_with_index> Edge_container;
        typedef typename Edge_container::const_iterator Edges_iterator;

        Edge_container edges_;
        std::vector<Point_with_index> vertices_;

        void push_back (Segment const& s, int is, int it) {
            edges_.emplace_back(std::make_pair(s.source(), is),
                                std::make_pair(s.target(), it));
        }

        void push_back (Segment_with_index const& s) {
            edges_.push_back(s);
        }

        void emplace_back (Point const& s, int is,
                           Point const& t, int it) {
            edges_.emplace_back(std::make_pair(s, is),
                                std::make_pair(t, it));
        }

        std::vector<Point> vertices() const {
            assert(vertices_.size() > 0);

            std::vector<Point> ret;
            std::transform(vertices_.begin(), vertices_.end(), std::back_inserter(ret),
                           [] (Point_with_index const& p) {
                               return p.first;
                           });
            return ret;
        }

        std::vector<Segment> edges() const {
            std::vector<Segment> ret;
            std::transform(edges_.begin(), edges_.end(), std::back_inserter(ret),
                           [] (Segment_with_index const& s) {
                               return Segment(s.first.first, s.second.first);
                           });
            return ret;
        }

        void order () {
            if (vertices_.size() > 0)
                return;

            Segment_with_index scur = *edges_.begin();
            Point_with_index p0 = scur.first;
            Point_with_index cur = p0;

            vertices_.push_back(cur);

            bool toClose = true;
            while (1) {
                bool found = false;
                for (const auto& s: edges_) {
                    if (s.first.second == scur.first.second &&
                        s.second.second == scur.second.second)
                    /* if (is_close(s.first.first, scur.first.first) && */
                    /*     is_close(s.second.first, scur.second.first)) */
                        continue;

                    if (s.first.second == cur.second) {
                    /* if (is_close(s.first.first, cur.first)) { */
                        scur = s;
                        cur = s.second;
                        vertices_.push_back(cur);
                        found = true;
                        break;
                    } else if (s.second.second == cur.second) {
                    /* } else if (is_close(s.second.first, cur.first)) { */
                        scur = s;
                        cur = s.first;
                        vertices_.push_back(cur);
                        found = true;
                        break;
                    }
                }

                if (!found) {
                    /* assert(false); */
                    break;
                }


                if (cur.second == p0.second) {
                /* if (is_close(cur.first, p0.first)) { */
                    toClose = false;
                    break;
                }
            }

            // Eventually close the polygon
            if (toClose) {
                /* assert(false); */
                vertices_.push_back(p0);
            }
        }

        // Area (not signed) of the polygon.
        // Polygon should be oriented and closed.
        FT area () const {
            assert(vertices_.size() >= 1);

            FT res = 0;

            for (int i = 1; i < int(vertices_.size()) - 1; ++i) {
                Vector const& pi  = vertices_[i].first - vertices_[0].first,
                              ppi = vertices_[i + 1].first - vertices_[0].first;
                FT len = CGAL::cross_product(pi, ppi).squared_length();
                if (len != 0) {
                    res += sqrt(len);
                }
            }

            return res / 2;
        }

        Vector normal () const {
            assert(vertices_.size() >= 4);

            Plane p(vertices_[0].first,
                    vertices_[1].first,
                    vertices_[2].first);
            Vector n = p.orthogonal_vector();
            n = n / CGAL::sqrt(n.squared_length());
            return n;
        }

        // Centroid of the polygon.
        // Polygon should be oriented and closed.
        Point centroid () const {
            assert(vertices_.size() >= 1);

            FT A = 0;
            Vector c(0, 0, 0);
            for (int i = 1; i < int(vertices_.size()) - 1; ++i) {
                Vector const& pi  = vertices_[i].first - vertices_[0].first,
                              ppi = vertices_[i + 1].first - vertices_[0].first;
                FT len = CGAL::cross_product(pi, ppi).squared_length();
                if (len != 0) {
                    FT curarea = sqrt(len);
                    c = c + curarea * ((vertices_[0].first - CGAL::ORIGIN) + (vertices_[i].first - CGAL::ORIGIN) +
                                       (vertices_[i + 1].first - CGAL::ORIGIN)) / 3;
                    A += curarea;
                }
            }
            c = c / A;

            return CGAL::ORIGIN + c;
        }
    };

    // Centroid of a collection of 3D polygons
    template < typename K >
    typename CGAL::Polygon_3<K>::Point
    centroid (std::vector<CGAL::Polygon_3<K>> const& polygons) {
        typedef typename CGAL::Polygon_3<K> Polygon;
        typedef typename K::FT FT;
        typedef typename Polygon::Vector Vector;

        Vector res(0, 0 , 0);
        FT area = 0;
        for (const auto& p: polygons) {
            FT curarea = p.area();
            if (curarea != 0) {
                area += curarea;
                res = res + curarea * (p.centroid() - CGAL::ORIGIN);
            }
        }

        if (area != 0) {
            res = res / area;
        }

        /* return project_on_sphere(CGAL::ORIGIN + res, tri.radius()); */
        return CGAL::ORIGIN + res;
    }
} // namespace CGAL

#endif

