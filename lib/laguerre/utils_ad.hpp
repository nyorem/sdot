#ifndef _UTILS_AD_HPP_
#define _UTILS_AD_HPP_

// extracted and modified from Regular_triangulation_3.h
namespace CGAL {
    template < typename Point_ad, typename RT >
    Point_ad
    dual (RT const& rt, typename RT::Cell_handle const& c) {
        typedef typename CGAL::Kernel_traits<Point_ad>::Kernel K_ad;
        typename Kernel_ad::Construct_weighted_circumcenter_3 Construct_weighted_circumcenter_3;

      CGAL_triangulation_precondition(rt.dimension()==3);
      CGAL_triangulation_precondition( ! rt.is_infinite(c) );

      return Construct_weighted_circumcenter_3(c->vertex(0)->info().ad_point,
                                               c->vertex(1)->info().ad_point,
                                               c->vertex(2)->info().ad_point,
                                               c->vertex(3)->info().ad_point);
    }

    template < typename Weighted_point_ad, typename RT >
    CGAL::Object
    dual (RT const& rt, typename RT::Facet const& f) {
        typename RT::Cell_handle c = f.first;
        int i = f.second;

        typedef typename CGAL::Kernel_traits<Weighted_point_ad>::Kernel K_ad;
        typedef typename K_ad::Point_3 Point_ad;
        typedef typename K_ad::Segment_3 Segment_ad;
        typedef typename K_ad::Ray_3 Ray_ad;
        typedef typename K_ad::Plane_3 Plane_ad;
        typedef typename K_ad::Line_3 Line_ad;

        typename Kernel_ad::Construct_weighted_circumcenter_3 Construct_weighted_circumcenter_3;

        if ( rt.dimension() == 2 ) {
          CGAL_triangulation_precondition( i == 3 );
          return make_object(Construct_weighted_circumcenter_3(c->vertex(0)->info().ad_point,
                                                               c->vertex(1)->info().ad_point,
                                                               c->vertex(2)->info().ad_point) );
        }

        // dimension() == 3
        typename RT::Cell_handle n = c->neighbor(i);
        if ( ! rt.is_infinite(c) && ! rt.is_infinite(n) )
          return make_object(Segment_ad( dual<Point_ad>(rt, c), dual<Point_ad>(rt, n) ));

        // either n or c is infinite
        int in;
        if ( rt.is_infinite(c) )
          in = n->index(c);
        else {
          n = c;
          in = i;
        }

        // n now denotes a finite cell, either c or c->neighbor(i)
        int ind[3] = {(in+1)&3,(in+2)&3,(in+3)&3};
        if ( (in&1) == 1 )
          std::swap(ind[0], ind[1]);
        const Weighted_point_ad& p = n->vertex(ind[0])->info().ad_point;
        const Weighted_point_ad& q = n->vertex(ind[1])->info().ad_point;
        const Weighted_point_ad& r = n->vertex(ind[2])->info().ad_point;

        Line_ad l = Plane_ad(p,q,r).perpendicular_line( Construct_weighted_circumcenter_3(p,q,r) );
        return make_object(Ray_ad( dual<Point_ad>(rt, n), l));
    }
} // namespace CGAL

#endif

