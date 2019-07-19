#ifndef _TYPES_HPP_
#define _TYPES_HPP_

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Filtered_kernel.h>
#include "surface_triangulation_2.hpp"
#include <CGAL/Regular_triangulation_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include "AD.hpp"
#include "CGAL_AD.hpp"

// Kernel
typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef Kernel::FT FT;

/* typedef CGAL::Simple_cartesian<AD> Kernel_ad; */
typedef CGAL::Filtered_kernel<CGAL::Simple_cartesian<AD> > Kernel_ad;
typedef Kernel_ad::FT FT_ad;

// Geometry
// Normal
typedef CGAL::Point_3<Kernel> Point;
typedef CGAL::Plane_3<Kernel> Plane;
typedef CGAL::Vector_3<Kernel> Vector;
typedef CGAL::Segment_3<Kernel> Segment;

// AD
typedef CGAL::Point_3<Kernel_ad> Point_ad;
typedef CGAL::Plane_3<Kernel_ad> Plane_ad;
typedef CGAL::Segment_3<Kernel_ad> Segment_ad;
typedef Kernel_ad::Weighted_point_3 Weighted_point_ad;

// Triangulated surface
typedef Surface_triangulation_2<Kernel> Surface_triangulation;
typedef Surface_triangulation::Triangle Triangle;
typedef Surface_triangulation::Face_handle Face_handle_T;

// Regular triangulation
struct Info {
    int index = -1;
    Weighted_point_ad ad_point;

    Info () = default;

    Info (int index) : index(index) {
    }

    Info (int index, Weighted_point_ad const& ad_point) : index(index), ad_point(ad_point) {
    }
};

typedef CGAL::Weighted_point_3<Kernel> Weighted_point;

typedef CGAL::Regular_triangulation_vertex_base_3<Kernel> Vbase;
typedef CGAL::Triangulation_vertex_base_with_info_3<Info, Kernel, Vbase> Vb;
typedef CGAL::Regular_triangulation_cell_base_3<Kernel> Cb;
typedef CGAL::Triangulation_data_structure_3<Vb, Cb> Tds;
typedef CGAL::Regular_triangulation_3<Kernel, Tds> RT;

#endif

