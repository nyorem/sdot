
// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Modified by Q. Merigot: removed expression templates for
// compatibility with CGAL

#ifndef AUTODIFF_SCALAR_H
#define AUTODIFF_SCALAR_H
#include <Eigen/Sparse>

/* #define PRINT_OP */

class AD
{
 public:
  typedef double Scalar;
  typedef Eigen::SparseVector<double> Vector;

 private:
  Scalar m_value;
  Vector m_derivatives;

 public:
  /** Default constructor without any initialization. */
  AD() {}

  /** Constructs an active scalar from its \a value,
      and initializes the \a nbDer derivatives such that it corresponds to the \a derNumber -th variable */
 AD(const Scalar& value, int nbDer, int derNumber)
   : m_value(value), m_derivatives(Vector(nbDer))
    {
      m_derivatives.coeffRef(derNumber) = Scalar(1);
    }

  /** Conversion from a scalar constant to an active scalar.
   * The derivatives are set to zero. */
  /*explicit*/
 AD(const Scalar& value)
    : m_value(value)
  {
    if(m_derivatives.size()>0)
      m_derivatives.setZero();
  }

  AD(const Scalar& value, int nbDer)
    : m_value(value), m_derivatives(Vector(nbDer))
  {
  }

  /** Constructs an active scalar from its \a value and derivatives \a der */
 AD(const Scalar& value, const Vector& der)
   : m_value(value), m_derivatives(der)
  {}

  friend inline
    std::ostream & operator << (std::ostream & s, const AD& a)
  {
    return s << a.value();
  }

 AD(const AD& other)
   : m_value(other.value()), m_derivatives(other.derivatives())
    {}

  inline AD& operator=(const AD& other)
    {
      m_value = other.value();
      m_derivatives = other.derivatives();
      return *this;
    }

  inline const Scalar& value() const { return m_value; }
  inline Scalar& value() { return m_value; }

  inline const Vector& derivatives() const { return m_derivatives; }
  inline Vector& derivatives() { return m_derivatives; }

  inline bool operator< (const Scalar& other) const
  { return m_value <  other; }
  inline bool operator< (const AD& other) const
  { return m_value <  other.value(); }
  inline bool operator<=(const Scalar& other) const
  { return m_value <= other; }
  inline bool operator<= (const AD& other) const
  { return m_value <= other.value(); }
  inline bool operator> (const Scalar& other) const
  { return m_value >  other; }
  inline bool operator> (const AD& other) const
  { return m_value >  other.value(); }
  inline bool operator>=(const Scalar& other) const
  { return m_value >= other; }
  inline bool operator>= (const AD& other) const
  { return m_value >= other.value(); }
  inline bool operator==(const Scalar& other) const
  { return m_value == other; }
  inline bool operator==(const AD& other) const
  { return m_value == other; }
  inline bool operator!=(const Scalar& other) const
  { return m_value != other; }
  inline bool operator!=(const AD& other) const
  { return m_value != other; }

  friend inline bool operator< (const Scalar& a, const AD& b)
  { return a <  b.value(); }
  friend inline bool operator<=(const Scalar& a, const AD& b)
  { return a <= b.value(); }
  friend inline bool operator> (const Scalar& a, const AD& b)
  { return a >  b.value(); }
  friend inline bool operator>=(const Scalar& a, const AD& b)
  { return a >= b.value(); }
  friend inline bool operator==(const Scalar& a, const AD& b)
  { return a == b.value(); }
  friend inline bool operator!=(const Scalar& a, const AD& b)
  { return a != b.value(); }

  inline AD operator+(const Scalar& other) const
  {
#ifdef PRINT_OP
      std::cout << "AD operator+(Scalar)" << std::endl;
#endif
      return AD(m_value + other, m_derivatives);
  }

  friend inline AD operator+(const Scalar& a, const AD& b)
  {
#ifdef PRINT_OP
      std::cout << "AD operator+(Scalar, AD)" << std::endl;
#endif
      return AD(a + b.value(), b.derivatives());
  }

  inline AD& operator+=(const Scalar& other)
  {
#ifdef PRINT_OP
      std::cout << "AD operator+=(Scalar)" << std::endl;
#endif
      value() += other;
      return *this;
  }

  AD operator+(const AD& other) const
  {
#ifdef PRINT_OP
      std::cout << "AD operator+(AD)" << std::endl;
#endif
      if (m_derivatives.size() == 0) {
          return Scalar(value()) + other;
      }

      if (other.derivatives().size() == 0) {
          return *this + Scalar(other.value());
      }

      return AD(m_value + other.value(), m_derivatives + other.derivatives());
  }

  AD& operator+=(const AD& other)
  {
#ifdef PRINT_OP
      std::cout << "AD operator+=(AD)" << std::endl;
#endif
      (*this) = (*this) + other;
      return *this;
  }

  AD operator-(const Scalar& b) const
  {
#ifdef PRINT_OP
      std::cout << "AD operator-(Scalar)" << std::endl;
#endif
      return AD(m_value - b, m_derivatives);
  }

  friend inline AD operator-(const Scalar& a, const AD& b)
  {
#ifdef PRINT_OP
      std::cout << "AD operator-(Scalar, AD)" << std::endl;
#endif
      return AD(a - b.value(), -b.derivatives());
  }

  AD& operator-=(const Scalar& other)
  {
#ifdef PRINT_OP
      std::cout << "AD operator-=(AD)" << std::endl;
#endif
      value() -= other;
      return *this;
  }

  AD operator-(const AD& other) const
  {
#ifdef PRINT_OP
      std::cout << "AD operator-(AD)" << std::endl;
#endif
      if (m_derivatives.size() == 0) {
          return Scalar(value()) - other;
      }

      if (other.derivatives().size() == 0) {
        return *this - Scalar(other.value());
      }

      return AD(m_value - other.value(),
                m_derivatives - other.derivatives());
  }

  AD& operator-=(const AD& other)
  {
#ifdef PRINT_OP
      std::cout << "AD operator-=(AD)" << std::endl;
#endif
      *this = *this - other;
      return *this;
  }

  AD operator-() const
  {
#ifdef PRINT_OP
      std::cout << "AD operator-()" << std::endl;
#endif
      return AD(-m_value, -m_derivatives);
  }

  AD operator*(const Scalar& other) const
  {
#ifdef PRINT_OP
      std::cout << "AD operator*(Scalar)" << std::endl;
#endif
      return AD(m_value * other, (m_derivatives * other));
  }

  friend inline AD operator*(const Scalar& other, const AD& a)
  {
#ifdef PRINT_OP
      std::cout << "AD operator*(Scalar, AD)" << std::endl;
#endif
      return AD(a.value() * other,
                a.derivatives() * other);
  }

  AD operator/(const Scalar& other) const
  {
      assert(other != 0);

#ifdef PRINT_OP
      std::cout << "AD operator/(Scalar)" << std::endl;
#endif
      return AD(m_value / other, (m_derivatives * (Scalar(1)/other)));
  }

  friend inline AD operator/(const Scalar& other, const AD& a)
  {
      assert(a.value() != 0);

#ifdef PRINT_OP
      std::cout << "AD operator/(Scalar, AD)" << std::endl;
#endif
      return AD(other / a.value(),
                a.derivatives() *
                (Scalar(-other) / (a.value()*a.value())));
  }

  AD operator/(const AD& other) const
  {
      assert(other.value() != 0);

#ifdef PRINT_OP
      std::cout << "AD operator/(AD)" << std::endl;
#endif

      if (m_derivatives.size() == 0) {
          return Scalar(value()) / other;
      }

      if (other.derivatives().size() == 0) {
          return *this / Scalar(other.value());
      }

      return AD(m_value / other.value(),
                ((m_derivatives * other.value()) -
                 (m_value * other.derivatives())) *
                (Scalar(1)/(other.value()*other.value())));
  }

  AD operator*(const AD& other) const
  {
#ifdef PRINT_OP
      std::cout << "AD operator*(AD)" << std::endl;
#endif

      if (m_derivatives.size() == 0) {
          return Scalar(value()) * other;
      }

      if (other.derivatives().size() == 0) {
          return *this * Scalar(other.value());
      }

      return AD(m_value * other.value(),
                (m_derivatives * other.value()) +
                (m_value * other.derivatives()));
  }

  inline AD& operator*=(const Scalar& other)
  {
#ifdef PRINT_OP
      std::cout << "AD operator*=(Scalar)" << std::endl;
#endif
      *this = *this * other;
      return *this;
  }

  inline AD& operator*=(const AD& other)
  {
#ifdef PRINT_OP
      std::cout << "AD operator*=(AD)" << std::endl;
#endif
      *this = *this * other;
      return *this;
  }

  inline AD& operator/=(const Scalar& other)
  {
#ifdef PRINT_OP
      std::cout << "AD operator/=(Scalar)" << std::endl;
#endif
      *this = *this / other;
      return *this;
  }

  inline AD& operator/=(const AD& other)
  {
#ifdef PRINT_OP
      std::cout << "AD operator/=(AD)" << std::endl;
#endif
      *this = *this / other;
      return *this;
  }
};

inline AD sqrt(const AD &x) {
#ifdef PRINT_OP
    std::cout << "AD sqrt(AD)" << std::endl;
#endif

    using std::sqrt;
    double sqrtx = sqrt(x.value());

    assert(sqrtx != 0);

    return AD(sqrtx,x.derivatives() * (double(0.5) / sqrtx));
}

inline AD atan2(const AD &y, const AD& x) {
#ifdef PRINT_OP
    std::cout << "AD atan2(AD, AD)" << std::endl;
#endif

    using std::atan2;
    double atan2yx = atan2(y.value(), x.value());

    double tmp2 = y.value() * y.value();
    double tmp3 = x.value() * x.value();

    assert(tmp2 + tmp3 != 0);

    // For compatibility with Eigen >= 3.3
    Eigen::SparseVector<double> der = (y.derivatives() * x.value() - y.value() * x.derivatives());
    der = (der.eval()) / (tmp2 + tmp3);

    return AD(atan2yx, der);
}

inline AD acos (const AD &x) {
#ifdef PRINT_OP
    std::cout << "AD acos(AD)" << std::endl;
#endif
    assert(abs(x.value()) != 1);

    using std::acos;
    double acosx = acos(x.value());

    return AD(acosx, x.derivatives() / (sqrt(1 - x.value() * x.value())));
}

inline AD exp(const AD &x) {
#ifdef PRINT_OP
    std::cout << "AD exp(AD)" << std::endl;
#endif

    using std::exp;
    double expx = exp(x.value());

    return AD(expx, expx * x.derivatives());
}

inline AD log(const AD &x) {
#ifdef PRINT_OP
    std::cout << "AD log(AD)" << std::endl;
#endif

    assert(x.value() > 0);

    using std::log;
    double logx = log(x.value());

    assert(logx != 0);

    return AD(logx, x.derivatives() / x.value());
}

// hash support
namespace std {
    template <>
    struct hash<AD> {
        std::size_t operator() (AD const& x) const {
            std::hash<AD::Scalar> hasher;
            return hasher(x.value());
        }
    };
} // namespace std

#endif // EIGEN_AUTODIFF_SCALAR_H
