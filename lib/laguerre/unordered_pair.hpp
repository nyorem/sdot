#ifndef _UNORDERED_PAIR_HPP_
#define _UNORDERED_PAIR_HPP_

namespace std {
    template < typename T >
    struct unordered_pair {
        T first;
        T second;

        unordered_pair () = default;

        unordered_pair (T const& a, T const& b) {
            if (a < b) {
                first = a;
                second = b;
            } else {
                first = b;
                second = a;
            }
        }

        bool operator== (unordered_pair<T> const& other) const {
            return first == other.first && second == other.second;
        }

        bool operator!= (unordered_pair<T> const& other) const {
            return !(*this == other);
        }

        bool operator< (unordered_pair<T> const& other) const {
            if (first < other.first) {
                return true;
            } else if (first == other.first && second < other.second) {
                return true;
            }

            return false;
        }
    };

    template < typename T >
    unordered_pair<T> make_unordered_pair (T const& a, T const& b) {
        return unordered_pair<T>(a, b);
    }
} // namespace std

#endif

