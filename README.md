# Semi-discrete optimal transport algorithms in Python

- [Detailed installation instructions](docs/INSTALL.pdf)
- [Documentation](docs/README.pdf)

## Dependencies

- cmake >= 2.8.12
- a recent C/C++ compiler (at least C++11 capable)
- Python 3
- CGAL >= 4.12
- Eigen

The other dependencies (pybind11, NNRVD, geogram) are included as git submodules and can be fetched using the following command:

```
git submodule update --init --recursive
```

## Installation

Set the environment variable `CGAL_DIR` to point to the directory containing the file `CGALConfig.cmake`

Install with:
```
python3 setup.py install --user
```

Run the tests with:
```
pytest -p no:warnings
```

We ignore the warnings since `scipy.sparse` raises a `PendingDeprecationWarning` exception due to its usage of the `numpy.matrix` class, see: https://github.com/scipy/scipy/issues/9093.

## References

- http://www.benjack.io/2018/02/02/python-cpp-revisited.html

- [An algorithm for optimal transport between a simplex soup and a point cloud](https://doi.org/10.1137/17M1137486),  Quentin Mérigot, Jocelyn Meyron, Boris Thibert. [arXiv preprint](https://arxiv.org/abs/1707.01337)

- [Initialization procedures for discrete and semi-discrete optimal transport](https://doi.org/10.1016/j.cad.2019.05.037), Jocelyn Meyron. [preprint](https://www.meyronj.com/static/articles/2019_initialization_ot.pdf)
