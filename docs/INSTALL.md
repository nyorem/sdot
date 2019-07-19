---
title: Installation instructions for the sdot package
author: Jocelyn Meyron
urlcolor: blue
---

# Dependencies

- CMake >= 2.8.12
- a C++11 capable compiler, for instance `g++ >= 4.8.1` or `clang++`
- Python 3 and the `pip` package manager with `python3` pointing to the Python executable
- CGAL >= 4.12
- Eigen

Installation of the dependencies:

### OS X

```
brew install cmake python cgal eigen
```

### Archlinux

```
sudo pacman -S cmake python python-pip cgal eigen
```

# Installation

- If not installed in a standard location (different from `/usr`), set the environment variable `CGAL_DIR` to point to the directory containing the file `CGALConfig.cmake` of your CGAL installation
- Download [this](https://github.com/nyorem/sdot/raw/master/releases/sdot-0.1.zip) zip file
- Unzip and install the package locally (Python dependencies are automatically fetched and installed) using the following instructions:

```
unzip sdot.zip
cd sdot
python3 setup.py install --user
```
