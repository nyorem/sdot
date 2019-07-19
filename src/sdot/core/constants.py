from functools import wraps
import os

VERBOSE = True
DEBUG   = False
PROFILE = False

LIBRARY_ROOT             = os.path.dirname(os.path.realpath(os.path.dirname(__file__)))
ASSETS_DIR               = os.path.join(LIBRARY_ROOT, "assets")
PLANE_MESH               = os.path.join(ASSETS_DIR, "plane.off")
CUBE_MESH                = os.path.join(ASSETS_DIR, "cube.off")
HEMISPHERE_MESH          = os.path.join(ASSETS_DIR, "hemisphere.off")
SPHERE_MESH              = os.path.join(ASSETS_DIR, "sphere.off")
PLANE_SIMPLE_MESH        = os.path.join(ASSETS_DIR, "plane_simple.off")

# Decorator to add keyword arguments to a function
def add_keywords_decorator(kwds):
    def decorated(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for key, value in kwds.items():
                kwargs[key] = value
            ret = func(*args, **kwargs)
            return ret
        return wrapper
    return decorated

verbose = add_keywords_decorator({"verbose" : VERBOSE})
debug = add_keywords_decorator({"debug" : DEBUG})

# Test if a keyword exist in a dictionary of keywords
def is_keyword_present(kwd, **kwds):
    return kwd in kwds.keys() and kwds[kwd]
