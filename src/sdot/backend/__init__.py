class Backend(object):
    available_settings = {}

# Re-export
from .nnrvd import NNRVDBackend
from .geogram import GeogramBackend
from .laguerre import LaguerreBackend

# Default
Density = NNRVDBackend().available_settings["Surface"]

available_backends = [NNRVDBackend, GeogramBackend, LaguerreBackend]
