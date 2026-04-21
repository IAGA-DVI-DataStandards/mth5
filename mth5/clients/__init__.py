"""Client package exports.

Use lazy imports so optional dependencies do not fail package import at module
import time. Symbols are loaded on first access.
"""

from importlib import import_module


_SYMBOL_TO_MODULE = {
    "FDSN": ".fdsn",
    "USGSGeomag": ".geomag",
    "PhoenixClient": ".phoenix",
    "ZenClient": ".zen",
    "LEMIClient": ".lemi",
    "LEMI424Client": ".lemi",
    "MetronixClient": ".metronix",
    "NIMSClient": ".nims",
    "UoAClient": ".uoa",
    "MakeMTH5": ".make_mth5",
}

__all__ = [
    "FDSN",
    "USGSGeomag",
    "PhoenixClient",
    "ZenClient",
    "LEMIClient",
    "LEMI424Client",  # Deprecated alias for backward compatibility
    "MetronixClient",
    "NIMSClient",
    "UoAClient",
    "MakeMTH5",
]


def __getattr__(name):
    """Lazily import client symbols on first access."""
    if name not in _SYMBOL_TO_MODULE:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(_SYMBOL_TO_MODULE[name], package=__name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(list(globals().keys()) + __all__)
