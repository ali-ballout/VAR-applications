"""shoplift-vision package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("shoplift-vision")
except PackageNotFoundError:  # pragma: no cover - fallback for local installs
    __version__ = "0.1.0"

__all__ = ["__version__"]
