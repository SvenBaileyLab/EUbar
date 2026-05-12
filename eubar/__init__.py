from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("eubar")
except PackageNotFoundError:
    __version__ = "unknown"