import importlib.metadata

try:
    __version__ = importlib.metadata.version("autora")
except importlib.metadata.PackageNotFoundError:
    __version__ = "source_repository"
