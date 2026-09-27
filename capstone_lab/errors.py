class CapstoneLabError(Exception):
    """Base exception for expected command failures."""


class ConfigError(CapstoneLabError):
    """The campaign configuration is invalid or unsafe."""


class StateError(CapstoneLabError):
    """Persisted campaign state conflicts with the requested run."""


class ArtifactError(CapstoneLabError):
    """A completed artifact did not satisfy its contract."""


class ManifestError(CapstoneLabError):
    """A source or generated manifest violated the S2 contract."""
