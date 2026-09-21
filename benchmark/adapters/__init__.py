"""Model adapters. Importing this package registers all built-in adapters."""

from benchmark.adapters import mock as _mock  # noqa: F401  (registers the mock adapter)
from benchmark.adapters.base import adapter_registry

__all__ = ["adapter_registry"]
