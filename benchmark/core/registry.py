"""Name-to-factory registries (adapters now; datasets and layers in later phases)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class RegistryError(KeyError):
    pass


class Registry(Generic[T]):
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        def decorator(item: T) -> T:
            self.add(name, item)
            return item

        return decorator

    def add(self, name: str, item: T) -> None:
        if name in self._items:
            raise RegistryError(f"{self.kind} '{name}' is already registered")
        self._items[name] = item

    def get(self, name: str) -> T:
        try:
            return self._items[name]
        except KeyError:
            known = ", ".join(sorted(self._items)) or "<none>"
            raise RegistryError(f"Unknown {self.kind} '{name}'. Known: {known}") from None

    def names(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: object) -> bool:
        return name in self._items
