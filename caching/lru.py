"""Simple thread-safe LRU cache implementation."""
from __future__ import annotations

from collections import OrderedDict
from threading import RLock
from typing import Generic, Iterator, Tuple, TypeVar, overload


K = TypeVar("K")
V = TypeVar("V")
T = TypeVar("T")


class LRUCache(Generic[K, V]):
    """Least recently used (LRU) cache with fixed capacity."""

    def __init__(self, maxsize: int = 1024) -> None:
        if maxsize <= 0:
            raise ValueError("maxsize must be positive")
        self.maxsize = int(maxsize)
        self._store: "OrderedDict[K, V]" = OrderedDict()
        self._lock = RLock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def __contains__(self, key: object) -> bool:
        with self._lock:
            return key in self._store

    @overload
    def get(self, key: K) -> V | None:
        ...

    @overload
    def get(self, key: K, default: T) -> V | T:
        ...

    def get(self, key: K, default: T | None = None) -> V | T | None:
        with self._lock:
            try:
                value = self._store.pop(key)
            except KeyError:
                return default
            self._store[key] = value
            return value

    def __getitem__(self, key: K) -> V:
        value = self.get(key)
        if value is None and key not in self:
            raise KeyError(key)
        return value  # type: ignore[return-value]

    def put(self, key: K, value: V) -> None:
        with self._lock:
            if key in self._store:
                self._store.pop(key)
            elif len(self._store) >= self.maxsize:
                self._store.popitem(last=False)
            self._store[key] = value

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def items(self) -> Iterator[Tuple[K, V]]:
        with self._lock:
            return iter(self._store.copy().items())


__all__ = ["LRUCache"]
