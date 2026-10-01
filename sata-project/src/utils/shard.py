"""Splitting a list of work units across parallel workers (`--shard i/N`)."""

from __future__ import annotations


def parse_shard(spec: str | None) -> tuple[int, int]:
    """"i/N" -> (i, N), 0 <= i < N; None -> (0, 1)."""
    if not spec:
        return 0, 1
    i, n = (int(x) for x in spec.split("/"))
    if not 0 <= i < n:
        raise ValueError(f"shard {spec!r}: need 0 <= i < N")
    return i, n


def take_shard(items: list, spec: str | None) -> list:
    """Every N-th item starting at i, so strategies and tasks spread evenly over the shards."""
    i, n = parse_shard(spec)
    return list(items[i::n])
