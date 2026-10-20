#!/usr/bin/env python3
"""Validate lm-eval rank-local SQLite caches before an evaluation attempt.

Interrupted multi-rank evaluations can leave one or more response databases
unreadable.  Retrying against the same malformed database fails before model
inference starts.  This helper keeps healthy rank caches and moves only broken
databases (plus their SQLite sidecars) into a recoverable quarantine directory.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterable


SQLITE_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")


def quick_check(database: Path) -> tuple[bool, str]:
    """Return whether SQLite considers *database* structurally healthy."""

    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(str(database), timeout=10.0)
        connection.execute("PRAGMA query_only = ON")
        rows = connection.execute("PRAGMA quick_check").fetchall()
    except sqlite3.Error as exc:
        return False, f"{type(exc).__name__}: {exc}"
    finally:
        if connection is not None:
            connection.close()

    messages = [str(row[0]) for row in rows]
    if messages == ["ok"]:
        return True, "ok"
    return False, "; ".join(messages) or "PRAGMA quick_check returned no result"


def cache_files(database: Path) -> Iterable[Path]:
    """Yield a database and any SQLite sidecars that currently exist."""

    yield database
    for suffix in SQLITE_SIDECAR_SUFFIXES:
        sidecar = Path(f"{database}{suffix}")
        if sidecar.exists():
            yield sidecar


def prepare_caches(
    cache_prefix: Path,
    quarantine_root: Path | None = None,
    stamp: str | None = None,
) -> tuple[int, int, Path | None]:
    """Check all ``<prefix>_rank*.db`` files and quarantine corrupt groups."""

    cache_prefix = cache_prefix.expanduser().resolve()
    candidates = sorted(cache_prefix.parent.glob(f"{cache_prefix.name}_rank*.db"))
    if not candidates:
        print(f"[CACHE CHECK] No existing rank cache found for {cache_prefix}_rank<N>.db")
        return 0, 0, None

    if quarantine_root is None:
        quarantine_root = Path(f"{cache_prefix}_corrupt")
    quarantine_root = quarantine_root.expanduser().resolve()
    stamp = stamp or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    quarantine_dir: Path | None = None
    healthy_count = 0
    corrupt_count = 0

    for database in candidates:
        healthy, detail = quick_check(database)
        if healthy:
            healthy_count += 1
            print(f"[CACHE CHECK] Healthy: {database}")
            continue

        corrupt_count += 1
        if quarantine_dir is None:
            quarantine_dir = quarantine_root / f"{stamp}_pid{os.getpid()}"
            quarantine_dir.mkdir(parents=True, exist_ok=False)

        print(f"[CACHE CHECK] Corrupt: {database} ({detail})")
        for source in cache_files(database):
            destination = quarantine_dir / source.name
            source.replace(destination)
            print(f"[CACHE CHECK] Quarantined: {source} -> {destination}")

    print(
        "[CACHE CHECK] Summary: "
        f"healthy={healthy_count}, quarantined={corrupt_count}, total={len(candidates)}"
    )
    if quarantine_dir is not None:
        print(f"[CACHE CHECK] Recoverable corrupt files: {quarantine_dir}")
    return healthy_count, corrupt_count, quarantine_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate lm-eval SQLite response caches and quarantine corrupt files."
    )
    parser.add_argument(
        "--cache-prefix",
        type=Path,
        required=True,
        help="Cache prefix passed to lm-eval --use_cache (without _rank<N>.db).",
    )
    parser.add_argument(
        "--quarantine-root",
        type=Path,
        default=None,
        help="Directory under which timestamped quarantine folders are created.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        prepare_caches(args.cache_prefix, args.quarantine_root)
    except OSError as exc:
        print(f"[CACHE CHECK] Failed to quarantine a corrupt cache: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
