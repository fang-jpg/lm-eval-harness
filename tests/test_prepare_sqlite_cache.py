import importlib.util
import sqlite3
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_sqlite_cache.py"
SPEC = importlib.util.spec_from_file_location("prepare_sqlite_cache", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def make_healthy_database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE cache (key TEXT PRIMARY KEY, value TEXT)")
        connection.execute("INSERT INTO cache VALUES ('answer', '42')")


def test_prepare_caches_keeps_healthy_rank_cache(tmp_path: Path) -> None:
    prefix = tmp_path / "responses"
    healthy = tmp_path / "responses_rank0.db"
    make_healthy_database(healthy)

    healthy_count, corrupt_count, quarantine = MODULE.prepare_caches(
        prefix, stamp="test"
    )

    assert (healthy_count, corrupt_count) == (1, 0)
    assert quarantine is None
    assert healthy.exists()


def test_prepare_caches_quarantines_only_corrupt_group(tmp_path: Path) -> None:
    prefix = tmp_path / "responses"
    healthy = tmp_path / "responses_rank0.db"
    corrupt = tmp_path / "responses_rank1.db"
    sidecar = tmp_path / "responses_rank1.db-wal"
    make_healthy_database(healthy)
    corrupt.write_bytes(b"this is not a sqlite database")
    sidecar.write_bytes(b"stale wal")

    def check(database: Path) -> tuple[bool, str]:
        return (True, "ok") if database == healthy else (False, "forced corruption")

    with mock.patch.object(MODULE, "quick_check", side_effect=check):
        healthy_count, corrupt_count, quarantine = MODULE.prepare_caches(
            prefix, stamp="test"
        )

    assert (healthy_count, corrupt_count) == (1, 1)
    assert quarantine is not None
    assert healthy.exists()
    assert not corrupt.exists()
    assert not sidecar.exists()
    assert (quarantine / corrupt.name).read_bytes() == b"this is not a sqlite database"
    assert (quarantine / sidecar.name).read_bytes() == b"stale wal"


def test_quick_check_rejects_malformed_database(tmp_path: Path) -> None:
    corrupt = tmp_path / "responses_rank0.db"
    corrupt.write_bytes(b"this is not a sqlite database")

    healthy, detail = MODULE.quick_check(corrupt)

    assert not healthy
    assert "database" in detail.lower()


def test_prepare_caches_accepts_missing_cache(tmp_path: Path) -> None:
    healthy_count, corrupt_count, quarantine = MODULE.prepare_caches(
        tmp_path / "responses", stamp="test"
    )

    assert (healthy_count, corrupt_count) == (0, 0)
    assert quarantine is None
