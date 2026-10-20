import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "find_working_cann.py"
SPEC = importlib.util.spec_from_file_location("find_working_cann", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_find_cast_marker_accepts_legacy_cast_json(tmp_path):
    opp = tmp_path / "cann" / "opp"
    marker = (
        opp
        / "built-in"
        / "op_impl"
        / "ai_core"
        / "tbe"
        / "kernel"
        / "config"
        / "ascend910b"
        / "cast.json"
    )
    marker.parent.mkdir(parents=True)
    marker.write_text("{}", encoding="utf-8")

    assert MODULE.find_cast_marker(opp) == marker


def test_find_cast_marker_rejects_unrelated_binary_index(tmp_path):
    opp = tmp_path / "cann" / "opp"
    marker = opp / "kernel" / "config" / "ascend910b" / "ops_oam" / "binary_info_config.json"
    marker.parent.mkdir(parents=True)
    marker.write_text('{"opType": "HealthCheck"}', encoding="utf-8")

    assert MODULE.find_cast_marker(opp) is None


def test_find_cast_marker_accepts_binary_index_containing_cast(tmp_path):
    opp = tmp_path / "cann" / "opp"
    marker = opp / "kernel" / "config" / "ascend910b" / "binary_info_config.json"
    marker.parent.mkdir(parents=True)
    marker.write_text('{"opType": "Cast"}', encoding="utf-8")

    assert MODULE.find_cast_marker(opp) == marker


def test_discovers_opp_and_matching_set_env(tmp_path):
    cann = tmp_path / "copied" / "cann-9.1.0"
    opp = cann / "opp"
    opp.mkdir(parents=True)
    set_env = cann / "set_env.sh"
    set_env.write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    assert MODULE.discover_opp_roots([tmp_path], max_depth=5) == [opp.resolve()]
    assert MODULE.nearest_set_env(opp) == set_env.resolve()


def test_discovers_renamed_copied_opp_root(tmp_path):
    copied_opp = tmp_path / "copied_cann" / "operator_package"
    (copied_opp / "built-in" / "op_impl").mkdir(parents=True)

    assert MODULE.discover_opp_roots([tmp_path], max_depth=5) == [
        copied_opp.resolve()
    ]


def test_rebases_all_paths_exported_by_copied_set_env(tmp_path):
    old_root = "/share/user/copied-cann/cann-9.1.0"
    new_root = "/mnt/pipeline/copied-cann/cann-9.1.0"
    opp = Path(new_root) / "opp"
    environment = {
        "ASCEND_HOME_PATH": old_root,
        "ASCEND_OPP_PATH": f"{old_root}/opp",
        "LD_LIBRARY_PATH": f"{old_root}/lib64:/usr/lib64",
        "PYTHONPATH": f"{old_root}/python/site-packages",
        "UNCHANGED": "/somewhere/else",
    }

    rebased = MODULE.rebase_environment(environment, old_root, new_root, opp)

    assert rebased["ASCEND_HOME_PATH"] == new_root
    assert rebased["ASCEND_TOOLKIT_HOME"] == new_root
    assert rebased["ASCEND_OPP_PATH"] == str(opp)
    assert rebased["LD_LIBRARY_PATH"] == f"{new_root}/lib64:/usr/lib64"
    assert rebased["PYTHONPATH"] == f"{new_root}/python/site-packages"
    assert rebased["UNCHANGED"] == "/somewhere/else"
