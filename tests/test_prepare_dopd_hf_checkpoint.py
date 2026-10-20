import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_dopd_hf_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("prepare_dopd_hf_checkpoint", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def make_source(path: Path, model_type: str = "a2d-qwen3") -> None:
    path.mkdir()
    write_json(
        path / "config.json",
        {
            "model_type": model_type,
            "architectures": ["A2DQwen3ForCausalLM"],
            "auto_map": {"AutoConfig": "dopd.A2DQwen3Config"},
            "vocab_size": 32,
            "hidden_size": 8,
        },
    )
    write_json(path / "tokenizer_config.json", {"tokenizer_class": "Qwen2Tokenizer"})
    write_json(path / "tokenizer.json", {"version": "1.0"})


def test_prepare_checkpoint_normalizes_config_and_copies_tokenizer(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    make_source(source)
    output.mkdir()
    (output / "model.safetensors").write_bytes(b"x" * 32)

    weights, assets = MODULE.prepare_checkpoint(source, output, mask_token_id=31)

    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert weights == "model.safetensors"
    assert config["model_type"] == "qwen3"
    assert config["architectures"] == ["Qwen3ForCausalLM"]
    assert config["mask_token_id"] == 31
    assert "auto_map" not in config
    assert "tokenizer.json" in assets
    assert (output / "tokenizer_config.json").is_file()


def test_normalize_rejects_unknown_model_type() -> None:
    try:
        MODULE.normalize_qwen3_config({"model_type": "llama", "vocab_size": 32}, 1)
    except ValueError as exc:
        assert "model_type" in str(exc)
    else:
        raise AssertionError("unknown model type was accepted")


def test_normalize_rejects_mask_outside_vocabulary() -> None:
    try:
        MODULE.normalize_qwen3_config(
            {"model_type": "a2d-qwen3", "vocab_size": 32}, 32
        )
    except ValueError as exc:
        assert "outside vocab_size" in str(exc)
    else:
        raise AssertionError("out-of-range mask token was accepted")


def test_validate_exported_weights_rejects_missing_shard(tmp_path: Path) -> None:
    write_json(
        tmp_path / "model.safetensors.index.json",
        {"weight_map": {"model.embed_tokens.weight": "model-00001.safetensors"}},
    )

    try:
        MODULE.validate_exported_weights(tmp_path)
    except ValueError as exc:
        assert "Missing or truncated" in str(exc)
    else:
        raise AssertionError("missing weight shard was accepted")
