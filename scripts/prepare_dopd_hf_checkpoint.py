#!/usr/bin/env python3
"""Finish a d-OPD DCP export as a standard Qwen3 HF checkpoint.

The d-OPD student starts from an ``a2d-qwen3`` config, a local training-only
alias of Qwen3.  The generic DCP converter can export its tensors, but
Transformers cannot serialize that unregistered config.  This helper copies
the tokenizer assets and writes an equivalent standard ``qwen3`` config so
lm-eval can load the converted student with AutoModelForCausalLM.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


TOKENIZER_ASSET_NAMES = {
    "added_tokens.json",
    "merges.txt",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.model",
    "vocab.json",
}


def read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def validate_exported_weights(output_dir: Path) -> str:
    """Validate that the converter produced a non-empty single/sharded export."""

    single_file = output_dir / "model.safetensors"
    index_file = output_dir / "model.safetensors.index.json"
    if single_file.is_file():
        if single_file.stat().st_size <= 16:
            raise ValueError(f"Exported weight file is empty or truncated: {single_file}")
        return single_file.name

    if not index_file.is_file():
        raise ValueError(
            f"No model.safetensors or model.safetensors.index.json in {output_dir}."
        )
    index = read_json(index_file)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Invalid or empty weight_map in {index_file}.")
    shard_names = sorted(set(weight_map.values()))
    for shard_name in shard_names:
        if not isinstance(shard_name, str):
            raise ValueError(f"Invalid shard name in {index_file}: {shard_name!r}")
        shard = output_dir / shard_name
        if not shard.is_file() or shard.stat().st_size <= 16:
            raise ValueError(f"Missing or truncated exported weight shard: {shard}")
    return f"{len(shard_names)} shards"


def normalize_qwen3_config(
    source_config: dict[str, Any], mask_token_id: int
) -> dict[str, Any]:
    model_type = source_config.get("model_type")
    if model_type not in {"a2d-qwen3", "qwen3"}:
        raise ValueError(
            "d-OPD source config must have model_type 'a2d-qwen3' or 'qwen3', "
            f"got {model_type!r}."
        )

    vocab_size = source_config.get("vocab_size")
    if not isinstance(vocab_size, int) or vocab_size <= 0:
        raise ValueError(f"Invalid vocab_size in source config: {vocab_size!r}")
    if not 0 <= mask_token_id < vocab_size:
        raise ValueError(
            f"mask_token_id={mask_token_id} is outside vocab_size={vocab_size}."
        )

    config = dict(source_config)
    config["model_type"] = "qwen3"
    config["architectures"] = ["Qwen3ForCausalLM"]
    config["mask_token_id"] = mask_token_id
    # A training-only remote mapping must not override the standard Qwen3
    # implementation used by lm-eval's explicit diffusion attention masks.
    config.pop("auto_map", None)
    return config


def is_tokenizer_asset(path: Path) -> bool:
    return path.name in TOKENIZER_ASSET_NAMES or path.name.startswith(
        ("tokenizer", "chat_template")
    )


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def prepare_checkpoint(
    source_model_dir: Path,
    output_dir: Path,
    mask_token_id: int,
) -> tuple[str, list[str]]:
    source_model_dir = source_model_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not source_model_dir.is_dir():
        raise ValueError(f"Source model directory does not exist: {source_model_dir}")
    if not output_dir.is_dir():
        raise ValueError(f"Converted checkpoint directory does not exist: {output_dir}")

    weights = validate_exported_weights(output_dir)
    source_config = read_json(source_model_dir / "config.json")
    config = normalize_qwen3_config(source_config, mask_token_id)

    copied_assets: list[str] = []
    for source in sorted(source_model_dir.iterdir()):
        if source.is_file() and is_tokenizer_asset(source):
            shutil.copy2(source, output_dir / source.name)
            copied_assets.append(source.name)

    source_generation_config = source_model_dir / "generation_config.json"
    if source_generation_config.is_file():
        shutil.copy2(source_generation_config, output_dir / source_generation_config.name)
        copied_assets.append(source_generation_config.name)

    if not (output_dir / "tokenizer_config.json").is_file():
        raise ValueError(
            f"No tokenizer_config.json found in source model {source_model_dir}."
        )
    if not any(
        (output_dir / name).is_file()
        for name in ("tokenizer.json", "tokenizer.model", "vocab.json")
    ):
        raise ValueError(f"No tokenizer vocabulary asset was copied to {output_dir}.")

    write_json_atomic(output_dir / "config.json", config)
    return weights, sorted(set(copied_assets))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a converted d-OPD checkpoint for standard Qwen3 HF loading."
    )
    parser.add_argument("--source-model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mask-token-id", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        weights, assets = prepare_checkpoint(
            args.source_model_dir, args.output_dir, args.mask_token_id
        )
    except (OSError, ValueError) as exc:
        print(f"[d-OPD HF] Preparation failed: {exc}")
        return 1
    print(f"[d-OPD HF] Weights validated: {weights}")
    print("[d-OPD HF] Wrote standard Qwen3 config.json (model_type=qwen3).")
    print(f"[d-OPD HF] Tokenizer assets ready: {', '.join(assets)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
