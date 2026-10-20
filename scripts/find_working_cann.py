#!/usr/bin/env python3
"""Find and probe complete Ascend 910B CANN/OPP installations.

The tool is read-only: it searches existing directories and runs a tiny Cast
operation in a fresh subprocess for each candidate.  It never installs,
deletes, links, or overwrites CANN files.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform
import re
import subprocess
import sys
from pathlib import Path


DEFAULT_SEARCH_ROOTS = (
    Path("/usr/local/Ascend"),
    Path("/mnt/pipeline-data/f50058111"),
    Path("/share/f50058111"),
)
SKIP_DIR_NAMES = {
    ".git",
    "__pycache__",
    "datasets",
    "eval_results",
    "eval_results_diffusion_1020",
    "huggingface_cache",
    "models",
    "node_modules",
    "response_cache",
}
CAST_TOKEN = re.compile(rb"(?i)(?<![a-z0-9_])cast(?![a-z0-9_])")
PROBE_CODE = r"""
import torch
import torch_npu

print(f"torch={torch.__version__}")
print(f"torch_npu={getattr(torch_npu, '__version__', 'unknown')}")
print(f"ASCEND_HOME_PATH={__import__('os').environ.get('ASCEND_HOME_PATH', '')}")
print(f"ASCEND_OPP_PATH={__import__('os').environ.get('ASCEND_OPP_PATH', '')}")
if not torch.npu.is_available():
    raise SystemExit("NPU is unavailable")
torch.npu.set_device(0)
x = torch.tensor([[1, 0]], dtype=torch.int64, device="npu:0")
y = x.bool()
torch.npu.synchronize()
print(f"CAST_OK dtype={y.dtype}")
""".strip()


def _has_910b_component(path: Path) -> bool:
    return "ascend910b" in str(path).lower()


def _binary_index_contains_cast(path: Path) -> bool:
    """Search a possibly large JSON index without loading it all into memory."""
    overlap = b""
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                data = overlap + chunk
                if CAST_TOKEN.search(data):
                    return True
                overlap = data[-32:]
    except OSError:
        return False
    return False


def find_cast_marker(opp_root: Path) -> Path | None:
    """Return a 910B kernel index that actually contains the Cast operator."""
    if not opp_root.is_dir():
        return None

    cast_files: list[Path] = []
    binary_indexes: list[Path] = []
    for current, dirs, files in os.walk(opp_root, followlinks=False):
        dirs[:] = [name for name in dirs if name not in SKIP_DIR_NAMES]
        current_path = Path(current)
        if not _has_910b_component(current_path):
            # Keep walking because the SoC directory can be below this level.
            pass
        for name in files:
            lowered = name.lower()
            path = current_path / name
            if not _has_910b_component(path):
                continue
            if lowered == "cast.json":
                cast_files.append(path)
            elif lowered == "binary_info_config.json":
                binary_indexes.append(path)

    for path in sorted(cast_files):
        if os.access(path, os.R_OK):
            return path
    for path in sorted(binary_indexes):
        if _binary_index_contains_cast(path):
            return path
    return None


def discover_opp_roots(
    search_roots: list[Path] | tuple[Path, ...], max_depth: int
) -> list[Path]:
    """Find OPP roots, including copied trees whose directory was renamed."""
    found: dict[str, Path] = {}
    for search_root in search_roots:
        if not search_root.is_dir():
            continue
        root_depth = len(search_root.parts)
        for current, dirs, _files in os.walk(search_root, followlinks=False):
            current_path = Path(current)
            depth = len(current_path.parts) - root_depth
            dirs[:] = [
                name
                for name in dirs
                if name not in SKIP_DIR_NAMES and depth < max_depth
            ]
            if current_path.name == "opp" or (
                current_path / "built-in" / "op_impl"
            ).is_dir():
                resolved = current_path.resolve()
                found[str(resolved)] = resolved
                dirs[:] = []
                continue
            # os.walk does not enter a directory symlink with followlinks=False.
            # Register an `opp` symlink (or renamed OPP child) from its parent.
            for name in dirs:
                child = current_path / name
                if name == "opp" or (child / "built-in" / "op_impl").is_dir():
                    resolved = child.resolve()
                    found[str(resolved)] = resolved
    return sorted(found.values(), key=str)


def nearest_set_env(opp_root: Path) -> Path | None:
    """Find the activation script belonging to the candidate OPP tree."""
    current = opp_root.parent
    for _ in range(6):
        candidates = (
            current / "set_env.sh",
            current / "ascend-toolkit" / "set_env.sh",
            current / "latest" / "set_env.sh",
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        if current.parent == current:
            break
        current = current.parent
    return None


def _last_output_lines(stdout: str, stderr: str, limit: int = 14) -> str:
    lines = [line for line in (stdout + "\n" + stderr).splitlines() if line.strip()]
    return "\n".join(lines[-limit:])


def probe_candidate(
    opp_root: Path,
    env_script: Path | None,
    timeout_seconds: int,
) -> tuple[bool, str]:
    """Run Cast in a new process so every candidate gets a clean CANN init."""
    environment = os.environ.copy()
    environment["ASCEND_OPP_PATH"] = str(opp_root)
    if env_script is None:
        command = [sys.executable, "-c", PROBE_CODE]
    else:
        shell_program = (
            'set +e; set +u; export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-}"; '
            'source "$1" >/dev/null 2>&1; export ASCEND_OPP_PATH="$2"; '
            'exec "$3" -c "$4"'
        )
        command = [
            "bash",
            "--noprofile",
            "--norc",
            "-c",
            shell_program,
            "cann-probe",
            str(env_script),
            str(opp_root),
            sys.executable,
            PROBE_CODE,
        ]
    try:
        result = subprocess.run(  # noqa: S603 - argv is passed without shell=True.
            command,
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return False, f"probe timed out after {timeout_seconds}s: {exc}"
    output = _last_output_lines(result.stdout, result.stderr)
    return result.returncode == 0 and "CAST_OK" in result.stdout, output


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search existing CANN/OPP trees and test Ascend 910B Cast.",
    )
    parser.add_argument(
        "--search-root",
        action="append",
        type=Path,
        help="root to scan; repeat for multiple roots",
    )
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="list complete Cast candidates without using the NPU",
    )
    parser.add_argument(
        "--check-opp",
        type=Path,
        help="validate one OPP root and print its Cast index",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.max_depth < 1 or args.timeout < 1:
        raise SystemExit("--max-depth and --timeout must be positive")

    if args.check_opp is not None:
        marker = find_cast_marker(args.check_opp)
        if marker is None:
            print(
                f"No readable Ascend 910B Cast kernel index under {args.check_opp}",
                file=sys.stderr,
            )
            return 2
        print(marker)
        return 0

    search_roots = args.search_root or list(DEFAULT_SEARCH_ROOTS)
    print("=== Current Python environment ===")
    print(f"python={sys.executable}")
    print(f"architecture={platform.machine()}")
    print(f"torch={_package_version('torch')}")
    print(f"torch_npu={_package_version('torch-npu')}")
    print(f"ASCEND_HOME_PATH={os.environ.get('ASCEND_HOME_PATH', '')}")
    print(f"ASCEND_OPP_PATH={os.environ.get('ASCEND_OPP_PATH', '')}")
    print("search_roots=" + ",".join(str(root) for root in search_roots))

    opp_roots = discover_opp_roots(search_roots, args.max_depth)
    candidates: list[tuple[Path, Path, Path | None]] = []
    print("\n=== OPP candidates ===")
    for opp_root in opp_roots:
        marker = find_cast_marker(opp_root)
        if marker is None:
            print(f"[INCOMPLETE] {opp_root} (no 910B Cast kernel index)")
            continue
        env_script = nearest_set_env(opp_root)
        candidates.append((opp_root, marker, env_script))
        print(f"[CANDIDATE] opp={opp_root}")
        print(f"            index={marker}")
        print(f"            set_env={env_script or '<reuse current runtime>'}")

    if not candidates:
        print(
            "\nNo complete Ascend 910B Cast candidate was found. The CANN ops/"
            "kernels package is absent or outside the search roots.",
            file=sys.stderr,
        )
        return 1
    if args.list_only:
        return 0

    print("\n=== Isolated Cast probes ===")
    for number, (opp_root, _marker, env_script) in enumerate(candidates, start=1):
        print(f"[{number}/{len(candidates)}] testing {opp_root}")
        passed, output = probe_candidate(opp_root, env_script, args.timeout)
        if output:
            print(output)
        if passed:
            print("\n[WORKING] A complete, executable CANN/OPP candidate was found.")
            prefix = f"CANN_OPP_PATH={opp_root}"
            if env_script is not None:
                prefix = f"CANN_ENV={env_script} {prefix}"
            print(f"Run:\n  {prefix} bash hf_diffusion_xyf2.sh")
            return 0
        print("[FAILED] Cast did not execute with this candidate.\n")

    print(
        "No candidate passed Cast. Install/copy a mutually compatible CANN "
        "Toolkit + ops package for the displayed torch_npu version.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
