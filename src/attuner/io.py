"""Durable JSON records with readable, non-cryptographic run metadata."""
from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path, recover_tail=False):
    path = Path(path)
    if not path.exists():
        return []
    rows, offset = [], 0
    with path.open("rb") as handle:
        lines = handle.readlines()
    for index, line in enumerate(lines):
        try:
            if not line.strip():
                raise ValueError("blank JSONL record")
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            if not recover_tail or index != len(lines) - 1 or line.endswith(b"\n"):
                raise
            quarantine = path.with_name(path.name + f".tail-{os.getpid()}")
            quarantine.write_bytes(line)
            with path.open("r+b") as handle:
                handle.truncate(offset)
            break
        rows.append(row)
        offset += len(line)
    return rows


def indexed(rows):
    result = {}
    for row in rows:
        key = str(row["instance_id"])
        if key in result:
            raise ValueError(f"duplicate instance_id: {key}")
        result[key] = row
    return result


def exact_coverage(rows, expected):
    actual = indexed(rows)
    expected = [str(key) for key in expected]
    if len(expected) != len(set(expected)):
        raise ValueError("Expected IDs contain duplicates")
    expected = set(expected)
    if set(actual) != expected:
        raise ValueError(
            f"coverage mismatch: missing={sorted(expected-set(actual))[:10]}, "
            f"unexpected={sorted(set(actual)-expected)[:10]}"
        )
    return actual


def file_info(path):
    path = Path(path)
    stat = path.stat()
    return {"bytes": stat.st_size, "modified_ns": stat.st_mtime_ns,
            "changed_ns": stat.st_ctime_ns, "inode": stat.st_ino, "device": stat.st_dev}


def execution_config(config):
    """Only inference-affecting settings, not training paths or task selection."""
    keys = {"model", "dtype", "attention_backend", "enable_thinking", "seed",
            "cache_dir", "cache_memory_gib", "adapter", "generation", "dataset_generation",
            "quality_only", "efficiency_only", "efficiency", "recompute_query_batch"}
    return {k: v for k, v in config.items() if k in keys or
            k.startswith(("ape_", "epic_", "cacheblend_"))}


def comparable_identity(value):
    """Ignore retired audit snapshots when reading older runs."""
    ignored = {"code", "evaluator_source", "model_files", "runtime_versions",
               "configuration", "mixed_data_manifest", "toolqa", "toolqa_environment", "changed_ns", "inode", "device"}
    if isinstance(value, dict):
        if "image" in value and "engine" in value:
            value = {k: v for k, v in value.items() if k not in {"image_description", "environment"}}
        return {k: comparable_identity(execution_config(v) if k == "config" and isinstance(v, dict) else v)
                for k, v in value.items() if k not in ignored}
    if isinstance(value, list):
        return [comparable_identity(v) for v in value]
    return value


def check_file_record(path, record, message="Completed file content changed"):
    """Cheap stat check only; no content reads, reference copies or digests."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Invalid completed file: {path}")
    current = file_info(path)
    if any(key in record and record[key] != current[key] for key in ("bytes", "modified_ns")):
        raise ValueError(f"{message}: {path}")


def ensure_manifest(directory, identity, metadata=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "run_manifest.json"
    if path.exists():
        old = json.loads(path.read_text())
        if comparable_identity(old.get("identity")) != comparable_identity(identity):
            raise ValueError(f"refusing incompatible resume: {path}")
        return old
    result = {
        "identity": identity,
        "command": sys.argv,
        "hostname": platform.node(),
        "python": sys.version,
        **(metadata or {}),
    }
    write_json(path, result)
    return result


REQUIRED_RESULTS = (
    "run_manifest.json",
    "predictions.jsonl",
    "summary.json",
    "paired_analysis.json",
)


def verified_complete(directory, expected_identity=None):
    directory = Path(directory)
    marker_path = directory / "COMPLETE"
    if not marker_path.exists():
        return False
    marker = json.loads(marker_path.read_text())
    if not isinstance(marker, dict) or "files" not in marker:
        raise ValueError(f"Incomplete completion certificate: {directory}")
    names = [item['path'] for item in marker['files']]
    if len(names) != len(set(names)) or not set(REQUIRED_RESULTS) <= set(names):
        raise ValueError(f'Incomplete completion file list: {directory}')
    for item in marker["files"]:
        if item["path"] == "training_inputs.jsonl":
            continue  # obsolete snapshot; never required to load an adapter
        path = Path(item["path"])
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Invalid completed file: {path}")
        check_file_record(directory / path, item)
    manifest = json.loads((directory / "run_manifest.json").read_text())
    if expected_identity is not None:
        for key, value in comparable_identity(expected_identity).items():
            if comparable_identity(manifest["identity"]).get(key) != value:
                raise ValueError(f"Completed run differs from current {key}: {directory}")
    if manifest['identity'].get('student') == 'pic' and manifest['identity'].get('teacher') == 'full':
        if not {'adapter/adapter.pt', 'adapter/adapter_config.json'} <= set(names):
            raise ValueError(f'Training completion lacks required artifacts: {directory}')
    return True


def complete(directory):
    directory = Path(directory)
    names = list(REQUIRED_RESULTS)
    for optional in (
        "official/evaluation.json",
        "official/command.json",
        "adapter/adapter.pt",
        "adapter/adapter_config.json",
    ):
        if (directory / optional).exists():
            names.append(optional)
    files = []
    for name in names:
        path = directory / name
        if not path.is_file():
            raise ValueError(f"cannot complete: {name} missing in {directory}")
        files.append({"path": name, **file_info(path)})
    write_json(directory / "COMPLETE", {"files": files, "verification": "file-stats-v1"})
    verified_complete(directory)
