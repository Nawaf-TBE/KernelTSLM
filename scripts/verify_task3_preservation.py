"""Recheck the immutable workspace/source inventory saved before Task 3."""

import hashlib
import json
from pathlib import Path
import subprocess


def file_hash(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    base = root / "outputs/phase2/task3"
    upstream = root.parent / "TimeNet"
    before = json.loads((base / "preservation_before.json").read_text())
    changed, missing = [], []
    inventories = [
        ("workspace", root, before["workspace_files"]),
        ("original_timenet", upstream, before["upstream_source_files"]),
    ]
    for prefix, parent, files in inventories:
        for relative, expected in files.items():
            path = parent / relative
            if not path.is_file():
                missing.append(prefix + ":" + relative)
            elif file_hash(path) != expected:
                changed.append(prefix + ":" + relative)
    aliases = {name: str((root / name).readlink()) for name in before["aliases"]}
    status = subprocess.check_output(
        ["git", "-C", str(upstream), "status", "--short"], text=True
    )
    passed = (
        not changed and not missing and aliases == before["aliases"]
        and status == before["upstream_status"]
    )
    result = {
        "status": "passed" if passed else "failed",
        "workspace_files_checked": len(before["workspace_files"]),
        "original_timenet_source_files_checked": len(before["upstream_source_files"]),
        "changed": changed,
        "missing": missing,
        "aliases_unchanged": aliases == before["aliases"],
        "upstream_status_unchanged": status == before["upstream_status"],
        "upstream_status": status,
    }
    (base / "preservation_after.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit("Task 3 preservation failed")


if __name__ == "__main__":
    main()
