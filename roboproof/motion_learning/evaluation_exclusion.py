import hashlib
import json
import math
from pathlib import Path
import stat


def geometry_keys(task):
    start, goal = task["start"], task["goal"]
    values = [pose[name] for pose in (start, goal) for name in ("xIn", "yIn", "headingDeg")]
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
        raise ValueError("Finite original motion poses required")
    angle = start["headingDeg"] * math.pi / 180
    delta_x, delta_y = goal["xIn"] - start["xIn"], goal["yIn"] - start["yIn"]
    coordinates = ((delta_x * math.cos(angle) - delta_y * math.sin(angle)) * 10,
                   (delta_x * math.sin(angle) + delta_y * math.cos(angle)) * 10,
                   (goal["headingDeg"] - start["headingDeg"] + 540) % 360 - 180)
    bins = []
    for value in coordinates:
        rounded = math.floor(value + 0.5)
        choices = [rounded]
        lower = math.floor(value)
        if abs(value - (lower + 0.5)) <= 1e-8:
            choices = [lower, lower + 1]
        bins.append(choices)
    return {hashlib.sha256(f"{right}:{forward}:{heading}".encode()).hexdigest()
            for right in bins[0] for forward in bins[1] for heading in bins[2]}


def _record(filename, kind):
    before = filename.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > 4 * 1024 * 1024:
        raise ValueError("Bounded regular consumed-final audit files required")
    data = filename.read_bytes()
    after = filename.stat()
    if len(data) != before.st_size or after.st_size != before.st_size or after.st_mtime_ns != before.st_mtime_ns:
        raise ValueError("Consumed-final audit changed during read")
    def reject_constant(value):
        raise ValueError("Finite consumed-final JSON required")
    value = json.loads(data.decode("utf-8"), parse_constant=reject_constant)
    if (not isinstance(value, dict) or value.get("schemaVersion") != 1 or
            value.get("kind") != f"local-filesystem-holdout-audit-{kind}" or
            value.get("localFilesystemAuditOnlyNotTamperProofSecrecy") is not True):
        raise ValueError("Valid historical consumed-final audit required")
    return value, hashlib.sha256(data).hexdigest()


def reject_consumed_final(world, root):
    directory = Path(root) / "roboproof/runs/motion/holdout"
    if not directory.exists():
        return world
    folders = list(directory.iterdir())
    if len(folders) > 2048:
        raise ValueError("Bounded consumed-final directory inventory required")
    incoming = geometry_keys(world["task"])
    world_count = bytes_read = 0
    for folder in folders:
        if folder.is_symlink():
            raise ValueError("Regular owned holdout directories required")
        if not folder.is_dir() or not (folder / "consumed.json").exists():
            continue
        records = {}
        for kind in ("commitment", "candidates", "consumed"):
            filename = folder / f"{kind}.json"
            records[kind] = _record(filename, kind)
            bytes_read += filename.stat().st_size
            if bytes_read > 128 * 1024 * 1024:
                raise ValueError("Consumed-final read budget exceeded")
        commitment, commitment_sha = records["commitment"]
        freeze, freeze_sha = records["candidates"]
        consumed = records["consumed"][0]
        if (freeze.get("commitmentSha256") != commitment_sha or consumed.get("commitmentSha256") != commitment_sha or
                consumed.get("freezeSha256") != freeze_sha or not commitment.get("identitySha256") or
                any(record.get("identitySha256") != commitment["identitySha256"] for record in (freeze, consumed))):
            raise ValueError("Consumed-final audit hash linkage mismatch")
        cases = commitment.get("corpus", {}).get("cases")
        if not isinstance(cases, list) or not cases or len(cases) != commitment.get("caseCount") or len(cases) > 4096:
            raise ValueError("Bounded complete consumed-final cases required")
        world_count += len(cases)
        if world_count > 200000:
            raise ValueError("Consumed-final world budget exceeded")
        for case in cases:
            if world["seed"] == case["seed"] or incoming.intersection(geometry_keys(case["task"])):
                raise RuntimeError("Consumed final world or body-geometry alias is excluded from training")
    return world
