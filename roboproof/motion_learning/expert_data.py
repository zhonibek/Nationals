import argparse
import hashlib
import json
import math
from pathlib import Path
import stat
import time

import torch

from roboproof.motion_env import MotionEnvironment
from .imitation import collect_demonstration
from .model import MotionActorCritic, DEADLINE_CONTEXT, normalization_definition
from .teacher_profiles import ADAPTIVE_FINE_POSE_HEADING_PROFILE, teacher_profile_definition
from .train import atomic_json, exclude_frozen_world, source_identity


METHOD = "measured-public-motion-corpus-v1"
FEATURE_TRANSFORM = "deadline-context-v1"
MAX_FILE_BYTES = 4 * 1024 * 1024
MAX_WORLDS = 256
FEATURE_SIZE = 145


def read_json(filename):
    filename = Path(filename)
    before = filename.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_FILE_BYTES:
        raise ValueError("Bounded regular teacher-data file required")
    with filename.open("rb") as source:
        content = source.read(MAX_FILE_BYTES + 1)
    after = filename.stat()
    if len(content) != before.st_size or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Teacher-data file changed during read")

    def reject_constant(value):
        raise ValueError("Finite teacher-data JSON required")

    return json.loads(content.decode("utf-8-sig"), parse_constant=reject_constant), hashlib.sha256(content).hexdigest()


def validate_worlds(packet):
    worlds = packet.get("cases") if isinstance(packet, dict) else None
    if not isinstance(worlds, list) or not 1 <= len(worlds) <= MAX_WORLDS:
        raise ValueError("One to256 predeclared training worlds required")
    ids, seeds = set(), set()
    for world in worlds:
        if (not isinstance(world, dict) or not isinstance(world.get("id"), str) or not 1 <= len(world["id"]) <= 128 or
                not isinstance(world.get("group"), str) or not 1 <= len(world["group"]) <= 128 or
                isinstance(world.get("seed"), bool) or not isinstance(world.get("seed"), int) or not 0 <= world["seed"] <= 2**32 - 1 or
                world["id"] in ids or world["seed"] in seeds or not isinstance(world.get("configuration"), dict)):
            raise ValueError("Unique declared world IDs/seeds and family groups required")
        task = world.get("task")
        if not isinstance(task, dict) or set(task) != {"start", "goal", "deadlineSeconds"}:
            raise ValueError("Original reach task required")
        deadline = task["deadlineSeconds"]
        if isinstance(deadline, bool) or not isinstance(deadline, (int, float)) or not math.isfinite(deadline) or not 1 <= deadline <= 10:
            raise ValueError("Bounded original training deadline required")
        for pose in (task["start"], task["goal"]):
            if not isinstance(pose, dict) or set(pose) != {"xIn", "yIn", "headingDeg"}:
                raise ValueError("Original finite training pose required")
            for name, limit in (("xIn", 60), ("yIn", 60), ("headingDeg", 180)):
                value = pose[name]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value) > limit:
                    raise ValueError("Original finite training pose required")
        exclude_frozen_world(world)
        ids.add(world["id"])
        seeds.add(world["seed"])
    return worlds


def validate_rows(rows, feature_size, deadline):
    if not isinstance(rows, list) or len(rows) > math.ceil(deadline / 0.05) + 2:
        raise ValueError("Bounded complete teacher trajectory required")
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"modelInput", "teacherAction"}:
            raise ValueError("Public encoded input and actually applied teacher action required")
        for name, size in (("modelInput", feature_size), ("teacherAction", 4)):
            values = row[name]
            if (not isinstance(values, list) or len(values) != size or
                    any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in values)):
                raise ValueError("Finite matching public teacher trajectory required")
        if any(abs(value) > 1 for value in row["teacherAction"]):
            raise ValueError("Original bounded four-axis teacher action required")
    return rows


def accepted_rows(measured, feature_size, deadline):
    rows = validate_rows(measured["rows"], feature_size, deadline)
    if measured.get("selectedPlan") is None and not rows:
        if measured["reason"] == "success":
            raise ValueError("Successful teacher needs its actual selected trajectory")
        return [], "teacher-unavailable"
    matches = [attempt for attempt in measured["attempts"] if attempt.get("plan") == measured["selectedPlan"]]
    if len(matches) != 1 or measured["reason"] != "success" or matches[0]["reason"] != "success":
        raise ValueError("Actually completed selected teacher measurement required")
    if matches[0]["metrics"]["contactSeconds"] != 0:
        return [], "selected-contact-success-excluded"
    if not rows:
        raise ValueError("Complete selected zero-contact trajectory rows required")
    return rows, "success-zero-contact"


def collect(worlds_file, directory, max_queries=250000, max_seconds=1800, resume=False, worlds_per_call=256):
    if (isinstance(max_queries, bool) or not isinstance(max_queries, int) or not 1 <= max_queries <= 2000000 or
            isinstance(max_seconds, bool) or not isinstance(max_seconds, int) or not 1 <= max_seconds <= 3600):
        raise ValueError("Bounded total original-engine data-collection budgets required")
    if isinstance(worlds_per_call, bool) or not isinstance(worlds_per_call, int) or not 1 <= worlds_per_call <= MAX_WORLDS:
        raise ValueError("Bounded clean world slices required")
    packet, input_sha = read_json(worlds_file)
    worlds = validate_worlds(packet)
    sources = source_identity()
    directory = Path(directory).resolve()
    definition = {"method": METHOD, "featureTransform": FEATURE_TRANSFORM,
                  "featureSize": FEATURE_SIZE, "deadlineContextDefinition": dict(DEADLINE_CONTEXT),
                  "normalization": normalization_definition(FEATURE_TRANSFORM), "teacherSelection": "pose-deadline-margin-v2",
                  "teacherProfile": ADAPTIVE_FINE_POSE_HEADING_PROFILE,
                  "teacherDefinition": teacher_profile_definition(ADAPTIVE_FINE_POSE_HEADING_PROFILE),
                  "labelAdmission": "Entire actually measured selected success with exactly zero contact; all failed worlds/costs retained",
                  "maxQueries": max_queries, "maxSeconds": max_seconds, "worldsInputSha256": input_sha}
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = MotionActorCritic("broad-pace-v1", FEATURE_TRANSFORM, False, 32, 2)
    if len(model.scales) != FEATURE_SIZE:
        raise ValueError("Declared original public encoder size required")
    with MotionEnvironment() as environment:
        if resume:
            manifest, _ = read_json(directory / "dataset.json")
            if (manifest.get("schemaVersion") != 1 or manifest.get("definition") != definition or manifest.get("sources") != sources or
                    manifest.get("engineIdentity") != environment.contract["identity"] or manifest.get("torchVersion") != str(torch.__version__) or
                    manifest.get("status") != "budget-stopped" or manifest.get("pendingWorld") is not None):
                raise ValueError("Exact clean source/runtime/world/budget continuation required; unknown interrupted cost cannot resume")
            stored_worlds, stored_sha = read_json(directory / "worlds.json")
            if stored_sha != manifest["worldsSha256"] or stored_worlds != packet:
                raise ValueError("Original declared training worlds changed")
            exposure, _ = read_json(directory / "exposure.json")
            for _ in validate_records(directory, manifest, worlds):
                pass
        else:
            if directory.exists():
                raise ValueError("Never overwrite a teacher-data collection")
            directory.mkdir(parents=True)
            atomic_json(directory / "worlds.json", packet)
            _, stored_sha = read_json(directory / "worlds.json")
            manifest = {"schemaVersion": 1, "definition": definition, "sources": sources,
                        "engineIdentity": environment.contract["identity"], "torchVersion": str(torch.__version__),
                        "worldsSha256": stored_sha, "worldCount": len(worlds), "records": [], "discardedAttempts": [],
                        "actualQueries": 0, "discardedQueries": 0, "elapsedCollectionSeconds": 0,
                        "status": "budget-stopped", "pendingWorld": None, "gradientsPerformed": False,
                        "learnedImprovementVerified": False,
                        "scope": "Original Simulator/iraLIB training experience only; not independent evaluation, learned improvement or physical evidence"}
            exposure = {"schemaVersion": 1, "cases": []}
            atomic_json(directory / "dataset.json", manifest)
        started = time.monotonic()
        deadline = started + max(0, max_seconds - manifest["elapsedCollectionSeconds"])
        elapsed_before = manifest["elapsedCollectionSeconds"]
        first_index = len(manifest["records"])
        for index in range(first_index, min(len(worlds), first_index + worlds_per_call)):
            if manifest["actualQueries"] >= max_queries or time.monotonic() >= deadline:
                break
            world = worlds[index]
            exposure["cases"].append({**world, "kind": "measured-expert-data", "worldIndex": index})
            atomic_json(directory / "exposure.json", exposure)
            manifest["status"] = "collecting-world"
            manifest["pendingWorld"] = index
            atomic_json(directory / "dataset.json", manifest)
            try:
                measured = collect_demonstration(environment, model, world, deadline, max_queries - manifest["actualQueries"],
                    definition["teacherSelection"], definition["teacherProfile"])
                manifest["actualQueries"] += measured["observedSteps"]
                manifest["elapsedCollectionSeconds"] = elapsed_before + time.monotonic() - started
                if measured["interrupted"]:
                    manifest["discardedQueries"] += measured["observedSteps"]
                    manifest["discardedAttempts"].append({"worldIndex": index, "observedSteps": measured["observedSteps"],
                                                         "attempts": measured["attempts"]})
                    manifest["status"] = "budget-stopped"
                    manifest["pendingWorld"] = None
                    atomic_json(directory / "dataset.json", manifest)
                    break
                rows, label_reason = accepted_rows(measured, len(model.scales), world["task"]["deadlineSeconds"])
                record = {"schemaVersion": 1, "worldIndex": index, "world": world, "featureSize": len(model.scales),
                          "labelReason": label_reason, "rows": rows,
                          "excludedRows": measured["rows"] if label_reason == "selected-contact-success-excluded" else [],
                          "measurement": {name: value for name, value in measured.items() if name != "rows"}}
                filename = f"world-{index:04d}.json"
                if (directory / filename).exists():
                    raise ValueError("Never overwrite measured world evidence")
                atomic_json(directory / filename, record)
                _, record_sha = read_json(directory / filename)
                manifest["records"].append({"worldIndex": index, "filename": filename, "sha256": record_sha,
                                            "rowCount": len(rows), "actualQueries": measured["observedSteps"], "labelReason": label_reason})
                manifest["status"] = "budget-stopped"
                manifest["pendingWorld"] = None
                atomic_json(directory / "dataset.json", manifest)
                print(json.dumps({"event": "expert-world-recorded", "worldIndex": index, "rows": len(rows),
                                  "actualQueries": manifest["actualQueries"], "labelReason": label_reason}), flush=True)
            except BaseException:
                manifest["status"] = "interrupted-or-failed"
                manifest["unknownInterruptedCost"] = True
                atomic_json(directory / "dataset.json", manifest)
                raise
        manifest["elapsedCollectionSeconds"] = elapsed_before + time.monotonic() - started
        manifest["status"] = "completed" if len(manifest["records"]) == len(worlds) else "budget-stopped"
        if source_identity() != sources:
            raise ValueError("Source drift during original-engine data collection")
        atomic_json(directory / "dataset.json", manifest)
        return manifest


def validate_records(directory, manifest, worlds):
    if not isinstance(manifest.get("records"), list) or len(manifest["records"]) > len(worlds):
        raise ValueError("Bounded committed teacher records required")
    spent = manifest["discardedQueries"]
    if isinstance(spent, bool) or not isinstance(spent, int) or spent < 0:
        raise ValueError("Known nonnegative discarded teacher cost required")
    for index, saved in enumerate(manifest["records"]):
        if saved.get("worldIndex") != index or saved.get("filename") != f"world-{index:04d}.json":
            raise ValueError("Every declared world in original order required")
        record, record_sha = read_json(directory / saved["filename"])
        if (record_sha != saved["sha256"] or record.get("world") != worlds[index] or record.get("worldIndex") != index or
                record.get("schemaVersion") != 1 or record.get("featureSize") != FEATURE_SIZE):
            raise ValueError("Measured teacher record hash/world mismatch")
        measured = record["measurement"]
        if (measured.get("interrupted") is not False or measured["observedSteps"] != saved["actualQueries"] or
                isinstance(measured["observedSteps"], bool) or not isinstance(measured["observedSteps"], int) or measured["observedSteps"] < 1):
            raise ValueError("Complete measured teacher query record required")
        excluded_rows = record.get("excludedRows", [])
        if excluded_rows and (record["rows"] or record["labelReason"] != "selected-contact-success-excluded"):
            raise ValueError("Contact trajectory must remain separate from admitted teacher labels")
        rows, label_reason = accepted_rows({**measured, "rows": record["rows"] or excluded_rows}, record["featureSize"], worlds[index]["task"]["deadlineSeconds"])
        if label_reason != record["labelReason"] or label_reason != saved["labelReason"] or len(rows) != saved["rowCount"]:
            raise ValueError("Original complete admitted teacher rows required")
        spent += measured["observedSteps"]
        yield record
    if spent != manifest["actualQueries"]:
        raise ValueError("All actual/discarded teacher queries must reconcile")


def iter_demonstrations(directory):
    directory = Path(directory)
    manifest, _ = read_json(directory / "dataset.json")
    packet, packet_sha = read_json(directory / "worlds.json")
    worlds = validate_worlds(packet)
    if (manifest.get("schemaVersion") != 1 or manifest.get("definition", {}).get("method") != METHOD or
            manifest["definition"].get("featureTransform") != FEATURE_TRANSFORM or manifest["definition"].get("featureSize") != FEATURE_SIZE or
            manifest.get("status") != "completed" or manifest.get("pendingWorld") is not None or
            manifest.get("unknownInterruptedCost") is True or manifest.get("worldCount") != len(worlds) or
            manifest.get("sources") != source_identity() or packet_sha != manifest.get("worldsSha256") or
            manifest.get("torchVersion") != str(torch.__version__) or len(manifest.get("records", [])) != len(worlds)):
        raise ValueError("Complete exact source/runtime-bound teacher corpus required")
    with MotionEnvironment() as environment:
        if manifest.get("engineIdentity") != environment.contract["identity"]:
            raise ValueError("Original producing Simulator/iraLIB identity required")
    for _ in validate_records(directory, manifest, worlds):
        pass
    yield from validate_records(directory, manifest, worlds)


def main():
    parser = argparse.ArgumentParser(description="Record measured public expert trajectories in original Simulator/iraLIB; no model training")
    parser.add_argument("--worlds-file", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--max-queries", type=int, default=250000)
    parser.add_argument("--max-seconds", type=int, default=1800)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worlds-per-call", type=int, default=256)
    result = collect(**vars(parser.parse_args()))
    print(json.dumps({"event": "expert-data-finished", "status": result["status"], "worlds": len(result["records"]),
                      "actualQueries": result["actualQueries"], "rows": sum(row["rowCount"] for row in result["records"]),
                      "learnedImprovementVerified": False}), flush=True)


if __name__ == "__main__":
    main()
