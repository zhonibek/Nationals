import argparse
import hashlib
import json
from pathlib import Path
import platform
import re

import torch

from roboproof.motion_env import MotionEnvironment
from .corpus_warm import anchor_rows, descriptions, digest_json, load_corpus, validate_definition, world_row_weights
from .expert_data import FEATURE_SIZE, FEATURE_TRANSFORM, METHOD, read_json, validate_records, validate_worlds
from .train import source_identity


ROOT = Path(__file__).resolve().parents[2]
METHOD_ID = "explicit-measured-corpus-runtime-transfer-v2"
SCOPE = "Explicit locally verified data transfer, not external custody, interchangeable checkpoints or learned/physical qualification. Original producer runtime and every recorded byte remain unchanged."
KEYS = {"schemaVersion", "method", "datasetManifestSha256", "producerTorchVersion", "admissionTorchVersion",
        "admissionPlatform", "sources", "consumerSources", "engineIdentity", "corpusDefinition", "contentsSha256",
        "newNativeQueries", "consumerEngineRuntime", "scope"}


def validate_engine_runtime(value):
    if (not isinstance(value, dict) or set(value) != {"node", "v8", "platform", "arch"} or
            any(not isinstance(item, str) or not 1 <= len(item) <= 128 for item in value.values()) or
            re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+", value["node"]) is None or
            re.fullmatch(r"[A-Za-z0-9.+_-]+", value["v8"]) is None or
            value["platform"] not in ("win32", "linux", "darwin") or value["arch"] not in ("x64", "arm64")):
        raise ValueError("Exact explicitly declared consumer Node/V8/platform/architecture required")
    return dict(value)


def consumer_sources():
    names = ("roboproof/motion_learning/corpus_portability.py", "roboproof/motion_learning/corpus_warm.py")
    return {name: hashlib.sha256((ROOT / name).read_text(encoding="utf-8-sig").replace("\r\n", "\n").encode()).hexdigest()
            for name in names}


def contents_digest(corpus):
    encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"), allow_nan=False)
    digest = hashlib.sha256()
    content = {name: corpus[name] for name in ("rows", "anchorRows", "rowWeights", "worlds")}
    for chunk in encoder.iterencode(content):
        digest.update(chunk.encode())
    return digest.hexdigest()


def admit_transfer(directory, attestation_file, consumer_engine_runtime=None):
    consumer_runtime = validate_engine_runtime(consumer_engine_runtime) if consumer_engine_runtime is not None else None
    corpus = load_corpus(directory)
    manifest, manifest_sha = read_json(Path(directory) / "dataset.json")
    if manifest_sha != corpus["definition"]["manifestSha256"] or manifest["torchVersion"] != str(torch.__version__):
        raise ValueError("Actual strict producer-runtime corpus admission required")
    receipt = {"schemaVersion": 1, "method": METHOD_ID, "datasetManifestSha256": manifest_sha,
        "producerTorchVersion": manifest["torchVersion"], "admissionTorchVersion": str(torch.__version__),
        "admissionPlatform": platform.system(), "sources": source_identity(), "consumerSources": consumer_sources(),
        "engineIdentity": manifest["engineIdentity"], "corpusDefinition": corpus["definition"],
        "contentsSha256": contents_digest(corpus), "newNativeQueries": 0,
        "consumerEngineRuntime": consumer_runtime, "scope": SCOPE}
    with Path(attestation_file).open("x", encoding="utf-8") as output:
        json.dump(receipt, output, indent=2, allow_nan=False)
    return receipt


def load_transfer(directory, attestation_file, expected_attestation_sha256):
    receipt, receipt_sha = read_json(attestation_file)
    if (receipt_sha != expected_attestation_sha256 or not isinstance(receipt, dict) or set(receipt) != KEYS or
            type(receipt["schemaVersion"]) is not int or receipt["schemaVersion"] != 1 or
            receipt["method"] != METHOD_ID or receipt["scope"] != SCOPE or
            type(receipt["newNativeQueries"]) is not int or receipt["newNativeQueries"] != 0 or receipt["sources"] != source_identity() or
            receipt["consumerSources"] != consumer_sources() or
            not isinstance(receipt["producerTorchVersion"], str) or not 1 <= len(receipt["producerTorchVersion"]) <= 128 or
            receipt["admissionTorchVersion"] != receipt["producerTorchVersion"] or
            not isinstance(receipt["admissionPlatform"], str) or receipt["admissionPlatform"] not in ("Windows", "Linux", "Darwin")):
        raise ValueError("Exact explicit strict-admission receipt and consumer/producer sources required")
    expected = validate_definition(receipt["corpusDefinition"])
    declared_engine_runtime = receipt["consumerEngineRuntime"]
    if declared_engine_runtime is not None:
        declared_engine_runtime = validate_engine_runtime(declared_engine_runtime)
    directory = Path(directory)
    manifest, manifest_sha = read_json(directory / "dataset.json")
    packet, packet_sha = read_json(directory / "worlds.json")
    worlds = validate_worlds(packet)
    if (manifest_sha != receipt["datasetManifestSha256"] or manifest_sha != expected["manifestSha256"] or
            manifest.get("schemaVersion") != 1 or manifest.get("definition", {}).get("method") != METHOD or
            manifest["definition"].get("featureTransform") != FEATURE_TRANSFORM or manifest["definition"].get("featureSize") != FEATURE_SIZE or
            manifest.get("status") != "completed" or manifest.get("pendingWorld") is not None or
            manifest.get("unknownInterruptedCost") is True or manifest.get("worldCount") != len(worlds) or
            manifest.get("sources") != receipt["sources"] or packet_sha != manifest.get("worldsSha256") or
            manifest.get("torchVersion") != receipt["producerTorchVersion"] or
            len(manifest.get("records", [])) != len(worlds) or manifest.get("engineIdentity") != receipt["engineIdentity"]):
        raise ValueError("Original complete source/runtime-bound producer bytes must remain unchanged")
    records = list(validate_records(directory, manifest, worlds))
    rows = [row for record in records for row in record["rows"]]
    definition = {"schemaVersion": 1, "method": expected["method"], "manifestSha256": manifest_sha,
        "worldsSha256": manifest["worldsSha256"], "recordsSha256": digest_json(manifest["records"]),
        "producerSourcesSha256": digest_json(manifest["sources"]), "featureTransform": FEATURE_TRANSFORM,
        "featureSize": FEATURE_SIZE, "worldCount": len(records), "labelWorldCount": sum(bool(record["rows"]) for record in records),
        "rowCount": len(rows), "sourceActualQueries": manifest["actualQueries"],
        "nativeCurriculumOffset": len(records), **descriptions()}
    if validate_definition(definition) != expected:
        raise ValueError("Exact actual corpus counts/source costs and learning definition required")
    corpus = {"definition": definition, "rows": rows, "anchorRows": anchor_rows(records),
              "rowWeights": world_row_weights(records), "worlds": [record["world"] for record in records]}
    following, following_sha = read_json(directory / "dataset.json")
    if following_sha != manifest_sha or following != manifest or contents_digest(corpus) != receipt["contentsSha256"]:
        raise ValueError("Every admitted row/weight/anchor/world and producer manifest must remain unchanged")
    with MotionEnvironment() as environment:
        consumer_engine = environment.contract["identity"]
        producer_engine = receipt["engineIdentity"]
        if declared_engine_runtime is None:
            if consumer_engine != producer_engine:
                raise ValueError("Original producing Simulator/iraLIB identity required without explicit runtime transfer")
        elif (validate_engine_runtime(consumer_engine.get("runtime")) != declared_engine_runtime or
                {name: value for name, value in consumer_engine.items() if name != "runtime"} !=
                {name: value for name, value in producer_engine.items() if name != "runtime"}):
            raise ValueError("Exact unchanged Simulator sources/WASM/contract and separately declared consumer runtime required")
    corpus["portability"] = {"method": METHOD_ID, "attestationSha256": receipt_sha,
        "producerTorchVersion": receipt["producerTorchVersion"], "consumerTorchVersion": str(torch.__version__),
        "consumerPlatform": platform.system(), "producerEngineRuntime": receipt["engineIdentity"]["runtime"],
        "consumerEngineRuntime": consumer_engine["runtime"], "engineSourceIdentityPreserved": True,
        "engineRuntimeParityVerified": False, "newNativeQueries": 0, "scope": SCOPE}
    return corpus


def main():
    parser = argparse.ArgumentParser(description="Explicitly admit immutable measured data for another training runtime; no fitting or checkpoint conversion")
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--attestation-file", type=Path, required=True)
    parser.add_argument("--consumer-engine-runtime-file", type=Path)
    actual = parser.parse_args()
    runtime = read_json(actual.consumer_engine_runtime_file)[0] if actual.consumer_engine_runtime_file is not None else None
    receipt = admit_transfer(actual.directory, actual.attestation_file, runtime)
    print(json.dumps({"method": METHOD_ID, "rows": receipt["corpusDefinition"]["rowCount"],
                      "producerTorchVersion": receipt["producerTorchVersion"], "newNativeQueries": 0}), flush=True)


if __name__ == "__main__":
    main()
