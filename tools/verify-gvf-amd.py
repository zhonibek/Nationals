import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import errno
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import signal
import stat
import subprocess
import sys
import time
import traceback
import zipfile


ROOT = Path(__file__).absolute().parents[1]
TOTAL_SECONDS = 720
WORK_SECONDS = 600
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_SOURCE_BYTES = 8 * 1024 * 1024
MAX_SOURCE_TOTAL = 64 * 1024 * 1024
MAX_LOG_BYTES = 2 * 1024 * 1024
MAX_RESULT_BYTES = 16 * 1024 * 1024
MIN_MEMORY_BYTES = 180 * 1024 ** 3
REQUIRED_SOURCES = frozenset((
    "tools/verify-gvf-amd.py",
    "tests/roboproof_gvf_amd_guard_test.py",
    "tests/roboproof_gvf_model_remote_test.py",
    "tests/roboproof_gvf_td3_remote_test.py",
    "tests/gvf-model-contract.test.js",
    "roboproof/motion_learning/__init__.py",
    "roboproof/motion_learning/gvf_contract.py",
    "roboproof/motion_learning/gvf_model.py",
    "roboproof/motion_learning/gvf_td3.py",
    "roboproof/motion-gvf-learner.js",
    "simulator/gvf-policy.js",
))
NODE_TEST_PATTERN = "^(separate shallow/deep synthetic GVF networks|sensor-relative feature transform|synthetic fixed output|raw legacy vectors|old algorithms|sparse, malformed|stdlib Python and JavaScript)"

GPU_PROBE = r'''
import json, platform, sys, traceback
from pathlib import Path
evidence = {"system": platform.system(), "python": platform.python_version(), "executionCompleted": False}
try:
    import torch
    evidence.update(torch=str(torch.__version__), hip=torch.version.hip, cudaBuild=torch.version.cuda,
                    cudaAvailable=torch.cuda.is_available(), deviceCount=torch.cuda.device_count())
    if not evidence["hip"] or evidence["cudaBuild"] is not None or not evidence["cudaAvailable"] or evidence["deviceCount"] < 1:
        raise RuntimeError("Actual ROCm Torch and an available HIP device required; no CPU fallback")
    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    evidence.update(device="cuda:0", deviceIndex=torch.cuda.current_device(), name=properties.name,
                    totalMemoryBytes=properties.total_memory, gcnArchName=getattr(properties, "gcnArchName", None))
    if "AMD" not in properties.name.upper() or "MI300X" not in properties.name.upper() or properties.total_memory < 180 * 1024 ** 3:
        raise RuntimeError("Actual AMD MI300X with at least 180 GiB required before GPU arithmetic")
    torch.set_num_threads(1)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Tanh(), torch.nn.Linear(8, 2)).to("cuda:0")
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(0.125)
    inputs = (torch.arange(8, device="cuda:0", dtype=torch.float32).reshape(2, 4) / 8).requires_grad_(True)
    prediction = model(inputs)
    loss = prediction.square().mean()
    loss.backward()
    torch.cuda.synchronize(0)
    gradients = [inputs.grad, *(parameter.grad for parameter in model.parameters())]
    evidence.update(inputDevice=str(inputs.device), outputDevice=str(prediction.device), lossDevice=str(loss.device),
                    parameterDevices=[str(parameter.device) for parameter in model.parameters()],
                    gradientDevices=[None if gradient is None else str(gradient.device) for gradient in gradients],
                    inputShape=list(inputs.shape), outputShape=list(prediction.shape), loss=float(loss.detach().cpu()),
                    forwardFinite=bool(torch.isfinite(prediction).all().item() and torch.isfinite(loss).item()),
                    backwardFinite=all(gradient is not None and bool(torch.isfinite(gradient).all().item()) for gradient in gradients),
                    gradientL1=[float(gradient.abs().sum().detach().cpu()) for gradient in gradients],
                    synchronized=True, executionCompleted=True, optimizerSteps=0)
except BaseException as error:
    evidence["error"] = str(error)
    traceback.print_exc()
finally:
    with Path(sys.argv[1]).open("x", encoding="utf-8") as stream:
        json.dump(evidence, stream, indent=2, allow_nan=False)
    print(json.dumps(evidence, allow_nan=False), flush=True)
if not evidence["executionCompleted"]:
    sys.exit(1)
'''

CPU_CHECKS = r'''
import importlib.util, json, sys, types, unittest
from pathlib import Path
root = Path(sys.argv[1])
sys.path.insert(0, str(root))
package = types.ModuleType("roboproof")
package.__path__ = [str(root / "roboproof")]
sys.modules["roboproof"] = package
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.set_default_device("cpu")
if torch.cuda.is_initialized() or torch.cuda.is_available():
    raise RuntimeError("Synthetic optimizer checks must be CPU-only in a separate GPU-hidden process")
suite = unittest.TestSuite()
for name in ("roboproof_gvf_model_remote_test", "roboproof_gvf_td3_remote_test"):
    spec = importlib.util.spec_from_file_location(name, root / "tests" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
if suite.countTestCases() != 16:
    raise RuntimeError("Exactly the existing 16 synthetic model/TD3 checks required")
result = unittest.TextTestRunner(verbosity=2).run(suite)
summary = dict(scope="synthetic optimizer checks, not motion learning", testDevice="cpu", testsRun=result.testsRun,
               failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
               expectedFailures=len(result.expectedFailures), unexpectedSuccesses=len(result.unexpectedSuccesses),
               successful=result.wasSuccessful(), gpuInitialized=torch.cuda.is_initialized())
with Path(sys.argv[2]).open("x", encoding="utf-8") as stream:
    json.dump(summary, stream, indent=2)
print(json.dumps(summary), flush=True)
sys.exit(0 if result.wasSuccessful() else 1)
'''

NODE_GUARD = r'''
'use strict';
const Module = require('node:module');
const path = require('node:path');
const root = process.env.ROBOPROOF_FROZEN_ROOT;
const allowed = new Set(['tests/gvf-model-contract.test.js', 'roboproof/motion-gvf-learner.js', 'simulator/gvf-policy.js'].map(name => path.join(root, name)));
const headless = path.join(root, 'simulator/headless.js');
const original = Module._load;
Module._load = function(request, parent, isMain) {
  const filename = Module._resolveFilename(request, parent, isMain);
  if (filename === headless) return Object.freeze({loadHeadless() { throw Error('Native physics is forbidden in AMD synthetic checks'); }});
  if (path.isAbsolute(filename) && !allowed.has(filename)) throw Error('Only frozen synthetic JS modules allowed: ' + filename);
  return original.apply(this, arguments);
};
'''


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(content):
    return hashlib.sha256(content).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def no_links(path):
    for ancestor in (path, *path.parents):
        require(not ancestor.is_symlink() and not getattr(ancestor, "is_junction", lambda: False)(),
                "Symlink/junction or ancestor rejected: " + str(ancestor))


def read_regular(path, limit):
    no_links(path)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_size <= limit,
                "Bounded singly linked regular file required: " + str(path))
        content = stream.read(before.st_size + 1)
        after = os.fstat(stream.fileno())
        current = path.lstat()
        stamp = lambda value: (value.st_dev, value.st_ino, value.st_mode, value.st_nlink, value.st_size,
                               value.st_mtime_ns, value.st_ctime_ns)
        stable = stamp(before) == stamp(after) and (
            stamp(after)[:-1] == stamp(current)[:-1] if os.name == "nt" else stamp(after) == stamp(current))
        require(len(content) == before.st_size and stable,
                "File changed while reading: " + str(path))
    return content


def unique_object(pairs):
    result = {}
    for name, value in pairs:
        require(name not in result, "Duplicate JSON key rejected")
        result[name] = value
    return result


def parse_json(content):
    return json.loads(content, object_pairs_hook=unique_object,
                      parse_constant=lambda value: require(False, "Nonfinite JSON rejected: " + value))


def canonical_name(value):
    require(type(value) is str and bool(value) and "\\" not in value and ":" not in value and
            all(ord(character) >= 32 for character in value), "Canonical contained source path required")
    name = PurePosixPath(value)
    require(not name.is_absolute() and name.as_posix() == value and
            all(part not in (".", "..") for part in name.parts), "Canonical contained source path required")
    return name


def verify_sources(filename, expected, snapshot=None, frozen=None):
    snapshot = {} if snapshot is None else snapshot
    snapshot.update(checkedAtUtc=utc_now(), expectedManifestSha256=expected, verified=False, files=[])
    require(type(expected) is str and re.fullmatch("[a-f0-9]{64}", expected), "Exact lowercase manifest SHA256 required")
    content = read_regular(filename, MAX_MANIFEST_BYTES)
    snapshot["observedManifestSha256"] = digest(content)
    pinned = content if frozen is None else frozen
    require(digest(pinned) == expected, "Exact frozen source-manifest SHA256 required")
    manifest = parse_json(pinned)
    rows = manifest.get("files") if type(manifest) is dict else None
    require(type(rows) is list and 1 <= len(rows) <= 512, "Bounded source inventory required")
    names = set()
    total = 0
    matches = digest(content) == expected
    for row in rows:
        require(type(row) is dict and set(row) == {"path", "bytes", "sha256"}, "Exact source inventory row required")
        name = canonical_name(row["path"])
        require(row["path"] not in names, "Unique source inventory required")
        names.add(row["path"])
        require(type(row["bytes"]) is int and 0 <= row["bytes"] <= MAX_SOURCE_BYTES and
                type(row["sha256"]) is str and re.fullmatch("[a-f0-9]{64}", row["sha256"]),
                "Bounded source size and SHA256 required")
        total += row["bytes"]
        require(total <= MAX_SOURCE_TOTAL, "Total source inventory exceeds budget")
        observed = {"path": row["path"], "expectedBytes": row["bytes"], "expectedSha256": row["sha256"], "matches": False}
        snapshot["files"].append(observed)
        try:
            source = read_regular(ROOT.joinpath(*name.parts), MAX_SOURCE_BYTES)
            observed.update(bytes=len(source), sha256=digest(source))
            observed["matches"] = len(source) == row["bytes"] and digest(source) == row["sha256"]
        except (OSError, ValueError) as error:
            observed["error"] = str(error)
        matches = matches and observed["matches"]
    required = set(REQUIRED_SOURCES)
    if (ROOT / "roboproof/__init__.py").exists():
        required.add("roboproof/__init__.py")
    require(required <= names, "Frozen inventory missing required harness/test/module sources: " + ", ".join(sorted(required - names)))
    require(matches, "Frozen source or manifest byte mismatch")
    snapshot["verified"] = True
    return pinned


def validate_probe(value):
    require(type(value) is dict and value.get("system") == "Linux" and
            all(type(value.get(name)) is str and value[name].strip() for name in ("python", "torch", "hip")) and
            value.get("cudaBuild", "missing") is None and value.get("cudaAvailable") is True,
            "Actual Linux ROCm Torch evidence required")
    require(type(value.get("deviceCount")) is int and value["deviceCount"] >= 1 and
            type(value.get("deviceIndex")) is int and value["deviceIndex"] == 0 and value.get("device") == "cuda:0" and
            type(value.get("name")) is str and "AMD" in value["name"].upper() and
            re.search(r"\bMI300X\b", value["name"].upper()) and type(value.get("totalMemoryBytes")) is int and
            value["totalMemoryBytes"] >= MIN_MEMORY_BYTES, "Actual AMD MI300X device 0 with >=180 GiB required")
    require(all(value.get(name) is True for name in ("executionCompleted", "synchronized", "forwardFinite", "backwardFinite")) and
            all(value.get(name) == "cuda:0" for name in ("inputDevice", "outputDevice", "lossDevice")) and
            value.get("parameterDevices") == ["cuda:0"] * 4 and value.get("gradientDevices") == ["cuda:0"] * 5 and
            value.get("inputShape") == [2, 4] and value.get("outputShape") == [2, 2] and
            type(value.get("optimizerSteps")) is int and value["optimizerSteps"] == 0,
            "Synchronized tiny GPU forward/backward evidence required")
    values = value.get("gradientL1")
    require(type(values) is list and len(values) == 5 and
            all(type(number) in (int, float) and math.isfinite(number) and number > 0
                for number in [value.get("loss"), *values]), "Finite nonzero GPU loss/gradients required")


def validate_cpu(value):
    require(type(value) is dict and value.get("testDevice") == "cpu" and value.get("successful") is True and
            value.get("gpuInitialized") is False and type(value.get("testsRun")) is int and value["testsRun"] == 16 and
            all(type(value.get(name)) is int and value[name] == 0 for name in
                ("failures", "errors", "skipped", "expectedFailures", "unexpectedSuccesses")),
            "All 16 actual CPU synthetic checks must pass without skips or GPU initialization")


def validate_node(content):
    counts = {name: int(count) for name, count in re.findall(r"^# (tests|pass|fail|cancelled|skipped|todo) (\d+)\s*$",
                                                          content.decode("utf-8"), re.MULTILINE)}
    require(counts.get("pass") == 7 and counts.get("fail") == counts.get("cancelled") == counts.get("todo") == 0 and
            counts.get("skipped") in (0, 2) and counts.get("tests") == 7 + counts["skipped"],
            "Exactly seven non-native Node synthetic/parity checks must pass")


class ForegroundDeadline(TimeoutError):
    pass


@contextmanager
def foreground_bound():
    def expired(signum, frame):
        raise ForegroundDeadline("720-second whole-foreground deadline; model outcome unresolved")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, TOTAL_SECONDS)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def group_missing(identifier):
    try:
        os.killpg(identifier, 0)
    except ProcessLookupError:
        return True
    except OSError as error:
        if error.errno == errno.ESRCH:
            return True
        return False
    return False


def terminate_child(process, record):
    record["terminationRequested"] = True
    for name, number, seconds in (("SIGTERM", signal.SIGTERM, 2), ("SIGKILL", 9, 2)):
        record["lastTerminationSignal"] = name
        try:
            os.killpg(process.pid, number)
        except ProcessLookupError:
            pass
        except OSError as error:
            record["terminationError"] = str(error)
        try:
            process.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            continue
        if group_missing(process.pid):
            break
    record["directChildReaped"] = process.poll() is not None
    record["processGroupMissing"] = group_missing(process.pid)
    record["exitCode"] = process.returncode


def run_stage(stage, values, output, environment, terminal, deadline, timeout):
    remaining = min(timeout, deadline - time.monotonic())
    require(remaining > 0, "600-second workload budget exhausted before " + stage)
    record = {"stage": stage, "argv": values, "startedAtUtc": utc_now(), "timeoutSeconds": remaining,
              "state": "starting", "modelOutcome": "not established", "directChildReaped": False}
    terminal["commands"].append(record)
    process = None
    paths = [output / (stage + suffix) for suffix in (".stdout.log", ".stderr.log")]
    try:
        with paths[0].open("xb") as stdout, paths[1].open("xb") as stderr:
            process = subprocess.Popen(values, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                       stdout=stdout, stderr=stderr, start_new_session=True)
            record.update(pid=process.pid, processGroup=process.pid, state="running")
            finish = min(deadline, time.monotonic() + remaining)
            while True:
                require(all(path.stat().st_size <= MAX_LOG_BYTES for path in paths), "Child log byte budget exceeded")
                left = finish - time.monotonic()
                if left <= 0:
                    record.update(state="timed_out", timedOut=True, modelOutcome="unresolved after observation timeout")
                    raise subprocess.TimeoutExpired(values, remaining)
                try:
                    process.wait(timeout=min(left, 0.2))
                    break
                except subprocess.TimeoutExpired:
                    pass
            record.update(exitCode=process.returncode, directChildReaped=True,
                          processGroupMissing=group_missing(process.pid))
            require(record["processGroupMissing"], "Child exited but its process group remains; no restart")
            require(all(path.stat().st_size <= MAX_LOG_BYTES for path in paths), "Child log byte budget exceeded")
            require(process.returncode == 0, "Nonzero verification stage: " + stage)
            record.update(state="passed", modelOutcome="stage exited zero; not motion-learning evidence")
    except BaseException as error:
        record["error"] = str(error)
        if record["state"] != "timed_out":
            record["state"] = "interrupted" if isinstance(error, (KeyboardInterrupt, ForegroundDeadline)) else "failed"
        if process is not None:
            terminate_child(process, record)
        raise
    finally:
        record["terminalAtUtc"] = utc_now()


def write_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def prepare_output(value):
    parent = ROOT / "roboproof/runs/motion/research"
    output = value.absolute()
    require(output.parent == parent and re.fullmatch(r"gvf-amd-neural-core-[A-Za-z0-9_-]{1,80}", output.name),
            "New gvf-amd-neural-core-* output directly under roboproof/runs/motion/research required")
    archive = output.with_name(output.name + ".zip")
    no_links(output)
    no_links(archive)
    require(not output.exists() and not archive.exists(), "Output/archive overwrite rejected")
    parent.mkdir(parents=True, exist_ok=True)
    no_links(output)
    output.mkdir(mode=0o700)
    descriptor = os.open(archive, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    return output, archive, os.fdopen(descriptor, "w+b")


def preserve(output, archive_path, archive_stream):
    no_links(output)
    no_links(archive_path)
    require(archive_path.stat().st_ino == os.fstat(archive_stream.fileno()).st_ino, "Archive ownership changed")
    entries = sorted(output.iterdir())
    require(len(entries) <= 32, "Bounded flat result inventory required")
    contents = {}
    for path in entries:
        contents[path.name] = read_regular(path, MAX_LOG_BYTES)
        require(sum(map(len, contents.values())) <= MAX_RESULT_BYTES, "Result byte budget exceeded")
    manifest = {"schemaVersion": 1, "scope": "AMD synthetic optimizer checks, not motion learning or billing certification",
                "files": [{"path": name, "bytes": len(content), "sha256": digest(content)}
                          for name, content in contents.items()]}
    manifest_bytes = json.dumps(manifest, indent=2, allow_nan=False).encode("utf-8")
    with zipfile.ZipFile(archive_stream, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, content in contents.items():
            archive.writestr(name, content)
        archive.writestr("preservation-manifest.json", manifest_bytes)
    archive_stream.flush()
    require(archive_path.stat().st_size <= MAX_RESULT_BYTES + 1024 * 1024, "ZIP byte budget exceeded")
    with zipfile.ZipFile(archive_path) as archive:
        require(archive.namelist() == [*contents, "preservation-manifest.json"] and archive.testzip() is None,
                "ZIP exact inventory/CRC verification required")
        require(archive.read("preservation-manifest.json") == manifest_bytes, "ZIP manifest bytes changed")
        for row in manifest["files"]:
            content = archive.read(row["path"])
            require(len(content) == row["bytes"] and digest(content) == row["sha256"], "ZIP result hash mismatch")
    write_json(output / "preservation-manifest.json", manifest)
    return {"archive": str(archive_path), "bytes": archive_path.stat().st_size,
            "sha256": digest(read_regular(archive_path, MAX_RESULT_BYTES + 1024 * 1024)),
            "archiveVerified": True, "externalCustodyVerified": False}


def execute(options, started):
    output, archive_path, archive_stream = prepare_output(options.output)
    terminal = {"schemaVersion": 1, "startedAtUtc": utc_now(), "completed": False, "state": "running", "commands": [],
                "scope": "synthetic optimizer checks, not motion learning", "wholeForegroundLimitSeconds": TOTAL_SECONDS,
                "workloadLimitSeconds": WORK_SECONDS, "syntheticOptimizerChecksAttempted": False,
                "gpuExecutionVerified": False, "cpuSyntheticChecksVerified": False, "nodeParityChecksVerified": False,
                "movementTrainingPerformed": False, "newPhysicsQueries": 0, "teacherCollectionPerformed": False,
                "trainingFitPerformed": False, "fullRegressionPerformed": False, "learnedImprovementVerified": False,
                "independentFinalPrepared": False, "automaticTrainingOrPromotionEnabled": False,
                "providerBillingVerifiedByScript": False, "providerShutdownPerformed": False,
                "externalCustodyVerified": False}
    frozen = None
    before = {}
    after = {}
    environment = {name: value for name, value in os.environ.items() if name not in
                   ("PYTHONPATH", "PYTHONHOME", "NODE_OPTIONS", "NODE_PATH", "PYTORCH_TESTING_DEVICE_ONLY_FOR")}
    environment.update(PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1", ROBOPROOF_PYTHON=sys.executable,
                       ROBOPROOF_FROZEN_ROOT=str(ROOT), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       NO_COLOR="1", PYTHONUNBUFFERED="1")
    deadline = started + WORK_SECONDS
    try:
        write_json(output / "host.json", {"checkedAtUtc": utc_now(), "system": platform.system(), "uname": platform.uname()._asdict(),
                   "python": platform.python_version(), "pythonExecutable": sys.executable, "cpus": os.cpu_count(),
                   "operatorDeclaredApproval": {"provider": "AMD Developer Cloud", "device": "MI300X", "maxAllocationHours": 2,
                                                "maxPromotionalCreditsUSD": 3.98},
                   "providerBillingVerifiedByScript": False, "allocationOrUploadPerformedByScript": False,
                   "operatorMustVerifyCreditsAndStopAllocation": True, "syntheticTestDevice": "cpu", "probeDevice": "cuda:0"})
        frozen = verify_sources(options.source_manifest, options.source_manifest_sha256, before)
        with (output / "frozen-source-manifest.json").open("xb") as stream:
            stream.write(frozen)
        write_json(output / "source-before.json", before)
        (output / "node-no-physics.cjs").write_text(NODE_GUARD, encoding="utf-8")
        environment["NODE_OPTIONS"] = "--require=" + json.dumps(str(output / "node-no-physics.cjs"))
        run_stage("node-version", ["node", "--version"], output, environment, terminal, deadline, 15)
        require(re.fullmatch(rb"v\d+\.\d+\.\d+\s*", read_regular(output / "node-version.stdout.log", MAX_LOG_BYTES)),
                "Actual Node version evidence required")
        run_stage("mi300x-actual-execution", [sys.executable, "-I", "-B", "-c", GPU_PROBE, str(output / "gpu-probe.json")],
                  output, environment, terminal, deadline, 150)
        validate_probe(parse_json(read_regular(output / "gpu-probe.json", MAX_LOG_BYTES)))
        terminal["gpuExecutionVerified"] = True
        cpu_environment = dict(environment, CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES="")
        terminal["syntheticOptimizerChecksAttempted"] = True
        run_stage("cpu-synthetic-model-td3", [sys.executable, "-I", "-B", "-c", CPU_CHECKS, str(ROOT), str(output / "cpu-checks.json")],
                  output, cpu_environment, terminal, deadline, 240)
        validate_cpu(parse_json(read_regular(output / "cpu-checks.json", MAX_LOG_BYTES)))
        terminal["cpuSyntheticChecksVerified"] = True
        run_stage("node-synthetic-parity", ["node", "--test", "--test-reporter=tap", "--test-name-pattern=" + NODE_TEST_PATTERN,
                  "tests/gvf-model-contract.test.js"], output, cpu_environment, terminal, deadline, 120)
        validate_node(read_regular(output / "node-synthetic-parity.stdout.log", MAX_LOG_BYTES))
        terminal["nodeParityChecksVerified"] = True
        terminal["state"] = "checks_passed_pending_source_recheck"
    except BaseException as error:
        terminal.update(state="interrupted" if isinstance(error, (KeyboardInterrupt, ForegroundDeadline)) else "failed", error=str(error))
        (output / "exception.log").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        try:
            verify_sources(options.source_manifest, options.source_manifest_sha256, after, frozen=frozen)
        except BaseException as error:
            after.update(verified=False, error=str(error))
            terminal.update(state="failed_source_recheck", sourceRecheckError=str(error))
        if not (output / "source-before.json").exists():
            write_json(output / "source-before.json", before)
        write_json(output / "source-after.json", after)
        terminal["completed"] = terminal["state"] == "checks_passed_pending_source_recheck" and after.get("verified") is True
        if terminal["completed"]:
            terminal["state"] = "completed"
        terminal.update(terminalAtUtc=utc_now(), foregroundElapsedSeconds=time.monotonic() - started)
        write_json(output / "terminal.json", terminal)
        try:
            receipt = preserve(output, archive_path, archive_stream)
        finally:
            archive_stream.close()
    print(json.dumps({**receipt, "terminal": terminal}), flush=True)
    return 0 if terminal["completed"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description="Bounded approved AMD MI300X synthetic verification only; no fit, native physics, billing certification or shutdown")
    parser.add_argument("--approved-amd-mi300x", action="store_true", required=True,
                        help="Operator confirms AMD Developer Cloud MI300X, <=2h/$3.98 promotional credits; operator manages billing/shutdown")
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args(argv)
    require(options.approved_amd_mi300x and platform.system() == "Linux" and sys.platform == "linux" and
            os.environ.get("ROBOPROOF_APPROVED_REMOTE") == "1", "Explicitly approved Linux remote only; no laptop fallback")
    started = time.monotonic()
    with foreground_bound():
        return execute(options, started)


if __name__ == "__main__":
    sys.exit(main())
