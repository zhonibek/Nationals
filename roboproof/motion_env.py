import json
import math
import os
import queue
import shutil
import subprocess
import threading
from pathlib import Path


class MotionEnvironment:
    PROTOCOL_VERSION = 1
    MAX_RESPONSE_BYTES = 16 * 1024 * 1024
    GVF_TAGS = {"contractName": "gvf-residual", "contractVersion": 2,
                "observationSchema": "gvf-residual-sensor34-v2"}
    GVF_FIELDS = [["pose", 3], ["velocity", 3], ["wheelRimSpeeds", 4], ["targetError", 4],
                  ["previousAction", 3], ["remainingSeconds", 1], ["baseReference", 6],
                  ["projection", 1], ["remainingArcMeters", 1], ["crossTrackMeters", 1],
                  ["appliedFieldOffset", 3], ["offsetVelocity", 3], ["actionAgeTicks", 1]]
    GVF_LIMITS = {"translationOffsetMeters": 0.08, "headingOffsetRadians": 0.1,
                  "offsetSpeedMetersPerSecond": 0.08, "offsetAcceleration": 0.3,
                  "offsetYawRate": 0.25, "offsetYawAcceleration": 1, "fieldMeters": 60 * 0.0254,
                  "referenceSpeedMetersPerSecond": 0.35, "referenceYawRate": 1.5}

    def __init__(self, repository=None, node_executable=None, timeout_seconds=30, control_mode="policy"):
        if control_mode not in ("policy", "gvf-residual"):
            raise ValueError("Unknown motion control_mode")
        if not math.isfinite(timeout_seconds) or not 0.1 <= timeout_seconds <= 120:
            raise ValueError("timeout_seconds must be between 0.1 and 120")
        root = Path(repository or Path(__file__).resolve().parents[1]).resolve()
        executable = node_executable or os.environ.get("ROBOTAI_NODE") or shutil.which("node")
        if not executable:
            raise RuntimeError("Node.js is required; set ROBOTAI_NODE or add node to PATH")
        bridge = root / "roboproof" / "motion-bridge.js"
        if not bridge.is_file():
            raise ValueError("Repository must contain the canonical motion bridge")
        self.timeout_seconds = timeout_seconds
        self._control_mode = control_mode
        self._sequence = 0
        self._closed = False
        self._needs_reset = True
        self._responses = queue.Queue()
        command = [str(executable), str(bridge)]
        if control_mode == "gvf-residual":
            command.append("--contract=gvf-residual")
        self._process = subprocess.Popen(
            command, cwd=str(root), stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, shell=False,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        self._reader = threading.Thread(
            target=self._read_output,
            args=(self._process.stdout, self._responses, self.MAX_RESPONSE_BYTES),
            daemon=True,
        )
        self._reader.start()
        try:
            self.contract = self._request("contract")
            if control_mode == "gvf-residual":
                expected = {**self.GVF_TAGS, "protocolVersion": self.PROTOCOL_VERSION, "episodeSchemaVersion": 2,
                            "controlMode": control_mode, "policyContractVersion": 2,
                            "policyContractName": "gvf-residual", "actionSize": 3, "observationSize": 34,
                            "observationFields": self.GVF_FIELDS, "actionNames": ["dRight", "dForward", "dHeading"],
                            "actionBounds": [-1, 1], "zeroAction": [0, 0, 0], "fixedDt": 0.01,
                            "policyIntervalTicks": 5, "actionTimeoutTicks": 10, "policyLimits": self.GVF_LIMITS}
                if any(not self._same_contract_value(self.contract.get(key), value) for key, value in expected.items()):
                    raise RuntimeError("Unsupported GVF residual identity/schema/action contract")
            elif self.contract["actionSize"] != 4 or self.contract["observationSize"] != 34:
                raise RuntimeError("Unsupported motion observation/action contract")
        except Exception:
            self.close()
            raise

    @staticmethod
    def _same_contract_value(actual, expected):
        if isinstance(expected, dict):
            return isinstance(actual, dict) and actual.keys() == expected.keys() and all(
                MotionEnvironment._same_contract_value(actual[key], value) for key, value in expected.items())
        if isinstance(expected, list):
            return isinstance(actual, list) and len(actual) == len(expected) and all(
                MotionEnvironment._same_contract_value(left, right) for left, right in zip(actual, expected))
        if isinstance(expected, (int, float)) and not isinstance(expected, bool):
            return not isinstance(actual, bool) and isinstance(actual, (int, float)) and math.isfinite(actual) and actual == expected
        return type(actual) is type(expected) and actual == expected

    @staticmethod
    def _read_output(stream, responses, maximum):
        try:
            while True:
                line = stream.readline(maximum + 1)
                if not line:
                    responses.put(RuntimeError("Canonical motion bridge exited"))
                    break
                if len(line) > maximum or not line.endswith(b"\n"):
                    responses.put(RuntimeError("Bridge response exceeds limit or is truncated"))
                    break
                responses.put(line)
        except Exception as error:
            responses.put(error)

    def _request(self, operation, **fields):
        if self._closed:
            raise RuntimeError("Motion environment is closed")
        self._sequence += 1
        request = {"protocolVersion": self.PROTOCOL_VERSION, "id": self._sequence, "op": operation, **fields}
        encoded = json.dumps(request, allow_nan=False, separators=(",", ":")).encode("utf-8") + b"\n"
        if len(encoded) > 16384:
            raise ValueError("Bridge request exceeds 16 KiB")
        try:
            self._process.stdin.write(encoded)
            self._process.stdin.flush()
            reply = self._responses.get(timeout=self.timeout_seconds)
            if isinstance(reply, Exception):
                raise reply
            reply = json.loads(reply)
            if reply.get("id") != self._sequence or reply.get("protocolVersion") != self.PROTOCOL_VERSION:
                raise RuntimeError("Mismatched bridge response identity")
        except Exception as error:
            self.close()
            raise RuntimeError("Canonical motion bridge communication failed") from error
        if not reply.get("ok"):
            raise ValueError(reply.get("error", "Bridge rejected operation"))
        return reply["result"]

    def reset(self, seed=42, configuration=None, task=None, options=None):
        self._needs_reset = True
        if self._control_mode == "gvf-residual" and options is not None and not isinstance(options, dict):
            raise ValueError("GVF residual episode options must be an object")
        request = {"seed": seed, "configuration": configuration or {}, "options": options or {}}
        if task is not None:
            request["task"] = task
        result = self._request("reset", **request)
        if self._control_mode == "gvf-residual":
            self._validate_gvf_vector(result)
            info = {"identity": result["identity"], "options": result["options"], **self.GVF_TAGS}
        else:
            info = {"identity": result["identity"], "options": result["options"],
                    "referenceDurationSeconds": result["referenceDurationSeconds"]}
        self._needs_reset = False
        return result["vector"], info

    def _validate_gvf_vector(self, result):
        observation = result.get("observation", {})
        fields = set(self.GVF_TAGS) | {name for name, _ in self.GVF_FIELDS}
        if (not isinstance(observation, dict)
                or any(result.get(key) != value or observation.get(key) != value for key, value in self.GVF_TAGS.items())
                or set(observation) != fields):
            raise ValueError("Explicit GVF residual observation identity/schema required")
        vector = []
        for name, width in self.GVF_FIELDS:
            values = [observation[name]] if width == 1 else observation[name]
            if not isinstance(values, list) or len(values) != width:
                raise ValueError("Invalid GVF residual sensor fields")
            vector.extend(values)
        returned = result.get("vector")
        if (not isinstance(returned, list) or returned != vector
                or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                       for value in vector + returned)
                or any(not -1 <= value <= 1 for value in observation["previousAction"])
                or observation["remainingSeconds"] < 0 or not 0 <= observation["projection"] <= 1
                or observation["remainingArcMeters"] < 0 or observation["crossTrackMeters"] < 0
                or not float(observation["actionAgeTicks"]).is_integer() or not 0 <= observation["actionAgeTicks"] <= 6000):
            raise ValueError("Invalid GVF residual sensor vector")

    def step(self, action):
        if self._needs_reset:
            raise RuntimeError("Reset before stepping, including after an episode ends")
        try:
            values = list(action)
        except TypeError:
            values = []
        gvf = self._control_mode == "gvf-residual"
        if len(values) != (3 if gvf else 4) or any(isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not math.isfinite(value) or not -1 <= value <= 1 for value in values):
            self._needs_reset = True
            try:
                self._request("stop")
            except Exception:
                if gvf:
                    self.close()
                raise
            label = "GVF residual action must contain three" if gvf else "Policy action must contain four"
            raise ValueError(label + " finite values in [-1, 1]")
        try:
            result = self._request("step", action=values)
            if gvf:
                self._validate_gvf_vector(result)
        except Exception:
            self._needs_reset = True
            if gvf and not self._closed:
                try:
                    self._request("stop")
                except Exception:
                    self.close()
            raise
        self._needs_reset = result["terminated"] or result["truncated"]
        return result["vector"], result["reward"], result["terminated"], result["truncated"], result["info"]

    def report(self):
        return self._request("report")

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=3)
        for stream in (self._process.stdin, self._process.stdout):
            stream.close()
        if self._reader is not threading.current_thread():
            self._reader.join(timeout=1)

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception, traceback):
        self.close()
