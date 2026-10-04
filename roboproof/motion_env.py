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

    def __init__(self, repository=None, node_executable=None, timeout_seconds=30):
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
        self._sequence = 0
        self._closed = False
        self._needs_reset = True
        self._responses = queue.Queue()
        self._process = subprocess.Popen(
            [str(executable), str(bridge)], cwd=str(root), stdin=subprocess.PIPE,
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
            if self.contract["actionSize"] != 4 or self.contract["observationSize"] != 34:
                raise RuntimeError("Unsupported motion observation/action contract")
        except Exception:
            self.close()
            raise

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
        request = {"seed": seed, "configuration": configuration or {}, "options": options or {}}
        if task is not None:
            request["task"] = task
        result = self._request("reset", **request)
        self._needs_reset = False
        return result["vector"], {"identity": result["identity"], "options": result["options"]}

    def step(self, action):
        if self._needs_reset:
            raise RuntimeError("Reset before stepping, including after an episode ends")
        try:
            values = list(action)
        except TypeError:
            values = []
        if len(values) != 4 or any(isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not math.isfinite(value) or not -1 <= value <= 1 for value in values):
            self._request("stop")
            self._needs_reset = True
            raise ValueError("Policy action must contain four finite values in [-1, 1]")
        result = self._request("step", action=values)
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
