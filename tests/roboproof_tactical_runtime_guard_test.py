import contextlib
import io
import subprocess
import types
import unittest
from unittest import mock

from roboproof.tactical_learning import train


T4_KEYS = {"platform", "python", "torch", "cuda", "gpu", "capability", "memoryBytes", "node", "device"}


def fake_torch(name, capability, memory_bytes, cuda="12.8", hip=None, devices=1):
    properties = types.SimpleNamespace(name=name, major=capability[0], minor=capability[1], total_memory=memory_bytes)
    cuda_module = types.SimpleNamespace(is_available=lambda: True, device_count=lambda: devices,
                                        get_device_properties=lambda index: properties)
    return types.SimpleNamespace(version=types.SimpleNamespace(cuda=cuda, hip=hip), cuda=cuda_module,
                                 __version__="2.8.0+cu128")


@contextlib.contextmanager
def remote_runtime(torch_module, system="Linux", node="/usr/local/bin/node"):
    completed = subprocess.CompletedProcess(["node", "--version"], 0, stdout="v20.19.0\n", stderr="")
    with mock.patch.object(train, "torch", torch_module), \
            mock.patch.object(train.platform, "system", return_value=system), \
            mock.patch.object(train.shutil, "which", return_value=node), \
            mock.patch.object(train.subprocess, "run", return_value=completed):
        yield


L40S = ("NVIDIA L40S", (8, 9), 47_697_362_944)
T4 = ("Tesla T4", (7, 5), 15_642_329_088)


class TacticalRuntimeGuardTests(unittest.TestCase):
    def test_an_explicit_opt_in_is_required(self):
        with remote_runtime(fake_torch(*T4)):
            with self.assertRaisesRegex(RuntimeError, "opt-in is required"):
                train.validate_runtime(False)

    def test_free_t4_path_is_unchanged(self):
        with remote_runtime(fake_torch(*T4)):
            runtime = train.validate_runtime(True)
        self.assertEqual(set(runtime), T4_KEYS)
        self.assertEqual((runtime["gpu"], runtime["capability"], runtime["node"]), ("Tesla T4", [7, 5], "v20.19.0"))
        with remote_runtime(fake_torch(*L40S)):
            with self.assertRaisesRegex(RuntimeError, "Tesla T4"):
                train.validate_runtime(True)
        with remote_runtime(fake_torch(*T4)):
            with self.assertRaisesRegex(RuntimeError, "only to an approved Nebius"):
                train.validate_runtime(True, approved_max_cost_usd=2.0)

    def test_nebius_l40s_opt_in_records_provider_and_unverified_billing(self):
        with remote_runtime(fake_torch(*L40S)):
            runtime = train.validate_runtime(False, "L40S", 2.0)
        self.assertTrue(T4_KEYS < set(runtime))
        self.assertEqual(runtime["provider"], "nebius-serverless-ai-job")
        self.assertEqual((runtime["approvedGpu"], runtime["approvedMaxCostUsd"]), ("L40S", 2.0))
        self.assertIs(runtime["providerBillingVerified"], False)
        self.assertEqual(runtime["capability"], [8, 9])

    def test_nebius_opt_in_rejects_other_devices_and_runtimes(self):
        cases = {
            "free T4 device": (fake_torch(*T4), "Linux"),
            "L40 without S": (fake_torch("NVIDIA L40", (8, 9), 47_697_362_944), "Linux"),
            "wrong capability": (fake_torch("NVIDIA L40S", (8, 6), 47_697_362_944), "Linux"),
            "too little memory": (fake_torch("NVIDIA L40S", (8, 9), 24 * 1024 ** 3), "Linux"),
            "ROCm build": (fake_torch(*L40S, hip="6.4"), "Linux"),
            "CPU build": (fake_torch(*L40S, cuda=None), "Linux"),
            "two GPUs": (fake_torch(*L40S, devices=2), "Linux"),
            "laptop": (fake_torch(*L40S), "Windows"),
        }
        for label, (torch_module, system) in cases.items():
            with self.subTest(label), remote_runtime(torch_module, system), self.assertRaises(RuntimeError):
                train.validate_runtime(False, "L40S", 2.0)
        with remote_runtime(fake_torch(*L40S), node=None), self.assertRaisesRegex(RuntimeError, "Node.js"):
            train.validate_runtime(False, "L40S", 2.0)

    def test_nebius_opt_in_requires_a_bounded_cost_and_a_single_known_gpu(self):
        with remote_runtime(fake_torch(*L40S)):
            for cost in (None, 0, -1.0, float("nan"), float("inf"), True, "2", train.MAX_APPROVED_NEBIUS_COST_USD + 0.01):
                with self.subTest(cost=cost), self.assertRaisesRegex(RuntimeError, "approved maximum cost"):
                    train.validate_runtime(False, "L40S", cost)
            with self.assertRaisesRegex(RuntimeError, "Unapproved Nebius GPU"):
                train.validate_runtime(False, "H100", 2.0)
            with self.assertRaisesRegex(RuntimeError, "exactly one approved runtime"):
                train.validate_runtime(True, "L40S", 2.0)

    def test_command_line_opt_ins_are_mutually_exclusive(self):
        base = ["train", "--seed", "731201", "--evaluation-episodes", "8"]
        with mock.patch.object(train.sys, "argv", base + ["--approved-nebius-gpu", "L40S", "--approved-max-cost-usd", "2"]):
            args = train.parse_args()
        self.assertEqual((args.approved_free_t4, args.approved_nebius_gpu, args.approved_max_cost_usd), (False, "L40S", 2.0))
        for extra in (["--approved-free-t4", "--approved-nebius-gpu", "L40S"], ["--approved-nebius-gpu", "T4"]):
            with self.subTest(extra=extra), mock.patch.object(train.sys, "argv", base + extra), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                train.parse_args()


if __name__ == "__main__":
    unittest.main()
