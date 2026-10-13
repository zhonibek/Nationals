import copy
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from roboproof.motion_learning.execution import prepare_execution, snapshot_cuda_rng, restore_cuda_rng
from roboproof.motion_learning.td3_train import configuration, compatible_config


class MotionExecutionTests(unittest.TestCase):
    def test_cpu_preflight_and_rng_do_not_probe_cuda_or_consume_randomness(self):
        torch.manual_seed(190001)
        before = torch.get_rng_state().clone()
        with patch.object(torch.cuda, "is_available", side_effect=AssertionError("CUDA must not be probed")):
            device, identity = prepare_execution("cpu")
            self.assertEqual(device, torch.device("cpu"))
            self.assertEqual(identity["device"], "cpu")
            self.assertIsNone(snapshot_cuda_rng(device))
            restore_cuda_rng(None, device)
        torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)
        with self.assertRaises(ValueError):
            restore_cuda_rng(torch.zeros(8, dtype=torch.uint8), device)

    def test_cuda_unavailable_and_multiple_devices_fail_without_fallback(self):
        with patch.object(torch.version, "hip", None), patch.object(torch.version, "cuda", None):
            with self.assertRaisesRegex(ValueError, "no CPU fallback"):
                prepare_execution("cuda")
        with patch.object(torch.version, "hip", None), patch.object(torch.version, "cuda", "13.0"), \
                patch.object(torch.cuda, "is_available", return_value=True), patch.object(torch.cuda, "device_count", return_value=2):
            with self.assertRaisesRegex(ValueError, "exactly one"):
                prepare_execution("cuda")
        for value in (None, True, "cuda:0", "rocm", "automatic"):
            with self.assertRaises(ValueError):
                prepare_execution(value)

    def test_mock_metadata_binds_capability_and_rng_but_is_not_gpu_execution(self):
        properties = SimpleNamespace(name="Mock NVIDIA L4", major=8, minor=9, total_memory=24 * 1024 ** 3)
        with patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}), \
                patch.object(torch.version, "hip", None), patch.object(torch.version, "cuda", "13.0"), \
                patch.object(torch.cuda, "is_available", return_value=True), patch.object(torch.cuda, "device_count", return_value=1), \
                patch.object(torch.cuda, "get_device_properties", return_value=properties), \
                patch.object(torch.backends.cuda, "matmul", SimpleNamespace(allow_tf32=True)), \
                patch.object(torch.backends, "cudnn", SimpleNamespace(allow_tf32=True)):
            device, identity = prepare_execution("cuda")
        self.assertEqual(device, torch.device("cuda:0"))
        self.assertEqual(identity["capability"], [8, 9])
        self.assertEqual(identity["memoryBytes"], 24 * 1024 ** 3)
        self.assertIn("require actual training", identity["scope"])
        expected = torch.arange(16, dtype=torch.uint8)
        with patch.object(torch.cuda, "get_rng_state", return_value=expected), patch.object(torch.cuda, "set_rng_state") as restore:
            saved = snapshot_cuda_rng(device)
            expected.zero_()
            self.assertFalse(torch.equal(saved, expected))
            restore_cuda_rng(saved, device)
            torch.testing.assert_close(restore.call_args.args[0], saved, rtol=0, atol=0)
            with self.assertRaises(ValueError):
                restore_cuda_rng(saved.float(), device)
            self.assertEqual(restore.call_count, 1)

    def test_width_and_device_configuration_cannot_change_on_resume(self):
        parameters = (201, 1, 64, 50000, 256, 8, "deadline-context-v1", "finite-total-return-settle-margin-v4",
                      "mixed-full-reach-v5", 32, 1024, 0.05, 1, "public-deadline-residual-v1", 4)
        baseline = configuration(*parameters)
        wider = configuration(*parameters, network_width=128)
        gpu = configuration(*parameters, network_width=128, training_device="cuda")
        self.assertEqual(baseline["networkWidth"], 32)
        self.assertEqual(wider["networkWidth"], 128)
        self.assertEqual(gpu["trainingDevice"], "cuda")
        self.assertFalse(compatible_config(baseline, wider))
        self.assertFalse(compatible_config(wider, gpu))
        largest = configuration(*parameters, network_width=256)
        self.assertEqual(largest["networkWidth"], 256)
        self.assertFalse(compatible_config(wider, largest))
        extended = copy.deepcopy(gpu)
        extended["updates"] = 80
        self.assertTrue(compatible_config(gpu, extended))
        for width in (True, 32.0, 129, 512):
            with self.assertRaises(ValueError):
                configuration(*parameters, network_width=width)


if __name__ == "__main__":
    unittest.main()
