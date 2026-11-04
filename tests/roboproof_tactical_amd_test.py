import contextlib
import subprocess
import types
import unittest
from unittest import mock

import torch

from roboproof.tactical_learning import train
from roboproof.tactical_learning.model import (FEATURE_COUNT, FEATURE_COUNT_V3, TacticalActorCritic,
                                               parameter_count)


def rocm_torch(name, architecture, memory_bytes, hip="6.4.43483", devices=1):
    properties = types.SimpleNamespace(name=name, gcnArchName=architecture, total_memory=memory_bytes, major=9, minor=4)
    cuda_module = types.SimpleNamespace(is_available=lambda: True, device_count=lambda: devices,
                                        get_device_properties=lambda index: properties)
    return types.SimpleNamespace(version=types.SimpleNamespace(cuda=None, hip=hip), cuda=cuda_module,
                                 __version__="2.8.0+rocm6.4")


@contextlib.contextmanager
def remote_runtime(torch_module, system="Linux", node="/usr/bin/node"):
    completed = subprocess.CompletedProcess(["node", "--version"], 0, stdout="v20.19.0\n", stderr="")
    with mock.patch.object(train, "torch", torch_module), \
            mock.patch.object(train.platform, "system", return_value=system), \
            mock.patch.object(train.shutil, "which", return_value=node), \
            mock.patch.object(train.subprocess, "run", return_value=completed):
        yield


MI300X = ("AMD Instinct MI300X", "gfx942:sramecc+:xnack-", 205_822_885_888)


class TacticalAmdRuntimeTests(unittest.TestCase):
    def test_an_actual_mi300x_is_accepted_and_recorded(self):
        with remote_runtime(rocm_torch(*MI300X)):
            runtime = train.validate_runtime(False, approved_amd_mi300x=True)
        self.assertEqual(runtime["provider"], "amd-developer-cloud")
        self.assertEqual(runtime["gpu"], "AMD Instinct MI300X")
        self.assertTrue(runtime["gcnArchName"].startswith("gfx942"))
        self.assertEqual(runtime["hip"], "6.4.43483")
        self.assertIs(runtime["providerBillingVerified"], False)

    def test_other_devices_and_runtimes_are_refused(self):
        refusals = [
            (rocm_torch("AMD Instinct MI250X", "gfx90a", 68_702_699_520), {}, "MI300X"),
            (rocm_torch("AMD Instinct MI300X", "gfx942", 64 * 1024 ** 3), {}, "MI300X"),
            (rocm_torch(*MI300X, hip=None), {}, "ROCm"),
            (rocm_torch(*MI300X), {"system": "Windows"}, "ROCm"),
            (rocm_torch(*MI300X), {"node": None}, "Node.js"),
        ]
        for module, environment, message in refusals:
            with self.subTest(message=message), remote_runtime(module, **environment):
                with self.assertRaisesRegex(RuntimeError, message):
                    train.validate_runtime(False, approved_amd_mi300x=True)

    def test_only_one_runtime_and_no_trainer_side_cost_ceiling(self):
        with remote_runtime(rocm_torch(*MI300X)):
            with self.assertRaisesRegex(RuntimeError, "exactly one approved runtime"):
                train.validate_runtime(True, approved_amd_mi300x=True)
            with self.assertRaisesRegex(RuntimeError, "campaign"):
                train.validate_runtime(False, approved_max_cost_usd=5.0, approved_amd_mi300x=True)


class TacticalModelArchitectureTests(unittest.TestCase):
    def test_the_original_two_layer_model_is_unchanged(self):
        model = TacticalActorCritic()
        self.assertEqual(list(model.state_dict())[:4], ["encoder.0.weight", "encoder.0.bias", "encoder.2.weight", "encoder.2.bias"])
        self.assertEqual(model.architecture()["parameters"], 28_034)
        self.assertFalse(model.residual)

    def test_deep_models_use_residual_blocks_and_match_the_parameter_formula(self):
        for features, width, layers in ((FEATURE_COUNT_V3, 64, 10), (FEATURE_COUNT, 32, 4)):
            model = TacticalActorCritic(features, width, layers)
            self.assertTrue(model.residual)
            self.assertEqual(model.architecture()["parameters"], parameter_count(features, width, layers))
            logits, values = model(torch.zeros(2, 5, features), torch.ones(2, 5, dtype=torch.bool))
            self.assertEqual(tuple(logits.shape), (2, 5))
            self.assertEqual(tuple(values.shape), (2,))
        # The planned MI300X model: 1024 wide, 10 hidden layers, v3 features (counted, not built here).
        self.assertEqual(parameter_count(FEATURE_COUNT_V3, 1024, 10), 10_022_914)

    def test_dimensions_are_bounded(self):
        for arguments in ((FEATURE_COUNT, 2048, 2), (FEATURE_COUNT, 128, 11), (FEATURE_COUNT, 128, 1), (24, 128, 2)):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                TacticalActorCritic(*arguments)


if __name__ == "__main__":
    unittest.main()
