"""Run with roboproof/gpu/.venv/Scripts/python.exe tests/roboproof_gpu_test.py."""

import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from roboproof.gpu.controller import Cascade, inverse, resolve_device, wrap
    from roboproof.gpu.simulation import Simulation, records
    torch.set_num_threads(1)


def fixture(payload):
    def safe(value):
        if isinstance(value, dict):
            return {key: safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [safe(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            return "NaN" if math.isnan(value) else "Infinity" if value > 0 else "-Infinity"
        return value

    encoded = json.dumps(safe(payload), allow_nan=False)
    result = subprocess.run([shutil.which("node"), str(ROOT / "roboproof/gpu/wasm_fixture.cjs")],
                            input=encoded, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def scenario(seed=1, environment=None, task=None):
    spec = json.loads((ROOT / "roboproof/scenario-spec.json").read_text())
    result = dict(schema_version=1, scenario_id=f"test-{seed}", random_seed=seed,
                  environment=spec["nominal"], task=spec["task"], controller=spec["controller"])
    result["environment"].update(environment or {})
    result["task"].update(task or {})
    return result


@unittest.skipIf(torch is None, "PyTorch unavailable; install CPU torch before claiming CPU validation")
class ControllerTests(unittest.TestCase):
    def test_wasm_batched_sequences(self):
        generator = random.Random(971)
        lanes = []
        for lane in range(6):
            steps = []
            for tick in range(85):
                reference = [0.2 * math.sin(tick / 20), -0.3, lane + tick * 0.009,
                             0.04 * math.sin(tick / 9), 0.06 * math.cos(tick / 8), 0.1]
                feedback = [generator.uniform(-0.1, 0.1) for _ in range(10)]
                dt = [0.01, 0.02, 0.005, 0.015, 0.03, 0.05][lane]
                if tick == 8:
                    feedback[6] = float("nan")
                if tick == 18:
                    reference[3] = float("inf")
                if tick == 30:
                    dt = 0.00001
                if tick == 40:
                    dt = 0.101
                if tick > 55:
                    reference[:2] = [100, -100]
                    feedback[6:] = [(-1 if lane % 2 else 1) * 0.9] * 4
                steps.append(dict(reference=reference, feedback=feedback, dt=dt,
                                  valid=tick != 50, reset=tick == 65))
            lanes.append(dict(steps=steps))
        expected = fixture(dict(lanes=lanes))["results"]
        controller = Cascade(len(lanes))
        worst = 0.0
        for tick in range(len(lanes[0]["steps"])):
            inputs = [lane["steps"][tick] for lane in lanes]
            if inputs[0]["reset"]:
                controller.reset()
            output = controller.step([step["reference"] for step in inputs],
                                     [step["feedback"] for step in inputs],
                                     [step["dt"] for step in inputs],
                                     [step["valid"] for step in inputs])
            self.assertEqual(output.valid.tolist(), [lane[tick]["valid"] for lane in expected])
            for name in ("volts", "targets"):
                wanted = torch.tensor([lane[tick][name] for lane in expected], dtype=torch.float64)
                worst = max(worst, (getattr(output, name) - wanted).abs().max().item())
                torch.testing.assert_close(getattr(output, name), wanted, atol=2e-9, rtol=2e-9)
        print(f"controller WASM parity: 510 lane-steps, max absolute error {worst:.3g}")

    def test_cache_and_600_cap(self):
        controller = Cascade(3)
        reference = torch.zeros((3, 6), dtype=torch.float64)
        feedback = torch.zeros((3, 10), dtype=torch.float64)
        output = controller.step(reference, feedback, [0.01, 0.01, 0.00001])
        self.assertEqual(output.valid.tolist(), [True, True, False])
        self.assertEqual(controller.solve_iterations[2].item(), 600)
        controller.step(reference, feedback, 0.01)
        self.assertEqual(controller.solve_count.tolist(), [1, 1, 2])
        reference[0, 2] = 0.019
        reference[1, 3] = 0.019
        controller.step(reference, feedback, 0.0104)
        self.assertEqual(controller.solve_count.tolist(), [1, 1, 2])
        reference[0, 2] = 0.021
        reference[1, 3] = 0.021
        controller.step(reference, feedback, [0.0104, 0.0104, 0.0106])
        self.assertEqual(controller.solve_count.tolist(), [2, 2, 3])
        controller.configure(kV=12)
        self.assertFalse(controller.gain_valid.any())

    def test_derivative_filter_and_reset(self):
        controller = Cascade(2, config={"wheelKd": [0.0, 0.5]})
        reference = torch.zeros((2, 6), dtype=torch.float64)
        feedback = torch.zeros((2, 10), dtype=torch.float64)
        controller.step(reference, feedback, 0.01)
        feedback[:, 6:] = 0.1
        output = controller.step(reference, feedback, 0.01)
        derivative = 10 * (1 - math.exp(-0.01 / 0.03))
        torch.testing.assert_close(controller.derivative, torch.full((2, 4), derivative, dtype=torch.float64))
        torch.testing.assert_close(output.volts[0] - output.volts[1], torch.full((4,), 0.5 * derivative, dtype=torch.float64))
        feedback[0, 6] = float("nan")
        output = controller.step(reference, feedback, 0.01)
        self.assertEqual(output.valid.tolist(), [False, True])
        self.assertEqual(controller.first.tolist(), [True, False])
        self.assertEqual(controller.gain_valid.tolist(), [True, True])
        self.assertEqual(controller.derivative[0].tolist(), [0] * 4)

    def test_inverse_and_wrap(self):
        matrix = torch.tensor([[[0., 2, 1], [3, 0, 4], [2, 1, 0]],
                               [[1e-13, 0, 0], [0, 1, 0], [0, 0, 1]]], dtype=torch.float64)
        inverted, valid = inverse(matrix)
        self.assertEqual(valid.tolist(), [True, False])
        torch.testing.assert_close(inverted[0], torch.linalg.inv(matrix[0]))
        angles = torch.tensor([math.pi, -math.pi, 3 * math.pi, -3 * math.pi], dtype=torch.float64)
        self.assertEqual(wrap(angles).tolist(), [math.pi, -math.pi, -math.pi, math.pi])

    def test_device_guard_rejects_nvidia(self):
        with patch.object(torch.version, "hip", None), patch.object(torch.cuda, "is_available", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "AMD ROCm"):
                resolve_device("rocm")
        with self.assertRaises(ValueError):
            resolve_device("cuda")

    @unittest.skipUnless(torch is not None and torch.version.hip and torch.cuda.is_available(),
                         "Genuine AMD ROCm/HIP device unavailable; no GPU parity measured")
    def test_rocm_cpu_controller_parity(self):
        cpu = Cascade(8)
        gpu = Cascade(8, device="rocm")
        generator = torch.Generator().manual_seed(831)
        for tick in range(40):
            reference = torch.randn((8, 6), generator=generator, dtype=torch.float64) * 0.1
            feedback = torch.randn((8, 10), generator=generator, dtype=torch.float64) * 0.03
            expected = cpu.step(reference, feedback, 0.01)
            actual = gpu.step(reference, feedback, 0.01)
            torch.testing.assert_close(actual.volts.cpu(), expected.volts, atol=2e-8, rtol=2e-8)
            torch.testing.assert_close(actual.targets.cpu(), expected.targets, atol=2e-8, rtol=2e-8)
            self.assertEqual(actual.valid.cpu().tolist(), expected.valid.tolist())


@unittest.skipIf(torch is None, "PyTorch unavailable; CPU simulation parity not measured")
class SimulationTests(unittest.TestCase):
    def test_full_scalar_telemetry_and_metrics(self):
        scenarios = [scenario(task={"duration": 5}),
                     scenario(0xffffffff, environment={"friction": 0.08, "battery_voltage": 9,
                              "mass": 9.8, "inertia": 0.21, "radius": 0.205,
                              "max_wheel_speed": 0.76, "motor_tau": 0.17,
                              "lateral_friction": 0.08, "traction_stiffness": 115,
                              "left_motor_efficiency": 0.82, "right_motor_efficiency": 0.94,
                              "encoder_noise": 0.004, "encoder_bias": -0.001, "encoder_drift": 0.0001,
                              "encoder_period": 0.031, "encoder_latency": 0.023,
                              "encoder_dropout": 0.05, "encoder_quantization": 0.001,
                              "imu_noise": 0.006, "imu_bias": 0.02, "imu_drift": -0.0008,
                              "imu_period": 0.023, "imu_latency": 0.057, "imu_dropout": 0.05,
                              "imu_quantization": 0.0015, "control_latency": 0.027,
                              "initial_x_error": -0.01, "initial_y_error": 0.02,
                              "initial_heading_error": 0.03},
                              task={"duration": 5, "localization": "encoders-imu"}),
                     scenario(19, environment={"friction": 0.1, "encoder_noise": 0.003},
                              task={"duration": 5, "localization": "ground-truth-baseline"}),
                     scenario(83, task={"duration": 5, "start": [0.1, -0.2, 3.13],
                                      "goal": [-0.3, 0.2, -3.13]})]
        expected = fixture(dict(scenarios=scenarios))
        result = Simulation(scenarios).run(telemetry=True)
        actual = records(result)
        worst = 0.0
        for key, tensor in result["telemetry"].items():
            wanted = torch.tensor([[item["telemetry"][step][key] for item in expected]
                                   for step in range(500)], dtype=torch.float64)
            worst = max(worst, (tensor - wanted).abs().max().item())
            torch.testing.assert_close(tensor.cpu(), wanted, atol=2e-8, rtol=2e-8, msg=key)
        for got, wanted in zip(actual, expected):
            self.assertEqual(got["categories"], wanted["categories"])
            self.assertEqual(got["passed"], wanted["passed"])
            for key, value in got["metrics"].items():
                if value is None or wanted["metrics"][key] is None:
                    self.assertEqual(value, wanted["metrics"][key], key)
                else:
                    self.assertAlmostEqual(value, wanted["metrics"][key], delta=2e-8, msg=key)
            self.assertAlmostEqual(got["failure_score"], wanted["failure_score"], delta=2e-7)
            torch.testing.assert_close(torch.tensor(got["final_truth"]), torch.tensor(wanted["final_truth"]))
            torch.testing.assert_close(torch.tensor(got["final_estimate"]), torch.tensor(wanted["final_estimate"]))
        print(f"simulation JS/WASM parity: 4 scenarios, 2000 lane-steps, all telemetry fields; max absolute error {worst:.3g}")

    def test_replay_and_lane_isolation(self):
        inputs = [scenario(27, environment={"encoder_noise": 0.002, "imu_noise": 0.008},
                           task={"duration": 0.2, "dt": 0.02, "localization": "encoders-imu"})] * 2
        first = Simulation(inputs).run(telemetry=True)
        second = Simulation(inputs).run(telemetry=True)
        for key in first["telemetry"]:
            self.assertTrue(torch.equal(first["telemetry"][key], second["telemetry"][key]), key)
            self.assertTrue(torch.equal(first["telemetry"][key][:, 0], first["telemetry"][key][:, 1]), key)
        expected = fixture(dict(scenarios=inputs[:1]))[0]
        for key, tensor in first["telemetry"].items():
            wanted = torch.tensor([frame[key] for frame in expected["telemetry"]], dtype=torch.float64)
            torch.testing.assert_close(tensor[:, 0], wanted, atol=2e-8, rtol=2e-8)

    def test_schema_and_mixed_timestep_rejected(self):
        with self.assertRaisesRegex(ValueError, "common task.dt"):
            Simulation([scenario(), scenario(task={"dt": 0.02})])
        invalid = scenario(environment={"friction": -1})
        with self.assertRaisesRegex(ValueError, "friction"):
            Simulation([invalid])

    @unittest.skipUnless(torch is not None and torch.version.hip and torch.cuda.is_available(),
                         "Genuine AMD ROCm/HIP device unavailable; no GPU simulation parity measured")
    def test_rocm_cpu_simulation_parity(self):
        scenarios = [scenario(33, environment={"encoder_noise": 0.003, "control_latency": 0.02},
                              task={"duration": 0.5}), scenario(61, task={"duration": 0.5})]
        cpu = Simulation(scenarios).run(telemetry=True)
        gpu = Simulation(scenarios, device="rocm").run(telemetry=True)
        for key in cpu["telemetry"]:
            torch.testing.assert_close(gpu["telemetry"][key].cpu(), cpu["telemetry"][key], atol=2e-8, rtol=2e-8)


@unittest.skipIf(torch is None, "PyTorch unavailable; benchmark not measured")
class BenchmarkTests(unittest.TestCase):
    def test_measured_cpu_report(self):
        from roboproof.gpu.benchmark import benchmark, load_scenarios
        inputs = [scenario(task={"duration": 0.02})]
        with patch.object(Path, "read_bytes", return_value=json.dumps(dict(scenarios=inputs)).encode()):
            loaded, source = load_scenarios(ROOT / "roboproof/gpu/test-input.json", 2)
        self.assertEqual(len(loaded), 2)
        self.assertTrue(source["cycling_input"])
        report = benchmark(loaded, repeats=2, warmup=0)
        self.assertEqual(report["backend"], "torch-cpu")
        self.assertFalse(report["gpu_measured"])
        self.assertIsNone(report["hardware"]["gpu"])
        self.assertTrue(report["hardware"]["cpu"])
        self.assertTrue(report["determinism"]["repeated_results_exact"])
        self.assertEqual(report["workload"]["executed_control_steps"], 4)
        self.assertEqual(len(report["timing"]["seconds"]), 2)
        self.assertGreater(report["timing"]["median_seconds"], 0)
        self.assertGreater(report["timing"]["simulations_per_second"], 0)
        json.dumps(report, allow_nan=False)

    @unittest.skipIf(torch is not None and torch.version.hip and torch.cuda.is_available(),
                     "Unavailable-ROCm CLI path only applies when HIP is absent")
    def test_rocm_cli_fails_without_fake_report(self):
        result = subprocess.run([sys.executable, "-m", "roboproof.gpu.benchmark", "--device", "rocm",
                                 "--scenarios", "does-not-need-to-exist.json"], cwd=ROOT,
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertIn("AMD ROCm requested", result.stderr)
        self.assertIn("no GPU measurement was performed", result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
