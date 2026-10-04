"""Tensor translation of roboproof/sim.js; no per-environment stepping loop."""

import json
import math
from pathlib import Path

import torch

from .controller import Cascade

SPEC_PATH = Path(__file__).resolve().parents[1] / "scenario-spec.json"


def angle_wrap(angle):
    return torch.atan2(torch.sin(angle), torch.cos(angle))


class RandomStream:
    def __init__(self, seeds, device):
        self.state = torch.tensor(seeds, dtype=torch.int64, device=device)

    def uniform(self):
        self.state = (1664525 * self.state + 1013904223) & 0xffffffff
        return (self.state.to(torch.float64) + 0.5) / 4294967296

    def normal(self):
        return torch.sqrt(-2 * torch.log(self.uniform())) * torch.cos(2 * math.pi * self.uniform())


def validate(scenario):
    spec = json.loads(SPEC_PATH.read_text())
    if scenario.get("schema_version") != 1 or not isinstance(scenario.get("scenario_id"), str) or not scenario["scenario_id"]:
        raise ValueError("Unsupported schema or missing scenario_id")

    def finite(value, low, high, name):
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{name} must be finite in [{low}, {high}]")

    seed = scenario["random_seed"]
    finite(seed, 0, 0xffffffff, "random_seed")
    if int(seed) != seed:
        raise ValueError("random_seed must be an integer")
    environment, task, controller = (scenario[key] for key in ("environment", "task", "controller"))
    for name, bounds in spec["ranges"].items():
        finite(environment.get(name), *bounds, name)
    if set(environment) - set(spec["nominal"]):
        raise ValueError("Unknown environment parameter")
    for name in ("start", "goal"):
        if len(task[name]) != 3:
            raise ValueError(f"Invalid {name}")
        for value in task[name]:
            finite(value, -100, 100, name)
    finite(task["dt"], 0.005, 0.05, "dt")
    finite(task["duration"], task["dt"], 60, "duration")
    if task["localization"] not in ("encoders", "encoders-imu", "ground-truth-baseline"):
        raise ValueError("Unknown localization mode")
    for name in spec["task"]["thresholds"]:
        finite(task["thresholds"].get(name), 0.000001, 1000, name)
    finite(controller["radius"], 0.1, 0.3, "controller.radius")
    finite(controller["max_wheel_speed"], 0.1, 2, "controller.max_wheel_speed")
    finite(controller["reference_time_scale"], 1, 4, "reference_time_scale")
    if controller["adapter"] != "nationals-wasm":
        raise ValueError("Unsupported controller adapter")


class Simulation:
    def __init__(self, scenarios, device="cpu"):
        if not scenarios:
            raise ValueError("At least one scenario is required")
        for scenario in scenarios:
            validate(scenario)
        self.scenarios = scenarios
        self.batch = len(scenarios)
        self.controller = Cascade(self.batch, device=device)
        self.device = self.controller.device
        self.dt = scenarios[0]["task"]["dt"]
        self.duration = scenarios[0]["task"]["duration"]
        if any(scenario["task"]["dt"] != self.dt or scenario["task"]["duration"] != self.duration for scenario in scenarios):
            raise ValueError("A tensor batch requires common task.dt and task.duration; group scenarios first")
        self.steps = math.ceil(self.duration / self.dt)
        spec = json.loads(SPEC_PATH.read_text())
        self.environment = {key: self.tensor([scenario["environment"][key] for scenario in scenarios]) for key in spec["nominal"]}
        self.thresholds = {key: self.tensor([scenario["task"]["thresholds"][key] for scenario in scenarios]) for key in spec["task"]["thresholds"]}
        self.start = self.tensor([scenario["task"]["start"] for scenario in scenarios])
        self.goal = self.tensor([scenario["task"]["goal"] for scenario in scenarios])
        self.imu_mode = self.tensor([scenario["task"]["localization"] == "encoders-imu" for scenario in scenarios]).bool()
        self.truth_mode = self.tensor([scenario["task"]["localization"] == "ground-truth-baseline" for scenario in scenarios]).bool()
        self.radius = self.tensor([scenario["controller"]["radius"] for scenario in scenarios])
        speed = self.tensor([scenario["controller"]["max_wheel_speed"] for scenario in scenarios])
        self.controller.configure(radius=self.radius, maxWheelSpeed=speed, kV=12 / speed)
        distance = torch.hypot(self.goal[:, 0] - self.start[:, 0], self.goal[:, 1] - self.start[:, 1])
        from .controller import wrap
        self.heading_delta = wrap(self.goal[:, 2] - self.start[:, 2])
        self.reference_duration = torch.stack((torch.full_like(distance, 0.25), 1.875 * distance / 0.45,
                                              torch.sqrt(5.774 * distance / 0.8), (60 * distance / 3.5).pow(1 / 3),
                                              1.875 * self.heading_delta.abs() / 1.5), -1).amax(-1)
        self.reference_duration *= self.tensor([scenario["controller"]["reference_time_scale"] for scenario in scenarios])
        self.encoder_random = RandomStream([int(scenario["random_seed"]) ^ 0x85ebca6b for scenario in scenarios], self.device)
        self.imu_random = RandomStream([int(scenario["random_seed"]) ^ 0xc2b2ae35 for scenario in scenarios], self.device)
        self.pose = self.start + torch.stack([self.environment[key] for key in ("initial_x_error", "initial_y_error", "initial_heading_error")], -1)
        self.velocity = self.zeros(3)
        self.wheel = self.zeros(4)
        self.estimate = self.start.clone()
        self.estimated_velocity = self.zeros(3)
        self.encoders = self.zeros(4)
        self.imu = self.start[:, 2].clone()
        self.delay = {name: torch.ceil(self.environment[name + "_latency"] / self.dt).long() for name in ("encoder", "imu", "control")}
        self.period = {name: torch.ceil(self.environment[name + "_period"] / self.dt).clamp_min(1).long() for name in ("encoder", "imu")}
        self.ring_size = int(torch.stack(list(self.delay.values())).max().item()) + 1
        self.history_wheel = self.zeros(self.ring_size, 4).transpose(0, 1).contiguous()
        self.history_heading = self.zeros(self.ring_size).T.contiguous()
        self.history_heading[0] = self.pose[:, 2]
        self.commands = self.zeros(self.ring_size, 4).transpose(0, 1).contiguous()
        self.lanes = torch.arange(self.batch, device=self.device)
        self.active = torch.ones(self.batch, dtype=torch.bool, device=self.device)
        self.used = False

    def tensor(self, value):
        return torch.as_tensor(value, dtype=torch.float64, device=self.device)

    def zeros(self, *shape):
        return torch.zeros((self.batch, *shape), dtype=torch.float64, device=self.device)

    def reference(self, time):
        progress = (time / self.reference_duration).clamp(0, 1)
        square = progress * progress
        cube = square * progress
        fourth = cube * progress
        fifth = fourth * progress
        position = 10 * cube - 15 * fourth + 6 * fifth
        velocity = (30 * square - 60 * cube + 30 * fourth) / self.reference_duration
        delta = self.goal - self.start
        delta[:, 2] = self.heading_delta
        return torch.cat((self.start + delta * position[:, None], delta * velocity[:, None]), -1)

    def measure(self, raw, name, time, stream):
        environment = self.environment
        noisy = raw + environment[name + "_bias"] + environment[name + "_drift"] * time + environment[name + "_noise"] * stream.normal()
        quantum = environment[name + "_quantization"]
        return torch.where(quantum > 0, torch.floor(noisy / torch.where(quantum > 0, quantum, 1) + 0.5) * quantum, noisy)

    def sense(self, step):
        time = step * self.dt
        past_encoder = self.history_wheel[(step - self.delay["encoder"]).clamp_min(0) % self.ring_size, self.lanes]
        past_heading = self.history_heading[(step - self.delay["imu"]).clamp_min(0) % self.ring_size, self.lanes]
        reported = torch.stack([self.measure(past_encoder[:, wheel], "encoder", time, self.encoder_random) for wheel in range(4)], -1)
        heading = angle_wrap(self.measure(past_heading, "imu", time, self.imu_random))
        encoder_available = self.encoder_random.uniform() >= self.environment["encoder_dropout"]
        imu_available = self.imu_random.uniform() >= self.environment["imu_dropout"]
        self.encoders = torch.where(((step % self.period["encoder"] == 0) & encoder_available)[:, None], reported, self.encoders)
        old_heading = self.estimate[:, 2].clone()
        front_left, back_left, front_right, back_right = self.encoders.unbind(-1)
        diagonal = math.sqrt(0.5) / 2
        strafe = (front_left - back_left - front_right + back_right) * diagonal
        forward = (front_left + back_left + front_right + back_right) * diagonal
        omega = (front_left + back_left - front_right - back_right) / (4 * self.radius)
        if step > 0:
            self.imu = torch.where((step % self.period["imu"] == 0) & imu_available & self.imu_mode, heading, self.imu)
            self.estimate[:, 2] = torch.where(self.imu_mode, self.imu, angle_wrap(old_heading + omega * self.dt))
            midpoint = old_heading + angle_wrap(self.estimate[:, 2] - old_heading) / 2
            self.estimated_velocity = torch.stack((strafe * midpoint.cos() + forward * midpoint.sin(),
                                                   -strafe * midpoint.sin() + forward * midpoint.cos(),
                                                   angle_wrap(self.estimate[:, 2] - old_heading) / self.dt), -1)
            self.estimate[:, :2] += self.estimated_velocity[:, :2] * self.dt
        self.estimate = torch.where(self.truth_mode[:, None], self.pose, self.estimate)
        self.estimated_velocity = torch.where(self.truth_mode[:, None], self.velocity, self.estimated_velocity)

    def physics(self, voltage):
        environment = self.environment
        substeps = math.ceil(self.dt / 0.001)
        interval = self.dt / substeps
        direction = self.tensor([1, -1, -1, 1])
        rotation = self.tensor([1, 1, -1, -1])
        efficiency = torch.stack((environment["left_motor_efficiency"], environment["left_motor_efficiency"], environment["right_motor_efficiency"], environment["right_motor_efficiency"]), -1)
        battery = environment["battery_voltage"][:, None]
        target = voltage.clamp(-battery, battery) / 12 * environment["max_wheel_speed"][:, None] * efficiency
        tau = environment["motor_tau"][:, None]
        stiffness = environment["traction_stiffness"][:, None]
        coupling = stiffness / 0.6
        limit = (environment["friction"] * environment["mass"] * 9.81 / 4)[:, None]
        drag = environment["lateral_friction"] * environment["mass"] * 9.81
        for substep in range(substeps):
            cosine, sine = self.pose[:, 2].cos(), self.pose[:, 2].sin()
            strafe = self.velocity[:, 0] * cosine - self.velocity[:, 1] * sine
            forward = self.velocity[:, 0] * sine + self.velocity[:, 1] * cosine
            contact = (forward[:, None] + direction * strafe[:, None]) * math.sqrt(0.5) + rotation * environment["radius"][:, None] * self.velocity[:, 2, None]
            speed = (self.wheel + interval * (target / tau + coupling * contact)) / (1 + interval / tau + interval * coupling)
            requested = stiffness * (speed - contact)
            force = requested.clamp(-limit, limit)
            slipping = (self.wheel + interval * (target / tau - force / 0.6)) / (1 + interval / tau)
            self.wheel = torch.where(requested.abs() > limit, slipping, speed)
            front_left, back_left, front_right, back_right = force.unbind(-1)
            lateral = (front_left - back_left - front_right + back_right) * math.sqrt(0.5) - drag * torch.tanh(strafe / 0.05)
            longitudinal = (front_left + back_left + front_right + back_right) * math.sqrt(0.5) - drag * torch.tanh(forward / 0.05)
            torque = (front_left + back_left - front_right - back_right) * environment["radius"] - 0.05 * self.velocity[:, 2]
            self.velocity[:, 0] += (lateral * cosine + longitudinal * sine) / environment["mass"] * interval
            self.velocity[:, 1] += (-lateral * sine + longitudinal * cosine) / environment["mass"] * interval
            self.velocity[:, 2] += torque / environment["inertia"] * interval
            self.pose = self.pose + self.velocity * interval
            self.pose[:, 2] = angle_wrap(self.pose[:, 2])

    @torch.no_grad()
    def run(self, telemetry=False):
        if self.used:
            raise RuntimeError("Simulation instances are single-use; create a fresh instance for replay")
        self.used = True
        thresholds = self.thresholds
        maxima = {name: self.zeros() for name in ("max_path_deviation", "max_tracking_error", "max_heading_error", "max_localization_error")}
        saturated, oscillations, last_sign, streak, executed = (self.zeros() for _ in range(5))
        completion, divergence = (torch.full_like(streak, float("nan")) for _ in range(2))
        invalid = torch.zeros_like(self.active)
        frames = []
        for step in range(self.steps):
            time = step * self.dt
            old_estimate = self.estimate.clone()
            self.sense(step)
            self.estimate = torch.where(self.active[:, None], self.estimate, old_estimate)
            reference = self.reference(time)
            output = self.controller.step(reference, torch.cat((self.estimate, self.estimated_velocity, self.encoders), -1), self.dt)
            invalid |= self.active & ~output.valid
            self.commands[step % self.ring_size] = output.volts
            delayed = self.commands[(step - self.delay["control"]) % self.ring_size, self.lanes]
            applied = torch.where((step >= self.delay["control"])[:, None], delayed, 0)
            endpoint = torch.hypot(self.pose[:, 0] - self.goal[:, 0], self.pose[:, 1] - self.goal[:, 1])
            heading = angle_wrap(self.goal[:, 2] - self.pose[:, 2])
            delta = self.goal[:, :2] - self.start[:, :2]
            length_squared = delta.square().sum(-1)
            progress = (((self.pose[:, :2] - self.start[:, :2]) * delta).sum(-1) / torch.where(length_squared > 0, length_squared, 1)).clamp(0, 1)
            path_delta = self.pose[:, :2] - self.start[:, :2] - progress[:, None] * delta
            path_error = torch.hypot(path_delta[:, 0], path_delta[:, 1])
            tracking = torch.hypot(self.pose[:, 0] - reference[:, 0], self.pose[:, 1] - reference[:, 1])
            heading_error = angle_wrap(reference[:, 2] - self.pose[:, 2])
            localization = torch.hypot(self.pose[:, 0] - self.estimate[:, 0], self.pose[:, 1] - self.estimate[:, 1])
            values = (path_error, tracking, heading_error.abs(), localization)
            for name, value in zip(maxima, values):
                maxima[name] = torch.where(self.active, torch.maximum(maxima[name], value), maxima[name])
            saturated += self.active & (applied.abs() >= self.environment["battery_voltage"][:, None] - 0.01).any(-1)
            count_sign = self.active & (time >= self.reference_duration) & (heading.abs() > 0.02)
            oscillations += count_sign & (last_sign != 0) & (last_sign != heading.sign())
            last_sign = torch.where(count_sign, heading.sign(), last_sign)
            settled = self.active & (time >= self.reference_duration) & (endpoint <= thresholds["endpoint"]) & (heading.abs() <= thresholds["heading"])
            settled &= (torch.hypot(self.velocity[:, 0], self.velocity[:, 1]) <= thresholds["settle_speed"]) & (self.velocity[:, 2].abs() <= thresholds["settle_omega"])
            streak = torch.where(settled, streak + self.dt, 0)
            completion = torch.where((streak + 1e-12 >= thresholds["settle_seconds"]) & completion.isnan(), time, completion)
            divergence = torch.where(self.active & divergence.isnan() & ((path_error > thresholds["path"]) | (localization > thresholds["localization"])), time, divergence)
            if telemetry:
                frame = dict(time=torch.full_like(streak, time), reference=reference[:, :3], truth=self.pose,
                             estimate=self.estimate, velocity=self.velocity, wheel_speed=self.wheel,
                             measured_wheel=self.encoders, target_wheel=output.targets, command=output.volts,
                             applied=applied, path_error=path_error, tracking_error=tracking,
                             heading_error=heading_error, localization_error=localization)
                frames.append({key: value.clone() for key, value in frame.items()})
            executed += self.active
            self.physics(applied)
            self.active &= torch.isfinite(torch.cat((self.pose, self.velocity, self.wheel), -1)).all(-1)
            self.history_wheel[(step + 1) % self.ring_size] = self.wheel
            self.history_heading[(step + 1) % self.ring_size] = self.pose[:, 2]
        endpoint = torch.hypot(self.pose[:, 0] - self.goal[:, 0], self.pose[:, 1] - self.goal[:, 1])
        heading = angle_wrap(self.goal[:, 2] - self.pose[:, 2]).abs()
        metrics = dict(endpoint_error=torch.where(self.active, endpoint, float("nan")), **maxima,
                       final_heading_error=torch.where(self.active, heading, float("nan")),
                       saturation_fraction=saturated / executed, oscillations=oscillations,
                       completion_time=completion, first_divergence=divergence,
                       reference_duration=self.reference_duration, steps=executed)
        categories = {"NUMERICAL_FAILURE": ~self.active, "CONTROLLER_DIVERGENCE": invalid,
                      "PATH_DIVERGENCE": maxima["max_path_deviation"] > thresholds["path"],
                      "LOCALIZATION_DIVERGENCE": maxima["max_localization_error"] > thresholds["localization"],
                      "HEADING_INSTABILITY": self.active & (heading > thresholds["heading"]),
                      "OSCILLATION": oscillations >= thresholds["oscillations"],
                      "MOTOR_SATURATION": saturated / executed > thresholds["saturation_fraction"],
                      "ENDPOINT_FAILURE": self.active & (endpoint > thresholds["endpoint"]), "TIMEOUT": completion.isnan()}
        score = endpoint / thresholds["endpoint"] + maxima["max_path_deviation"] / thresholds["path"] + heading / thresholds["heading"] + maxima["max_localization_error"] / thresholds["localization"] + 2 * saturated / executed + oscillations / thresholds["oscillations"] + torch.where(completion.isnan(), 2, completion / self.duration)
        score = torch.where(~self.active | invalid, 1e6, score)
        return dict(metrics=metrics, categories=categories, failure_score=score,
                    final_truth=self.pose, final_estimate=self.estimate,
                    telemetry={key: torch.stack([frame[key] for frame in frames]) for key in frames[0]} if frames else {})


def records(result):
    """Convert completed batched tensors to scalar JSON records, outside stepping."""
    metrics = {key: value.cpu().tolist() for key, value in result["metrics"].items()}
    categories = {key: value.cpu().tolist() for key, value in result["categories"].items()}
    truth, estimate = result["final_truth"].cpu().tolist(), result["final_estimate"].cpu().tolist()
    scores = result["failure_score"].cpu().tolist()
    output = []
    for lane in range(len(scores)):
        failures = [key for key, values in categories.items() if values[lane]]
        output.append(dict(metrics={key: values[lane] if math.isfinite(values[lane]) else None for key, values in metrics.items()},
                           categories=failures or ["SUCCESS"], passed=not failures, failure_score=scores[lane],
                           final_truth=None if categories["NUMERICAL_FAILURE"][lane] else truth[lane], final_estimate=estimate[lane]))
    return output
