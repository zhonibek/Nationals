"""Cascade.hpp semantics, with tensor lanes representing independent controllers."""

from dataclasses import dataclass
import math

import torch


DEFAULT_CONFIG = dict(
    radius=0.1885853785424542, maxWheelSpeed=0.864462,
    maxVoltage=12.0, maxAcceleration=2.0, kS=0.40, kV=13.88, kA=0.15,
    wheelKp=2.5, wheelKi=0.8, wheelKd=0.0,
    qPosition=4.0, qHeading=3.0, rTranslation=0.5, rRotation=0.5,
)
POSITIVE = ("radius", "maxWheelSpeed", "maxVoltage", "maxAcceleration",
            "qPosition", "qHeading", "rTranslation", "rRotation")


def resolve_device(name="cpu"):
    if name == "cpu":
        return torch.device("cpu")
    if name != "rocm":
        raise ValueError("device must be cpu or rocm; NVIDIA CUDA is not AMD ROCm")
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("AMD ROCm requested, but torch.version.hip and an available "
                           "HIP device are required; no GPU measurement was performed")
    return torch.device("cuda")


def wrap(angle):
    period = 2 * math.pi
    return angle - torch.round(angle / period) * period


def inverse_x(strafe, forward, omega, radius):
    plus = (forward + strafe) * 0.7071067811865475
    minus = (forward - strafe) * 0.7071067811865475
    rotation = radius * omega
    return torch.stack((plus + rotation, minus + rotation,
                        minus - rotation, plus - rotation), -1)


def forward_x(wheel, radius):
    front_left, back_left, front_right, back_right = wheel.unbind(-1)
    return torch.stack(((front_left - back_left - front_right + back_right) * 0.3535533905932738,
                        (front_left + back_left + front_right + back_right) * 0.3535533905932738,
                        (front_left + back_left - front_right - back_right) / (4 * radius)), -1)


def inverse(matrix):
    """Batched partial-pivot Gauss-Jordan, including the C++ 1e-12 cutoff."""
    batch = matrix.shape[0]
    eye = torch.eye(3, dtype=matrix.dtype, device=matrix.device).expand(batch, -1, -1)
    augmented = torch.cat((matrix, eye), -1).clone()
    valid = torch.ones(batch, dtype=torch.bool, device=matrix.device)
    lanes = torch.arange(batch, device=matrix.device)
    for column in range(3):
        pivot = augmented[:, column:, column].abs().argmax(-1) + column
        selected = augmented[lanes, pivot].clone()
        old = augmented[:, column].clone()
        augmented[lanes, pivot] = old
        augmented[:, column] = selected
        divisor = augmented[:, column, column]
        good = torch.isfinite(divisor) & (divisor.abs() >= 1e-12)
        valid &= good
        row = augmented[:, column] / torch.where(good, divisor, 1.0)[:, None]
        factors = augmented[:, :, column].clone()
        factors[:, column] = 0
        augmented = augmented - factors[:, :, None] * row[:, None, :]
        augmented[:, column] = row
    return augmented[:, :, 3:], valid


@dataclass
class Output:
    volts: torch.Tensor
    targets: torch.Tensor
    valid: torch.Tensor


class Cascade:
    def __init__(self, batch_size, device="cpu", config=None):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self.dtype = torch.float64
        self.config = {key: self.lanes(value) for key, value in DEFAULT_CONFIG.items()}
        if config:
            self.configure(**config)
        self.gain = self.zeros(3, 3)
        self.gain_valid = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        self.last = self.zeros(4)
        self.integral = self.zeros(4)
        self.previous = self.zeros(4)
        self.previous_measured = self.zeros(4)
        self.derivative = self.zeros(4)
        self.first = torch.ones_like(self.gain_valid)
        self.solve_count = torch.zeros(batch_size, dtype=torch.int64, device=self.device)
        self.solve_iterations = torch.zeros_like(self.solve_count)

    def zeros(self, *shape):
        return torch.zeros((self.batch_size, *shape), dtype=self.dtype, device=self.device)

    def lanes(self, value):
        return torch.broadcast_to(torch.as_tensor(value, dtype=self.dtype, device=self.device),
                                  (self.batch_size,)).clone()

    def configure(self, **values):
        for key, value in values.items():
            if key not in DEFAULT_CONFIG:
                raise ValueError(f"Unknown controller config: {key}")
            self.config[key] = self.lanes(value)
        if hasattr(self, "gain_valid"):
            self.reset()

    def reset(self, mask=None, wheels_only=False):
        if mask is None:
            mask = torch.ones_like(self.first)
        if not wheels_only:
            self.gain_valid &= ~mask
        for state in (self.integral, self.previous, self.previous_measured, self.derivative):
            state.masked_fill_(mask[:, None], 0)
        self.first |= mask

    def valid_config(self):
        valid = torch.ones_like(self.first)
        for key, value in self.config.items():
            valid &= torch.isfinite(value) & (value > 0 if key in POSITIVE else value >= 0)
        return valid & (self.config["maxVoltage"] <= 12)

    def solve(self, reference, dt, requested):
        config = self.config
        transition = torch.eye(3, dtype=self.dtype, device=self.device).repeat(self.batch_size, 1, 1)
        transition[:, 0, 2] = reference[:, 4] * dt
        transition[:, 1, 2] = -reference[:, 3] * dt
        inputs = self.zeros(3, 3)
        cosine, sine = reference[:, 2].cos() * dt, reference[:, 2].sin() * dt
        inputs[:, 0, 0] = cosine
        inputs[:, 0, 1] = sine
        inputs[:, 1, 0] = -sine
        inputs[:, 1, 1] = cosine
        inputs[:, 2, 2] = dt
        cost = torch.diag_embed(torch.stack((config["qPosition"], config["qPosition"], config["qHeading"]), -1))
        effort = torch.diag_embed(torch.stack((config["rTranslation"], config["rTranslation"], config["rRotation"]), -1))
        riccati = cost.clone()
        active = requested.clone()
        converged = torch.zeros_like(active)
        self.solve_count += requested
        self.solve_iterations = torch.where(requested, 0, self.solve_iterations)
        for iteration in range(600):
            inverted, invertible = inverse(effort + inputs.mT @ riccati @ inputs)
            gain = inverted @ inputs.mT @ riccati @ transition
            updated = cost + transition.mT @ riccati @ transition - transition.mT @ riccati @ inputs @ gain
            finite = torch.isfinite(updated).all(dim=(-2, -1))
            delta = (updated - riccati).abs().amax(dim=(-2, -1))
            scale = updated.abs().amax(dim=(-2, -1)).clamp_min(1)
            good = active & invertible & finite
            riccati = torch.where(good[:, None, None], updated, riccati)
            done = good & (delta < 1e-7 * scale)
            converged |= done
            self.solve_iterations = torch.where(active, iteration + 1, self.solve_iterations)
            active = good & ~done
            if not bool(active.any()):
                break
        inverted, invertible = inverse(effort + inputs.mT @ riccati @ inputs)
        success = converged & invertible
        self.gain = torch.where(success[:, None, None], inverted @ inputs.mT @ riccati @ transition, self.gain)
        self.gain_valid |= success
        latest = torch.stack((reference[:, 2], reference[:, 3], reference[:, 4], dt), -1)
        self.last = torch.where(success[:, None], latest, self.last)
        return ~requested | success

    @torch.no_grad()
    def step(self, reference, feedback, dt, valid=True):
        reference = torch.as_tensor(reference, dtype=self.dtype, device=self.device)
        feedback = torch.as_tensor(feedback, dtype=self.dtype, device=self.device)
        if reference.shape != (self.batch_size, 6) or feedback.shape != (self.batch_size, 10):
            raise ValueError("reference must be [batch,6], feedback [batch,10]")
        dt = self.lanes(dt)
        healthy = self.lanes(valid).bool() & self.valid_config()
        healthy &= torch.isfinite(dt) & (dt > 0) & (dt <= 0.1)
        healthy &= torch.isfinite(reference).all(-1) & torch.isfinite(feedback[:, :6]).all(-1)
        requested = ~self.gain_valid | (wrap(reference[:, 2] - self.last[:, 0]).abs() > 0.02)
        requested |= (reference[:, 3] - self.last[:, 1]).abs() > 0.02
        requested |= (reference[:, 4] - self.last[:, 2]).abs() > 0.02
        requested |= (dt - self.last[:, 3]).abs() > 0.0005
        if bool((requested & healthy).any()):
            healthy &= self.solve(reference, dt, requested & healthy)
        self.reset(~healthy)
        error = reference[:, :3] - feedback[:, :3]
        error[:, 2] = wrap(error[:, 2])
        correction = (self.gain @ error[:, :, None]).squeeze(-1)
        heading = reference[:, 2]
        strafe = reference[:, 3] * heading.cos() - reference[:, 4] * heading.sin() + correction[:, 0]
        forward = reference[:, 3] * heading.sin() + reference[:, 4] * heading.cos() + correction[:, 1]
        return self.wheel_step(strafe, forward, reference[:, 5] + correction[:, 2], feedback[:, 6:], dt, healthy)

    def wheel_step(self, strafe, forward, omega, measured, dt, valid):
        config = self.config
        valid = valid & torch.isfinite(measured).all(-1)
        valid &= torch.isfinite(strafe) & torch.isfinite(forward) & torch.isfinite(omega)
        safe_dt = torch.where(valid, dt, 0.01)[:, None]
        target = inverse_x(strafe, forward, omega, config["radius"])
        peak = (target.abs() / config["maxWheelSpeed"][:, None]).amax(-1).clamp_min(1)
        target = target / peak[:, None]
        ramp = ((target - self.previous).abs() / (config["maxAcceleration"][:, None] * safe_dt)).amax(-1).clamp_min(1)
        target = self.previous + (target - self.previous) / ramp[:, None]
        acceleration = (target - self.previous) / safe_dt
        error = target - measured
        candidate = (self.integral + error * safe_dt).clamp(-2, 2)
        raw = torch.where(self.first[:, None], 0, (measured - self.previous_measured) / safe_dt)
        self.derivative += (1 - torch.exp(-safe_dt / 0.03)) * (raw - self.derivative)
        self.previous_measured = measured.clone()
        feedforward = config["kS"][:, None] * torch.tanh(target / 0.04) + config["kV"][:, None] * target + config["kA"][:, None] * acceleration
        proportional = config["wheelKp"][:, None] * error - config["wheelKd"][:, None] * self.derivative
        voltage = feedforward + proportional + config["wheelKi"][:, None] * candidate
        maximum = config["maxVoltage"][:, None]
        self.integral = torch.where((voltage.abs() <= maximum) | (voltage * error < 0), candidate, self.integral)
        voltage = (feedforward + proportional + config["wheelKi"][:, None] * self.integral).clamp(-maximum, maximum)
        self.previous = target.clone()
        self.first.fill_(False)
        self.reset(~valid, wheels_only=True)
        return Output(torch.where(valid[:, None], voltage, 0), torch.where(valid[:, None], target, 0), valid)
