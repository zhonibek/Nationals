import math


OBJECTIVE = "finite-deadline-pose-effort-v2"
TOTAL_RETURN_OBJECTIVE = "finite-total-return-pose-effort-v3"
SETTLING_MARGIN_OBJECTIVE = "finite-total-return-settle-margin-v4"
PROFILES = ("canonical-v1", "finite-deadline-effort-v1", OBJECTIVE, TOTAL_RETURN_OBJECTIVE, SETTLING_MARGIN_OBJECTIVE)
SETTLING_TOLERANCES = {"positionErrorMeters": 0.02032, "headingErrorRadians": 0.035,
                       "speedMetersPerSecond": 0.0254, "yawRateRadiansPerSecond": 0.0873}


def objective_definition(profile):
    if profile not in PROFILES:
        raise ValueError("Unsupported learning objective")
    return None if profile != SETTLING_MARGIN_OBJECTIVE else {
        "method": SETTLING_MARGIN_OBJECTIVE, "baseReward": TOTAL_RETURN_OBJECTIVE,
        "terminalReasons": ["success", "time_limit"], "successTolerances": dict(SETTLING_TOLERANCES),
        "bonus": "2*min(tolerance/(tolerance+nonnegativeMeasuredError)) across all four criteria",
        "range": [0, 2], "timing": "One bonus at intrinsic episode end; none for running or fault states",
        "scope": "Privileged training-only terminal feedback; canonical reward, actor inputs and pass gates unchanged"}


def terminal_settling_margin(info):
    if info.get("reason") not in ("success", "time_limit"):
        return None
    fractions = []
    for name, tolerance in SETTLING_TOLERANCES.items():
        value = info.get(name)
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(value) or value < 0):
            raise ValueError("Finite nonnegative measured terminal settling metrics required")
        fractions.append(tolerance / (tolerance + value))
    return min(fractions)


def discount_parameters(profile):
    if profile not in PROFILES:
        raise ValueError("Unsupported learning objective")
    if profile in (TOTAL_RETURN_OBJECTIVE, SETTLING_MARGIN_OBJECTIVE):
        return 1.0, 1.0
    return (0.99, 0.95) if profile == "canonical-v1" else (0.995, 0.98)


def return_estimator(profile):
    gamma, decay = discount_parameters(profile)
    return {"discountFactor": gamma, "gaeLambda": decay, "completeEpisodesOnly": True,
            "intrinsicDeadlineTerminal": profile != "canonical-v1"}


class LearningObjective:
    def __init__(self, task, profile=OBJECTIVE):
        if profile not in PROFILES:
            raise ValueError("Unsupported learning objective")
        self.profile = profile
        self.terminal_margin_score = None
        self.distance_scale = max(0.1, math.hypot(task["goal"]["xIn"] - task["start"]["xIn"],
                                                 task["goal"]["yIn"] - task["start"]["yIn"]) * 0.0254)
        initial_heading = math.radians(task["goal"].get("headingDeg", 0) - task["start"].get("headingDeg", 0))
        self.previous_heading = abs(math.atan2(math.sin(initial_heading), math.cos(initial_heading)))
        self.heading_scale = max(0.2, self.previous_heading)

    def reward(self, canonical_reward, info):
        if not math.isfinite(canonical_reward):
            raise ValueError("Nonfinite canonical learning reward")
        if self.profile == "canonical-v1":
            return canonical_reward
        components = info.get("rewardComponents")
        if not isinstance(components, dict) or any(name not in components or not math.isfinite(components[name])
                                                  for name in ("progress", "time", "effort", "contact", "success", "fault")):
            raise ValueError("Measured canonical reward components required")
        reward = (components["progress"] / self.distance_scale + components["time"] +
                  100 * components["effort"] + components["contact"] +
                  2 * components["success"] + 2 * components["fault"])
        if info.get("reason") == "time_limit":
            reward -= 2
        if self.profile in (OBJECTIVE, TOTAL_RETURN_OBJECTIVE, SETTLING_MARGIN_OBJECTIVE):
            heading = info.get("headingErrorRadians")
            if not isinstance(heading, (float, int)) or not math.isfinite(heading) or heading < 0:
                raise ValueError("Measured heading error required for pose objective")
            reward += 0.5 * (self.previous_heading - heading) / self.heading_scale
            self.previous_heading = heading
        if self.profile == SETTLING_MARGIN_OBJECTIVE:
            self.terminal_margin_score = terminal_settling_margin(info)
            if self.terminal_margin_score is not None:
                reward += 2 * self.terminal_margin_score
        return reward

    def terminal(self, terminated, truncated, info):
        return terminated or (self.profile != "canonical-v1" and truncated and info.get("reason") == "time_limit")
