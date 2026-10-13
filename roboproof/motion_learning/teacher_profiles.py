"""Pure, training-only recipes; nothing here runs or promotes a policy.

Measurements contain reason, metrics and pace. Probe measurements also contain
the exact plan used for their episode. Only measured terminal scalars rank plans;
teacher_action reads raw public reference time and the reset reference duration.
The caller must charge every actual episode and step, including failed or
interrupted attempts. Combined actions need their own measured episode: single
axis measurements do not establish combined success or reward.
"""

import math


DEFAULT_TEACHER_PROFILE = "three-pace-v1"
MEASURED_LATE_POSE_PROFILE = "measured-late-pose-v1"
FAILURE_GATED_LATE_POSE_PROFILE = "failure-gated-late-pose-v1"
FINE_POSE_HEADING_PROFILE = "failure-gated-fine-pose-heading-v1"
TEACHER_PROFILES = (DEFAULT_TEACHER_PROFILE, MEASURED_LATE_POSE_PROFILE,
                    FAILURE_GATED_LATE_POSE_PROFILE, FINE_POSE_HEADING_PROFILE)
TEACHER_PACES = (0.85, 0.95, 1.0)
MAX_TEACHER_EPISODES = 8
MAX_FINE_TEACHER_EPISODES = 10
LATE_PHASE_FRACTION = 0.8
PROBE_MAGNITUDE = 0.5
OBSERVATION_SIZE = 34
REFERENCE_SECONDS_INDEX = 25
POSITION_ERROR_SCALE = 0.02032
HEADING_ERROR_SCALE = 0.035
PROBE_OFFSETS = (("right-negative", -0.5, 0.0), ("right-positive", 0.5, 0.0),
                 ("forward-negative", 0.0, -0.5), ("forward-positive", 0.0, 0.5))
FINE_PROBE_OFFSETS = (("fine-right-negative", -0.125, 0.0, 0.0), ("fine-right-positive", 0.125, 0.0, 0.0),
                      ("fine-forward-negative", 0.0, -0.125, 0.0), ("fine-forward-positive", 0.0, 0.125, 0.0),
                      ("fine-heading-negative", 0.0, 0.0, -0.1), ("fine-heading-positive", 0.0, 0.0, 0.1))
NONFAULT_REASONS = ("success", "time_limit")
FAULT_REASONS = ("controller_or_sensor_fault", "stale_policy_action", "stopped")
METRIC_NAMES = ("contactSeconds", "positionErrorMeters", "headingErrorRadians",
                "effortProxyVAs", "elapsedSeconds")
PLAN_FIELDS = ("kind", "pace", "right", "forward", "heading", "phaseStartFraction")


def _number(value, name, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Finite numeric {name} required")
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise ValueError(f"Finite numeric {name} required") from None
    if (not math.isfinite(result) or minimum is not None and result < minimum or
            maximum is not None and result > maximum):
        raise ValueError(f"Bounded finite {name} required")
    return result


def _profile(profile):
    if not isinstance(profile, str) or profile not in TEACHER_PROFILES:
        raise ValueError("Unsupported teacher profile")
    return profile


def _pace(pace):
    result = _number(pace, "teacher pace", 0.25, 1.0)
    if result not in TEACHER_PACES:
        raise ValueError("Measured three-pace teacher pace required")
    return result


def _metrics(metrics):
    if not isinstance(metrics, dict):
        raise ValueError("Measured terminal metrics required")
    return {name: _number(metrics.get(name), name, 0.0) for name in METRIC_NAMES}


def terminal_pose_error_squared(metrics):
    """Rank measured terminal pose only; this is not a reward or success gate."""
    measured = _metrics(metrics)
    position = measured["positionErrorMeters"] / POSITION_ERROR_SCALE
    heading = measured["headingErrorRadians"] / HEADING_ERROR_SCALE
    return _number(position * position + heading * heading, "normalized terminal pose error", 0.0)


def _measurement(candidate):
    if not isinstance(candidate, dict):
        raise ValueError("Measured teacher candidate required")
    reason = candidate.get("reason")
    if not isinstance(reason, str) or reason not in NONFAULT_REASONS + FAULT_REASONS:
        raise ValueError("Measured terminal reason required")
    metrics = _metrics(candidate.get("metrics"))
    terminal_pose_error_squared(metrics)
    return {"pace": _pace(candidate.get("pace")), "reason": reason, "metrics": metrics}


def _base_candidates(candidates):
    if not isinstance(candidates, (list, tuple)) or len(candidates) > len(TEACHER_PACES):
        raise ValueError("At most three measured base candidates required")
    measured = [_measurement(candidate) for candidate in candidates]
    if len({candidate["pace"] for candidate in measured}) != len(measured):
        raise ValueError("Distinct measured base paces required")
    for candidate in candidates:
        if "plan" in candidate:
            plan = validate_plan(candidate["plan"])
            if plan["kind"] != "base" or plan["pace"] != candidate["pace"]:
                raise ValueError("Only matching base measurements may select probe pace")
    return measured


def _plan(kind, pace, right=0.0, forward=0.0, heading=0.0):
    return {"kind": kind, "pace": pace, "right": right, "forward": forward,
            "heading": heading, "phaseStartFraction": 0.0 if kind == "base" else LATE_PHASE_FRACTION}


def validate_plan(plan):
    """Return a fresh canonical plan, rejecting changed phases or probe shapes."""
    if not isinstance(plan, dict) or set(plan) != set(PLAN_FIELDS):
        raise ValueError("Exact declarative teacher plan required")
    kind = plan["kind"]
    if (not isinstance(kind, str) or kind not in ("base", "combined", "fine-combined") +
            tuple(item[0] for item in PROBE_OFFSETS + FINE_PROBE_OFFSETS)):
        raise ValueError("Bounded teacher plan kind required")
    fine = kind == "fine-combined" or kind in tuple(item[0] for item in FINE_PROBE_OFFSETS)
    pace = _pace(plan["pace"])
    magnitude = 0.125 if fine else PROBE_MAGNITUDE
    right = _number(plan["right"], "right offset", -magnitude, magnitude)
    forward = _number(plan["forward"], "forward offset", -magnitude, magnitude)
    heading = _number(plan["heading"], "heading offset", -0.1 if fine else 0.0, 0.1 if fine else 0.0)
    phase = _number(plan["phaseStartFraction"], "phase fraction", 0.0, 1.0)
    expected_phase = 0.0 if kind == "base" else LATE_PHASE_FRACTION
    if phase != expected_phase:
        raise ValueError("Exact public reference phase required")
    if kind == "base" and (right != 0.0 or forward != 0.0):
        raise ValueError("Default base teachers have zero pose offsets")
    if kind == "combined":
        if right not in (-0.5, 0.0, 0.5) or forward not in (-0.5, 0.0, 0.5) or (right == 0.0 and forward == 0.0):
            raise ValueError("Nonzero measured signed combined plan required")
    elif kind == "fine-combined":
        if (right not in (-0.125, 0.0, 0.125) or forward not in (-0.125, 0.0, 0.125) or
                heading not in (-0.1, 0.0, 0.1) or (right == 0.0 and forward == 0.0 and heading == 0.0)):
            raise ValueError("Nonzero measured fine-position/angular combination required")
    elif fine:
        expected = next(item for item in FINE_PROBE_OFFSETS if item[0] == kind)
        if (right, forward, heading) != expected[1:]:
            raise ValueError("Exact signed fine-position/angular probe required")
    elif kind != "base":
        expected = next(item for item in PROBE_OFFSETS if item[0] == kind)
        if (right, forward) != expected[1:]:
            raise ValueError("Exact signed single-axis probe required")
    return {"kind": kind, "pace": pace, "right": right, "forward": forward,
            "heading": heading, "phaseStartFraction": phase}


def validate_action(action):
    """Validate the original four-unit policy action without clamping values."""
    if not isinstance(action, (list, tuple)) or len(action) != 4:
        raise ValueError("Four-unit teacher action required")
    return [_number(value, "action unit", -1.0, 1.0) for value in action]


def base_plans(profile=DEFAULT_TEACHER_PROFILE):
    """All profiles retain exactly the original three zero-pose teachers."""
    _profile(profile)
    return [_plan("base", pace) for pace in TEACHER_PACES]


def select_probe_pace(candidates, *, allow_fallback=False):
    """Contact, normalized squared pose, effort, time, then pace break ties.

    Empty validated measurements permit fixed 0.95 only with explicit fallback.
    Fault-only measurements always skip probes; malformed data always rejects.
    """
    if not isinstance(allow_fallback, bool):
        raise ValueError("Explicit boolean fallback permission required")
    measured = _base_candidates(candidates)
    eligible = [candidate for candidate in measured if candidate["reason"] in NONFAULT_REASONS]
    if not eligible:
        return 0.95 if allow_fallback and not measured else None
    selected = min(eligible, key=lambda candidate: (
        candidate["metrics"]["contactSeconds"], terminal_pose_error_squared(candidate["metrics"]),
        candidate["metrics"]["effortProxyVAs"], candidate["metrics"]["elapsedSeconds"], candidate["pace"]))
    return selected["pace"]


def probe_plans(candidates, profile=DEFAULT_TEACHER_PROFILE, *, allow_fallback=False):
    """Opt-in probes; failure gating requires all three measured base attempts."""
    _profile(profile)
    pace = select_probe_pace(candidates, allow_fallback=allow_fallback)
    if profile in (FAILURE_GATED_LATE_POSE_PROFILE, FINE_POSE_HEADING_PROFILE):
        measured = _base_candidates(candidates)
        if len(measured) != len(TEACHER_PACES):
            raise ValueError("Failure gating requires all three measured base paces")
        if any(candidate["reason"] == "success" and candidate["metrics"]["contactSeconds"] == 0
               for candidate in measured):
            return []
    if profile == DEFAULT_TEACHER_PROFILE or pace is None:
        return []
    if profile == FINE_POSE_HEADING_PROFILE:
        return [_plan(kind, pace, right, forward, heading) for kind, right, forward, heading in FINE_PROBE_OFFSETS]
    return [_plan(kind, pace, right, forward) for kind, right, forward in PROBE_OFFSETS]


def combined_plan(baseline, probes, profile=DEFAULT_TEACHER_PROFILE):
    """Choose each axis by strictly lower position error at baseline contact.

    Each probe record requires its measured reason, metrics, pace and exact plan.
    Ties retain baseline zero, then negative before positive for equal improved
    errors. The returned plan makes no claim about its unmeasured outcome.
    """
    _profile(profile)
    if profile == FINE_POSE_HEADING_PROFILE:
        return fine_combined_plan(baseline, probes)
    measured_baseline = _base_candidates([baseline])[0]
    if not isinstance(probes, (list, tuple)) or len(probes) != len(PROBE_OFFSETS):
        raise ValueError("Exactly four measured signed probes required")
    by_kind = {}
    for probe in probes:
        measured = _measurement(probe)
        plan = validate_plan(probe.get("plan"))
        if (plan["kind"] not in tuple(item[0] for item in PROBE_OFFSETS) or plan["kind"] in by_kind or
                plan["pace"] != measured["pace"] or measured["pace"] != measured_baseline["pace"]):
            raise ValueError("Distinct same-pace measured signed probes required")
        by_kind[plan["kind"]] = measured
    if profile == DEFAULT_TEACHER_PROFILE or measured_baseline["reason"] not in NONFAULT_REASONS:
        return None
    if (profile == FAILURE_GATED_LATE_POSE_PROFILE and measured_baseline["reason"] == "success" and
            measured_baseline["metrics"]["contactSeconds"] == 0):
        return None
    baseline_metrics = measured_baseline["metrics"]
    offsets = []
    for axis in ("right", "forward"):
        best_error = baseline_metrics["positionErrorMeters"]
        best_offset = 0.0
        for sign, offset in (("negative", -PROBE_MAGNITUDE), ("positive", PROBE_MAGNITUDE)):
            probe = by_kind[f"{axis}-{sign}"]
            metrics = probe["metrics"]
            if (probe["reason"] in NONFAULT_REASONS and metrics["contactSeconds"] == baseline_metrics["contactSeconds"] and
                    metrics["positionErrorMeters"] < best_error):
                best_error = metrics["positionErrorMeters"]
                best_offset = offset
        offsets.append(best_offset)
    if offsets == [0.0, 0.0]:
        return None
    return _plan("combined", measured_baseline["pace"], offsets[0], offsets[1])


def fine_combined_plan(baseline, probes):
    """Measure each signed axis; combine only strictly lower normalized pose error."""
    measured_baseline = _base_candidates([baseline])[0]
    if not isinstance(probes, (list, tuple)) or len(probes) != len(FINE_PROBE_OFFSETS):
        raise ValueError("Exactly six measured fine-position/angular probes required")
    by_kind = {}
    for probe in probes:
        measured = _measurement(probe)
        plan = validate_plan(probe.get("plan"))
        if (plan["kind"] not in tuple(item[0] for item in FINE_PROBE_OFFSETS) or plan["kind"] in by_kind or
                plan["pace"] != measured["pace"] or measured["pace"] != measured_baseline["pace"]):
            raise ValueError("Distinct same-pace measured fine-position/angular probes required")
        by_kind[plan["kind"]] = measured
    if (measured_baseline["reason"] not in NONFAULT_REASONS or
            measured_baseline["reason"] == "success" and measured_baseline["metrics"]["contactSeconds"] == 0):
        return None
    baseline_metrics = measured_baseline["metrics"]
    offsets = []
    for axis, magnitude in (("right", 0.125), ("forward", 0.125), ("heading", 0.1)):
        best_error = terminal_pose_error_squared(baseline_metrics)
        best_offset = 0.0
        for sign, offset in (("negative", -magnitude), ("positive", magnitude)):
            probe = by_kind[f"fine-{axis}-{sign}"]
            metrics = probe["metrics"]
            error = terminal_pose_error_squared(metrics)
            if (probe["reason"] in NONFAULT_REASONS and metrics["contactSeconds"] == baseline_metrics["contactSeconds"] and
                    error < best_error):
                best_error = error
                best_offset = offset
        offsets.append(best_offset)
    if offsets == [0.0, 0.0, 0.0]:
        return None
    return _plan("fine-combined", measured_baseline["pace"], *offsets)


def teacher_action(plan, observation, reference_duration_seconds):
    """Use only raw public observation[25] and duration from this episode reset."""
    validated = validate_plan(plan)
    if not isinstance(observation, (list, tuple)) or len(observation) != OBSERVATION_SIZE:
        raise ValueError("Raw 34-value public policy observation required")
    raw = [_number(value, "public observation") for value in observation]
    duration = _number(reference_duration_seconds, "reset reference duration", 0.0)
    reference_seconds = _number(raw[REFERENCE_SECONDS_INDEX], "public reference seconds", 0.0, duration)
    late = reference_seconds >= validated["phaseStartFraction"] * duration
    pace_unit = 2 * (validated["pace"] - 0.25) / 0.75 - 1
    return validate_action([pace_unit, validated["right"] if late else 0.0,
                            validated["forward"] if late else 0.0, validated["heading"] if late else 0.0])


def reference_offset_target(action, reference_heading_radians):
    """Original current-reference-body target only, not the filtered reference."""
    units = validate_action(action)
    heading = _number(reference_heading_radians, "public current reference heading")
    right = units[1] * 0.08
    forward = units[2] * 0.08
    return [right * math.cos(heading) + forward * math.sin(heading),
            -right * math.sin(heading) + forward * math.cos(heading), units[3] * 0.1]


def validate_episode_count(attempted_episodes, profile=DEFAULT_TEACHER_PROFILE):
    """Check a caller-maintained attempt count; never infer or refund actual cost."""
    _profile(profile)
    maximum = len(TEACHER_PACES) if profile == DEFAULT_TEACHER_PROFILE else MAX_TEACHER_EPISODES
    if profile == FINE_POSE_HEADING_PROFILE:
        maximum = MAX_FINE_TEACHER_EPISODES
    if isinstance(attempted_episodes, bool) or not isinstance(attempted_episodes, int) or not 0 <= attempted_episodes <= maximum:
        raise ValueError("Bounded actual teacher episode count required")
    return attempted_episodes


def teacher_profile_definition(profile=DEFAULT_TEACHER_PROFILE):
    """Fresh JSON-compatible declarative training metadata, never actor features."""
    _profile(profile)
    measured = profile != DEFAULT_TEACHER_PROFILE
    definition = {"schemaVersion": 1, "id": profile, "default": not measured, "basePaces": list(TEACHER_PACES),
            "basePlans": base_plans(profile), "maximumEpisodesPerWorld": MAX_TEACHER_EPISODES if measured else 3,
            "maximumProbeEpisodes": 4 if measured else 0, "maximumCombinedEpisodes": 1 if measured else 0,
            "actionOrder": ["paceUnit", "right", "forward", "heading"], "actionBounds": [-1.0, 1.0],
            "paceMapping": "pace = 0.25 + (paceUnit + 1) * 0.75 / 2",
            "offsetFrame": "Current reference heading; not start, goal or true heading",
            "offsetScales": {"rightMeters": 0.08, "forwardMeters": 0.08, "headingRadians": 0.1},
            "latePose": None if not measured else {
                "observationSize": OBSERVATION_SIZE, "referenceSecondsIndex": REFERENCE_SECONDS_INDEX,
                "resetDurationKey": "referenceDurationSeconds", "phaseStartFraction": LATE_PHASE_FRACTION,
                "probeMagnitude": PROBE_MAGNITUDE,
                "probeOffsets": [{"kind": kind, "right": right, "forward": forward, "heading": 0.0}
                                 for kind, right, forward in PROBE_OFFSETS],
                "paceRanking": ["contactSeconds", "normalizedTerminalPoseErrorSquared", "effortProxyVAs", "elapsedSeconds", "pace"],
                "poseScales": {"positionErrorMeters": POSITION_ERROR_SCALE, "headingErrorRadians": HEADING_ERROR_SCALE},
                "nonfaultReasons": list(NONFAULT_REASONS), "fallbackPace": 0.95,
                "fallbackRule": "Explicit allow_fallback and validated empty candidates only; fault-only results skip",
                "combinedRule": "Each axis: baseline zero or measured signed probe, same contact and strictly lower position error",
                "combinedTieRule": "Baseline zero, then negative, then positive"},
            "costAccounting": "Caller charges every actual episode and step, including failures and interruptions; no refunds",
            "scope": "Training-only measured labels; original physics, filter, actor features, reward and success gates unchanged"}
    if profile in (FAILURE_GATED_LATE_POSE_PROFILE, FINE_POSE_HEADING_PROFILE):
        definition["latePose"]["probeGate"] = "All three measured base paces required; probe only if none succeeds with exactly zero contact"
        definition["latePose"]["fallbackRule"] = "No empty or partial base measurements; all three actual attempts required"
        definition["latePose"]["combinedGate"] = "No successful zero-contact base teacher; combined labels still require their own actual episode"
    if profile == FINE_POSE_HEADING_PROFILE:
        definition["maximumEpisodesPerWorld"] = MAX_FINE_TEACHER_EPISODES
        definition["maximumProbeEpisodes"] = len(FINE_PROBE_OFFSETS)
        late = definition["latePose"]
        del late["probeMagnitude"]
        late["probeMagnitudes"] = {"positionUnits": 0.125, "headingUnits": 0.1,
                                   "positionMeters": 0.01, "headingRadians": 0.01}
        late["probeOffsets"] = [{"kind": kind, "right": right, "forward": forward, "heading": heading}
                                for kind, right, forward, heading in FINE_PROBE_OFFSETS]
        late["combinedRule"] = "Each axis: baseline zero or measured signed probe, same contact and strictly lower normalized squared position/heading error; measure combination separately"
    return definition
