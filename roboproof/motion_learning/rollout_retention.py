import math

from .imitation import demonstration_samples, teacher_margins


def retained_success_rows(rows, report, task):
    if not isinstance(rows, list) or not rows or rows[-1].get("terminal") is not True:
        raise ValueError("Actual complete intrinsic episode rows required")
    if not isinstance(report, dict) or report.get("reason") not in (
            "success", "time_limit", "controller_or_sensor_fault", "stale_policy_action", "stopped"):
        raise ValueError("Measured terminal episode report required")
    metrics = report.get("metrics")
    margins = teacher_margins(metrics, task)
    contact = metrics.get("contactSeconds")
    if isinstance(contact, bool) or not isinstance(contact, (int, float)) or not math.isfinite(contact) or contact < 0:
        raise ValueError("Finite measured contact required")
    feature_size = None
    converted = []
    for index, row in enumerate(rows):
        features, action = row.get("modelInput"), row.get("action")
        if (not isinstance(features, list) or not features or not isinstance(action, list) or len(action) != 4 or
                any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                    for value in features + action) or any(abs(value) > 1 for value in action) or
                index < len(rows) - 1 and row.get("terminal") is not False):
            raise ValueError("Contiguous finite public features and actual bounded actions required")
        feature_size = len(features) if feature_size is None else feature_size
        if len(features) != feature_size:
            raise ValueError("Identical public feature shape required")
        converted.append({"modelInput": list(features), "teacherAction": list(action)})
    if report["reason"] != "success" or contact != 0 or margins["marginScore"] < 1:
        return []
    return demonstration_samples(converted)
