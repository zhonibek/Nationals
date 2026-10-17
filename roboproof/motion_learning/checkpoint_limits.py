import math

from .imitation import ANCHOR_CAPACITY
from .model import AUGMENTED_FEATURES, DEADLINE_FEATURES, FEATURE_TRANSFORMS, HISTORY_FRAMES, SCALES, validate_network_width, validate_network_depth


CHECKPOINT_METADATA_BYTES = 4 * 1024 * 1024
CHECKPOINT_SERIALIZATION_BYTES = 64 * 1024


def checkpoint_tensor_bytes(config):
    """Count float32 parameters, full Adam moments, buffers and bounded replay/anchors without allocations."""
    network_width = validate_network_width(config.get("networkWidth", 32))
    network_depth = validate_network_depth(config.get("networkDepth", 2))
    additional_parameters = (network_depth - 2) * (network_width ** 2 + network_width)
    feature_transform = config.get("featureTransform", "identity-v1")
    if feature_transform not in FEATURE_TRANSFORMS:
        raise ValueError("Supported checkpoint feature transform required")
    feature_size = len(SCALES) + int(feature_transform != "identity-v1")
    if feature_transform in AUGMENTED_FEATURES:
        feature_size += 6
    if feature_transform in HISTORY_FRAMES:
        feature_size = feature_size * HISTORY_FRAMES[feature_transform] + 4
    feature_size += int(feature_transform in DEADLINE_FEATURES)
    anchor_weight = config.get("demonstrationAnchorWeight", 0)
    if (isinstance(anchor_weight, bool) or not isinstance(anchor_weight, (int, float)) or
            not math.isfinite(anchor_weight) or not 0 <= anchor_weight <= 100):
        raise ValueError("Finite bounded checkpoint anchor weight required")
    anchor_bytes = ANCHOR_CAPACITY * (feature_size + 4) * 4 if anchor_weight else 0
    retention_weight = config.get("selfRetentionWeight", 0)
    if (isinstance(retention_weight, bool) or not isinstance(retention_weight, (int, float)) or
            not math.isfinite(retention_weight) or not 0 <= retention_weight <= 100):
        raise ValueError("Finite bounded successful-rollout retention weight required")
    if retention_weight:
        anchor_bytes += ANCHOR_CAPACITY * (feature_size + 4) * 4
    actor_parameters = feature_size * network_width + network_width ** 2 + 10 * network_width + 8 + additional_parameters
    algorithm = config.get("algorithm", "ppo-beta-reference-v1")
    if algorithm == "ppo-beta-reference-v1":
        critic_parameters = feature_size * network_width + network_width ** 2 + 3 * network_width + 1 + additional_parameters
        return 3 * (actor_parameters + critic_parameters) * 4 + feature_size * 4 + 4 * (network_depth + 1) * 4 + anchor_bytes
    if algorithm != "td3-beta-mean-reference-v1":
        raise ValueError("Supported checkpoint algorithm required")
    replay_capacity = config["replayCapacity"]
    if isinstance(replay_capacity, bool) or not isinstance(replay_capacity, int) or not 64 <= replay_capacity <= 2048:
        raise ValueError("Bounded checkpoint replay capacity from64 to2048 required")
    critic_parameters = feature_size * network_width + network_width ** 2 + 7 * network_width + 1 + additional_parameters
    replay_bytes = replay_capacity * (2 * feature_size * 4 + 4 * 4 + 2 * 4 + 8 + 1)
    return (4 * actor_parameters + 8 * critic_parameters) * 4 + 4 * feature_size * 4 + 6 * (network_depth + 1) * 4 + replay_bytes + anchor_bytes


def checkpoint_size_limit(config):
    """Reserve the unchanged 4 MiB JSON history budget plus archive, config, source and RNG overhead."""
    return checkpoint_tensor_bytes(config) + CHECKPOINT_METADATA_BYTES + CHECKPOINT_SERIALIZATION_BYTES
