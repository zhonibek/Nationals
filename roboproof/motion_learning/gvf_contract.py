import copy
import json
import math
import struct


ALGORITHM = "td3-tanh-gvf-residual-v1"
CONTRACT_NAME = "gvf-residual"
CONTRACT_VERSION = 2
OBSERVATION_SCHEMA = "gvf-residual-sensor34-v2"
FEATURE_TRANSFORM = "gvf-body-reference36-v1"
OBSERVATION_SIZE = 34
FEATURE_SIZE = 36
ACTION_SIZE = 3
FIELDS = (("pose", 3), ("velocity", 3), ("wheelRimSpeeds", 4), ("targetError", 4),
          ("previousAction", 3), ("remainingSeconds", 1), ("baseReference", 6),
          ("projection", 1), ("remainingArcMeters", 1), ("crossTrackMeters", 1),
          ("appliedFieldOffset", 3), ("offsetVelocity", 3), ("actionAgeTicks", 1))
TAGS = {"contractName": CONTRACT_NAME, "contractVersion": CONTRACT_VERSION, "observationSchema": OBSERVATION_SCHEMA}
SCALES = tuple(struct.unpack("!f", struct.pack("!f", value))[0] for value in (
    1.524, 1.524, 1, 2, 2, 4, 3, 3, 3, 3, 4.32, 4.32, 1, 1, 1, 1, 1, 60,
    4.32, 4.32, 1, 0.35, 0.35, 1.5, 1, 4.32, 4.32, 0.08, 0.08, 0.1, 0.08, 0.08, 0.25, 10, 1, 1))
NORMALIZATION = {"method": "fixed-public-sensor-scales-v1", "clip": [-10, 10]}
HEADER_KEYS = frozenset({"schemaVersion", "algorithm", "policyContractName", "policyContractVersion", "observationSchema",
    "observationSize", "actionSize", "featureTransform", "featureInputSize", "scales", "normalization", "width",
    "hiddenLayers", "trainingSeed", "steps", "updates", "trained", "identity"})


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def same_value(actual, expected):
    if type(expected) in (int, float):
        return finite(actual) and actual == expected
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        return set(actual) == set(expected) and all(same_value(actual[name], value) for name, value in expected.items())
    if type(expected) is list:
        return len(actual) == len(expected) and all(same_value(left, right) for left, right in zip(actual, expected))
    return actual == expected


def bounded_integer(value, maximum, label):
    require(type(value) is int and 0 <= value <= maximum, "Bounded integer required: " + label)
    return value


def sensor_vector(observation):
    require(type(observation) is dict and set(observation) == set(TAGS) | {name for name, _ in FIELDS},
            "Only explicitly tagged GVF sensor fields are admitted")
    require(type(observation["contractVersion"]) is int and all(observation[name] == value for name, value in TAGS.items()),
            "Explicit GVF name/version/schema required; vector length is not compatibility")
    values = []
    for name, width in FIELDS:
        entries = [observation[name]] if width == 1 else observation[name]
        require(type(entries) is list and len(entries) == width and all(finite(value) for value in entries),
                "Finite exact GVF sensor field required: " + name)
        values.extend(entries)
    require(all(-1 <= value <= 1 for value in observation["previousAction"]) and
            observation["remainingSeconds"] >= 0 and 0 <= observation["projection"] <= 1 and
            observation["remainingArcMeters"] >= 0 and observation["crossTrackMeters"] >= 0,
            "Bounded GVF action/deadline/geometry context required")
    age = observation["actionAgeTicks"]
    require(float(age).is_integer() and 0 <= age <= 6000, "Bounded integer physics action age required")
    require(len(values) == OBSERVATION_SIZE, "GVF sensor vector size mismatch")
    return values


def observation_from_vector(values, *, contract_name, contract_version, observation_schema):
    require(type(values) is list and len(values) == OBSERVATION_SIZE, "Explicit complete GVF sensor vector required")
    observation = {"contractName": contract_name, "contractVersion": contract_version, "observationSchema": observation_schema}
    offset = 0
    for name, width in FIELDS:
        observation[name] = values[offset] if width == 1 else values[offset:offset + width]
        offset += width
    sensor_vector(observation)
    return observation


def encode_observation(observation):
    raw = sensor_vector(observation)
    features = list(raw)
    cosine, sine = math.cos(raw[2]), math.sin(raw[2])
    for right_index, forward_index in ((3, 4), (18, 19), (21, 22), (27, 28), (30, 31)):
        right, forward = raw[right_index], raw[forward_index]
        if right_index == 18:
            right -= raw[0]
            forward -= raw[1]
        features[right_index] = right * cosine - forward * sine
        features[forward_index] = right * sine + forward * cosine
    difference = raw[20] - raw[2]
    features[2] = sine
    features[20] = math.sin(difference)
    features.extend((cosine, math.cos(difference)))
    require(len(features) == FEATURE_SIZE and all(finite(value) for value in features), "Finite GVF model features required")
    return features


def validate_architecture(width, hidden_layers):
    require(type(width) is int and width in (32, 64, 128, 256, 512), "GVF network width must be32,64,128,256,512")
    require(type(hidden_layers) is int and hidden_layers in (2, 10), "GVF hidden layer count must be2 or10")
    return width, hidden_layers


def artifact_header(identity, training_seed, steps, updates, width, hidden_layers):
    validate_architecture(width, hidden_layers)
    bounded_integer(training_seed, 0xffffffff, "training seed")
    bounded_integer(steps, 10000000, "steps")
    bounded_integer(updates, 100000, "updates")
    require(type(identity) is dict and bool(identity) and len(json.dumps(identity, allow_nan=False).encode()) <= 256 * 1024,
            "Bounded explicit original-engine identity required")
    return {"schemaVersion": 1, "algorithm": ALGORITHM, "policyContractName": CONTRACT_NAME,
            "policyContractVersion": CONTRACT_VERSION, "observationSchema": OBSERVATION_SCHEMA,
            "observationSize": OBSERVATION_SIZE, "actionSize": ACTION_SIZE, "featureTransform": FEATURE_TRANSFORM,
            "featureInputSize": FEATURE_SIZE, "scales": list(SCALES), "normalization": copy.deepcopy(NORMALIZATION),
            "width": width, "hiddenLayers": hidden_layers, "trainingSeed": training_seed, "steps": steps,
            "updates": updates, "trained": updates > 0, "identity": copy.deepcopy(identity)}


def validate_actor_artifact(value, expected_identity=None):
    require(type(value) is dict and set(value) == HEADER_KEYS | {"layers"}, "Exact distinct GVF actor artifact fields required")
    expected = artifact_header(value["identity"], value["trainingSeed"], value["steps"], value["updates"], value["width"], value["hiddenLayers"])
    require(type(value["trained"]) is bool and all(same_value(value[name], item)
            for name, item in expected.items()), "GVF algorithm/contract/features/scales/metadata mismatch")
    if expected_identity is not None:
        require(same_value(value["identity"], expected_identity), "Original Simulator/controller identity mismatch")
    sizes = [(FEATURE_SIZE, value["width"])] + [(value["width"], value["width"])] * (value["hiddenLayers"] - 1) + [(value["width"], ACTION_SIZE)]
    require(type(value["layers"]) is list and len(value["layers"]) == len(sizes), "Exact GVF layer count required")
    for layer, (inputs, outputs) in zip(value["layers"], sizes):
        require(type(layer) is dict and set(layer) == {"weight", "bias"} and type(layer["weight"]) is list and
                len(layer["weight"]) == outputs and type(layer["bias"]) is list and len(layer["bias"]) == outputs and
                all(finite(item) and abs(item) <= 10000 for item in layer["bias"]) and
                all(type(row) is list and len(row) == inputs and all(finite(item) and abs(item) <= 10000 for item in row)
                    for row in layer["weight"]), "Finite bounded exact GVF layer dimensions required")
    return value
