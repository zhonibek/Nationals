import copy
import math
import unittest

from roboproof.motion_learning import gvf_contract as contract


def observation():
    result = {**contract.TAGS}
    for name, width in contract.FIELDS:
        result[name] = 0 if width == 1 else [0] * width
    result.update(pose=[0.5, 0.8, math.pi / 2], velocity=[1, 2, 0.3], targetError=[0.1, 0.2, 0, 1],
                  remainingSeconds=4, baseReference=[0.8, 1.2, math.pi, 0.1, 0.2, 0.4],
                  appliedFieldOffset=[0.02, 0.03, 0.01], offsetVelocity=[0.01, 0.02, 0.03])
    return result


def artifact(depth=2):
    value = contract.artifact_header({"engine": "mock-original-identity", "canonicalSimulator": True}, 17, 0, 0, 32, depth)
    sizes = [(36, 32)] + [(32, 32)] * (depth - 1) + [(32, 3)]
    value["layers"] = [{"weight": [[0] * inputs for _ in range(outputs)], "bias": [0] * outputs} for inputs, outputs in sizes]
    return value


class GvfModelContractTests(unittest.TestCase):
    def test_tagged_sensor_vector_roundtrip_never_relabels_four_action_data(self):
        source = observation()
        vector = contract.sensor_vector(source)
        rebuilt = contract.observation_from_vector(vector, contract_name=contract.CONTRACT_NAME,
            contract_version=contract.CONTRACT_VERSION, observation_schema=contract.OBSERVATION_SCHEMA)
        self.assertEqual(rebuilt, source)
        self.assertEqual(len(vector), 34)
        with self.assertRaises(ValueError):
            contract.observation_from_vector(vector, contract_name="policy", contract_version=1,
                observation_schema=contract.OBSERVATION_SCHEMA)
        with self.assertRaises(TypeError):
            contract.observation_from_vector(vector)

    def test_body_reference_encoding_is_pure_and_uses_sensor_relative_geometry(self):
        source = observation()
        original = copy.deepcopy(source)
        features = contract.encode_observation(source)
        self.assertEqual(source, original)
        self.assertEqual(len(features), 36)
        for index, expected in {2: 1, 3: -2, 4: 1, 18: -0.4, 19: 0.3, 20: 1,
                                21: -0.2, 22: 0.1, 27: -0.03, 28: 0.02, 34: 0, 35: 0}.items():
            self.assertAlmostEqual(features[index], expected, places=12)

    def test_extra_truth_fields_bad_tags_booleans_shapes_and_limits_rejected(self):
        for patch in ({"truth": [0, 0, 0]}, {"contractName": "policy"}, {"contractVersion": True},
                      {"pose": [False, 0, 0]}, {"baseReference": [0] * 5}, {"previousAction": [0, 0, 0, 0]},
                      {"previousAction": [0, 0, 1.1]}, {"projection": 1.01}, {"remainingSeconds": -1},
                      {"actionAgeTicks": 0.5}, {"actionAgeTicks": 6001}, {"remainingArcMeters": float("inf")}):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                contract.encode_observation({**observation(), **patch})
        with self.assertRaises(ValueError):
            contract.encode_observation([0] * 34)
        self.assertFalse(contract.finite(10 ** 1000))

    def test_both_architectures_and_fixed_float32_scales_are_explicit(self):
        for depth in (2, 10):
            model = artifact(depth)
            self.assertIs(contract.validate_actor_artifact(model, model["identity"]), model)
            self.assertEqual(len(model["layers"]), depth + 1)
            self.assertEqual(len(model["layers"][-1]["bias"]), 3)
        self.assertEqual(len(contract.SCALES), 36)
        for values in ((True, 2), (32, True), (31, 2), (32, 3)):
            with self.assertRaises(ValueError):
                contract.validate_architecture(*values)

    def test_old_algorithm_contract_normalization_identity_and_mutable_aliases_rejected(self):
        source = artifact()
        for patch in ({"algorithm": "td3-beta-mean-reference-v1"}, {"actionSize": 4},
                      {"policyContractVersion": 1}, {"featureInputSize": 34}, {"trained": True},
                      {"normalization": {"method": "fitted", "clip": [-10, 10]}}, {"unknown": 0}):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                contract.validate_actor_artifact({**source, **patch}, source["identity"])
        boolean_scale = copy.deepcopy(source)
        boolean_scale["scales"][2] = True
        with self.assertRaises(ValueError):
            contract.validate_actor_artifact(boolean_scale)
        wrong_identity = copy.deepcopy(source)
        wrong_identity["identity"]["canonicalSimulator"] = 1
        with self.assertRaises(ValueError):
            contract.validate_actor_artifact(wrong_identity, source["identity"])
        original = {"engine": "mock", "nested": [1]}
        header = contract.artifact_header(original, 0, 0, 0, 32, 2)
        original["nested"][0] = 2
        self.assertEqual(header["identity"]["nested"], [1])

    def test_sparse_malformed_or_excessive_model_weights_rejected(self):
        for mutate in (lambda value: value["layers"].pop(),
                       lambda value: value["layers"][-1]["bias"].append(0),
                       lambda value: value["layers"][0]["weight"][0].pop(),
                       lambda value: value["layers"][0]["bias"].__setitem__(0, float("nan")),
                       lambda value: value["layers"][0]["bias"].__setitem__(0, 10001),
                       lambda value: value["layers"][0]["bias"].__setitem__(0, True)):
            value = artifact()
            mutate(value)
            with self.assertRaises(ValueError):
                contract.validate_actor_artifact(value)


if __name__ == "__main__":
    unittest.main()
