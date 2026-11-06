import importlib.util
import io
import json
import tarfile
import tempfile
import unittest
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("build_training_dashboard", ROOT / "tools/build_training_dashboard.py")
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


class CurveTests(unittest.TestCase):
    def test_curves_start_once_the_window_is_full_and_stay_bounded(self):
        points = builder.moving_average_curve([float(value) for value in range(1000)], 10, 50)
        self.assertEqual(points[0], [10, 4.5])
        self.assertEqual(points[-1], [1000, 994.5])
        self.assertLessEqual(len(points), 50)
        self.assertEqual(builder.moving_average_curve([3.0, 5.0], 10), [[2, 4.0]])
        self.assertEqual(builder.moving_average_curve([], 10), [])


class BundleTests(unittest.TestCase):
    def test_only_complete_bundles_with_a_matching_sidecar_are_used(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            bundle = folder / "campaign.results.tar.gz"
            with tarfile.open(bundle, "w:gz") as archive:
                data = json.dumps({"runs": []}).encode()
                info = tarfile.TarInfo("campaign/campaign.json")
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
            self.assertFalse(builder.verified(bundle))
            sidecar = bundle.with_name(bundle.name + ".sha256")
            sidecar.write_text("0" * 64 + "  campaign.results.tar.gz\n", encoding="utf-8")
            self.assertFalse(builder.verified(bundle))
            sidecar.write_text(hashlib.sha256(bundle.read_bytes()).hexdigest() + "  campaign.results.tar.gz\n", encoding="utf-8")
            self.assertTrue(builder.verified(bundle))
            self.assertEqual(json.loads(builder.read_bundle(bundle)["campaign.json"]), {"runs": []})
            result = builder.build({"single": folder / "missing.tar.gz", "single_check": folder / "missing-check.tar.gz",
                                    "multi": folder / "missing-multi.tar.gz", "multi_check": folder / "x.tar.gz"})
            self.assertEqual(result["missing"], ["missing.tar.gz", "missing-multi.tar.gz"])
            self.assertIs(result["learnedImprovementVerified"], False)


if __name__ == "__main__":
    unittest.main()
