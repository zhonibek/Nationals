import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from urllib.parse import urlsplit


URL = os.environ.get("COSMOS_TEST_URL")
HAS_UI = all(importlib.util.find_spec(name) for name in ("gradio_client", "httpx", "PIL"))


@unittest.skipUnless(URL and HAS_UI, "Explicit local Cosmos test server and optional UI dependencies required")
class CosmosNativeUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        parsed = urlsplit(URL)
        if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or parsed.path not in ("", "/"):
            raise ValueError("This integration test only contacts a declared loopback server")
        import httpx
        from gradio_client import Client

        cls.httpx = httpx
        cls.client = Client(URL, verbose=False)

    def test_actual_endpoint_and_local_upload_reach_instruction_admission(self):
        from gradio_client import handle_file
        from PIL import Image

        response = self.httpx.get(URL + "/gradio_api/info", timeout=15)
        self.assertEqual(response.status_code, 200)
        self.assertIn("/analyze", response.json()["named_endpoints"])
        with tempfile.TemporaryDirectory() as temporary:
            filename = Path(temporary) / "scene.png"
            Image.new("RGB", (64, 64), (30, 60, 120)).save(filename)
            commentary, details, message = self.client.predict(handle_file(str(filename)), "", 128, api_name="/analyze")
            self.assertEqual(commentary, "")
            self.assertEqual(details["status"], "error")
            self.assertEqual(details["code"], "instruction")
            self.assertTrue(message)

    def test_real_http_guard_rejects_remote_and_outside_paths(self):
        for path in ("https://example.invalid/video.mp4", "/etc/passwd"):
            with self.subTest(path=path):
                payload = {"data": [{"path": path, "meta": {"_type": "gradio.FileData"}}, "Describe this", 128]}
                response = self.httpx.post(URL + "/gradio_api/call/analyze", json=payload, timeout=15)
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.json()["code"], "path")
        response = self.httpx.post(URL + "/gradio_api/call/analyze", content=b" " * 65537,
            headers={"content-type": "application/json"}, timeout=15)
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json()["code"], "upload")


if __name__ == "__main__":
    unittest.main()
