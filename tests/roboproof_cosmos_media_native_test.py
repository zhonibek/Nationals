import importlib.util
from pathlib import Path
import tempfile
import unittest

from roboproof.cosmos.config import RuntimeConfig
from roboproof.cosmos.media import prepare_media


HAS_MEDIA = all(importlib.util.find_spec(name) for name in ("av", "PIL", "numpy"))


@unittest.skipUnless(HAS_MEDIA, "Pinned optional media dependencies not installed; no native decode claimed")
class CosmosNativeMediaTests(unittest.TestCase):
    def test_actual_png_decode_preserves_bounded_image(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            filename = root / "scene.png"
            Image.new("RGB", (640, 480), (30, 60, 120)).save(filename)
            prepared = prepare_media(filename, RuntimeConfig(media_roots=(root,)))
            self.assertEqual(prepared.kind, "image")
            self.assertEqual(len(prepared.frames), 1)
            self.assertEqual(prepared.frames[0].mode, "RGB")
            self.assertEqual(prepared.frames[0].size, (576, 416))
            self.assertLessEqual(576 * 416, 262144)
            self.assertEqual(prepared.frames[0].getpixel((20, 20)), (30, 60, 120))

    def test_actual_mp4_decode_covers_motion_with_real_times(self):
        import av
        import numpy as np

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            filename = root / "motion.mp4"
            with av.open(str(filename), mode="w") as output:
                stream = output.add_stream("mpeg4", rate=8)
                stream.width = 320
                stream.height = 192
                stream.pix_fmt = "yuv420p"
                for index in range(16):
                    pixels = np.zeros((192, 320, 3), dtype=np.uint8)
                    pixels[64:128, 16 + index * 12:64 + index * 12, 2] = 255
                    frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                    for packet in stream.encode(frame):
                        output.mux(packet)
                for packet in stream.encode():
                    output.mux(packet)
            prepared = prepare_media(filename, RuntimeConfig(media_roots=(root,)))
            self.assertEqual(prepared.kind, "video")
            self.assertEqual(prepared.source_frame_count, 16)
            self.assertEqual(prepared.source_fps, 8)
            self.assertAlmostEqual(prepared.duration_seconds, 2)
            self.assertEqual(len(prepared.frames), 8)
            self.assertEqual(prepared.timestamps_seconds[0], 0)
            self.assertGreaterEqual(prepared.timestamps_seconds[-1], 1.75)
            self.assertEqual(prepared.frames[0].size, (320, 192))
            first = np.asarray(prepared.frames[0])[:, :, 2].sum(axis=0).argmax()
            last = np.asarray(prepared.frames[-1])[:, :, 2].sum(axis=0).argmax()
            self.assertGreater(last - first, 140)


if __name__ == "__main__":
    unittest.main()
