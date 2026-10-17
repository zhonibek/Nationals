import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from roboproof.cosmos import motion_media


@unittest.skipUnless(importlib.util.find_spec('PIL'), 'Pillow is required for actual pose visualization')
class CosmosMotionExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name)
        self.root = self.project / 'roboproof'
        self.session_id = '00000000-0000-4000-8000-000000000001'
        self.filename = self.root / f'runs/motion/sessions/{self.session_id}.json'
        self.filename.parent.mkdir(parents=True)
        self.record = {'id': self.session_id, 'exactReplayVerified': True, 'report': {
            'fixedDt': 0.01, 'task': {'start': {'xIn': 0, 'yIn': 0, 'headingDeg': 0},
                                  'goal': {'xIn': 0, 'yIn': 2, 'headingDeg': 0}},
            'transitions': [{'tick': index, 'evaluator': {'poseInchesDegrees': [0, index + 1, 0]}} for index in range(3)]}}
        self.filename.write_text(json.dumps(self.record))
        self.root_patch = patch.object(motion_media, 'ROOT', self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def test_actual_pngs_bind_original_bytes_and_zero_based_post_step_times(self):
        result = motion_media.export_session(self.session_id, frame_count=4)
        self.assertFalse(result['modelInferencePerformed'])
        self.assertEqual(result['newSimulatorQueries'], 0)
        with zipfile.ZipFile(result['path']) as archive:
            manifest = json.loads(archive.read('manifest.json'))
            self.assertEqual(manifest['sessionSha256'], hashlib.sha256(self.filename.read_bytes()).hexdigest())
            self.assertEqual([frame['recordedTick'] for frame in manifest['frames']], [None, 0, 1, 2])
            self.assertEqual([frame['recordedTimeSeconds'] for frame in manifest['frames']], [0, 0.01, 0.02, 0.03])
            for frame in manifest['frames']:
                data = archive.read(frame['path'])
                self.assertTrue(data.startswith(b'\x89PNG\r\n\x1a\n'))
                self.assertEqual(hashlib.sha256(data).hexdigest(), frame['sha256'])
        with self.assertRaises(ValueError):
            motion_media.export_session(self.session_id)

    def test_missing_or_noncontiguous_evidence_never_creates_a_video_package(self):
        self.record['report']['transitions'][1]['tick'] = 3
        self.filename.write_text(json.dumps(self.record))
        with self.assertRaises(ValueError):
            motion_media.export_session(self.session_id)
        self.assertFalse((self.project / '.cache/cosmos-motion-media').exists())

    def test_unverified_sessions_and_external_output_paths_are_rejected(self):
        with self.assertRaises(ValueError):
            motion_media.export_session(self.session_id, directory=self.project / 'elsewhere')
        self.record['exactReplayVerified'] = False
        self.filename.write_text(json.dumps(self.record))
        with self.assertRaises(ValueError):
            motion_media.export_session(self.session_id)
