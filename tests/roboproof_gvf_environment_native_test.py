import hashlib
import json
import math
import os
from pathlib import Path
import unittest
import uuid

from roboproof.motion_env import MotionEnvironment


class GvfEnvironmentNativeTests(unittest.TestCase):
    def test_three_tick_original_bridge_contract_without_model_training(self):
        root = Path(__file__).resolve().parents[1]
        directory = root / 'roboproof/runs/motion/research' / ('gvf-residual-python-contract-smoke-' + str(uuid.uuid4()))
        directory.mkdir()
        task = {'start': {'xIn': 0, 'yIn': 0, 'headingDeg': 0},
                'goal': {'xIn': 0, 'yIn': 24, 'headingDeg': 0}, 'deadlineSeconds': 0.03}
        world = {'seed': 1874401, 'task': task, 'configuration': {}}

        def write(name, value):
            with (directory / name).open('x', encoding='utf-8') as output:
                json.dump(value, output, indent=2, allow_nan=False)
                output.write('\n')

        write('worlds.json', {'schemaVersion': 1, 'cases': [world], 'maxPhysicsTicks': 3,
                             'scope': 'Focused bridge contract test only, not training, teacher collection, a campaign or an independent final'})
        receipt = {'schemaVersion': 1, 'hostPlatform': os.name, 'modelTrainingPerformed': False,
                   'teacherCollectionPerformed': False, 'campaignPerformed': False, 'learnedImprovementVerified': False}
        try:
            with MotionEnvironment(control_mode='gvf-residual') as environment:
                self.assertEqual(environment.contract['actionSize'], 3)
                self.assertFalse(environment.contract['motionTrainingImplemented'])
                observation, info = environment.reset(**world)
                self.assertEqual(len(observation), 34)
                self.assertEqual(info['contractVersion'], 2)
                self.assertNotIn('referenceDurationSeconds', info)
                observation, reward, terminated, truncated, step_info = environment.step([0.25, -0.25, 0.5])
                self.assertFalse(terminated)
                self.assertTrue(truncated)
                self.assertTrue(math.isfinite(reward))
                self.assertEqual(step_info['physicsTicks'], 3)
                self.assertTrue(all(math.isfinite(value) for value in observation))
                report = environment.report()
                self.assertEqual(report['ticks'], 3)
                self.assertEqual(report['policyContractName'], 'gvf-residual')
                self.assertEqual(report['reason'], 'time_limit')
                write('original-report.json', report)
                receipt['controllerWasmSha256'] = report['identity']['controllerWasmSha256']
                receipt['physicsTicksExecuted'] = 3
                receipt['passed'] = True
        except Exception as error:
            receipt['passed'] = False
            receipt['error'] = repr(error)
            raise
        finally:
            write('native-test-receipt.json', receipt)
            files = [{'path': filename.name, 'bytes': filename.stat().st_size,
                      'sha256': hashlib.sha256(filename.read_bytes()).hexdigest()} for filename in sorted(directory.iterdir())]
            write('preservation-manifest.json', {'schemaVersion': 1, 'files': files})
            print(json.dumps({'evidenceDirectory': str(directory), **receipt}))


if __name__ == '__main__':
    unittest.main()
