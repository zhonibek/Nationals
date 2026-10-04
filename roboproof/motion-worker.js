'use strict';

const {parentPort, workerData} = require('node:worker_threads');
const {baseline, runPolicy, replay} = require('./motion');
const {normalizeTask} = require('../simulator/motion');
const {verify} = require('./motion-readiness');
const {evaluate} = require('./motion-evaluation');
const {writeJson} = require('./core');
const path = require('node:path');

try {
  if (workerData.operation === 'check') parentPort.postMessage(verify());
  else if (workerData.operation === 'evaluate') {
    const result = evaluate(2);
    writeJson(path.join(__dirname, 'runs/motion/evaluation.json'), result);
    parentPort.postMessage(result);
  }
  else if (workerData.operation === 'replay') {
    replay(workerData.report);
    parentPort.postMessage({exactReplayVerified: true, ticks: workerData.report.ticks,
      engine: workerData.report.identity.engine});
  } else if (workerData.operation === 'run') {
    const task = normalizeTask(workerData.task);
    let state = workerData.seed;
    const report = workerData.mode === 'scripted' ? baseline(workerData.seed, task, workerData.configuration, {recordTransitions: true})
      : runPolicy({seed: workerData.seed, task, configuration: workerData.configuration,
        options: {policyIdentity: {kind: 'scripted', id: workerData.mode}},
        ...(workerData.mode === 'random-fixture' ? {act: () => Array.from({length: 4}, () => {
          state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
          return state / 4294967296 * 2 - 1;
        })} : {})});
    replay(JSON.parse(JSON.stringify(report)));
    parentPort.postMessage({report, exactReplayVerified: true, inferencePerformed: false, motionPolicyTrained: false});
  } else throw Error('Unsupported motion worker operation');
} catch (error) { parentPort.postMessage({error: error.message}); }
