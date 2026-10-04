'use strict';

const {parentPort, workerData} = require('node:worker_threads');
const {baseline, replay} = require('./motion');
const {normalizeTask} = require('../simulator/motion');

try {
  const task = normalizeTask(workerData.task);
  const report = baseline(42, task);
  replay(report);
  parentPort.postMessage({report, exactReplayVerified: true});
} catch (error) { parentPort.postMessage({error: error.message}); }
