'use strict';

const {parentPort, workerData} = require('node:worker_threads');
const {performance} = require('node:perf_hooks');
const {makeScenario, sampleScenarios, provenance, spec} = require('./core');
const {runScenario} = require('./sim');
const {summary} = require('./analysis');

try {
  if (workerData.action === 'replay') {
    parentPort.postMessage(runScenario(workerData.scenario, {telemetry: true}));
  } else if (workerData.action === 'run') {
    const scenarios = sampleScenarios(workerData.count, workerData.seed);
    const nominal = runScenario(makeScenario(workerData.seed));
    const started = performance.now();
    const results = scenarios.map(scenario => runScenario(scenario));
    const seconds = (performance.now() - started) / 1000;
    const worst = results.reduce((left, right) => right.failure_score > left.failure_score ? right : left);
    parentPort.postMessage({schema_version: 1, ...provenance(), backend: 'cpu-wasm', distribution: spec.distribution,
      nominal, scenarios, results, summary: summary(results), benchmark: {seconds, simulations_per_second: scenarios.length / seconds, gpu: 'NOT MEASURED'},
      counterexample: runScenario(worst.scenario, {telemetry: true}), worst_scenario_id: worst.scenario.scenario_id});
  } else throw Error('Unsupported worker action');
} catch (error) {
  parentPort.postMessage({error: error.message});
}
