'use strict';

const {makeScenario, provenance} = require('../core');
const {runScenario} = require('../sim');
const {exportDataset} = require('./export-dataset');
const current = provenance();
const results = Array.from({length: 24}, (_, index) => runScenario(makeScenario(23000 + index, {
  environment: {mass: 5 + index / 5, friction: 0.09 + index / 25, battery_voltage: 9 + index / 8},
  task: {duration: 0.3}
})));
process.stdout.write(JSON.stringify(exportDataset(results, {allowLegacy: true, current})));
