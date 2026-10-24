'use strict';

const readline = require('node:readline');
const {TacticalEnvironment} = require('../tactical_environment');
const {evaluatePolicies} = require('../tactical_baselines');

const environment = new TacticalEnvironment();
const input = readline.createInterface({input: process.stdin, crlfDelay: Infinity});

function respond(value) {
  process.stdout.write(JSON.stringify(value) + '\n');
}

input.on('line', line => {
  try {
    const request = JSON.parse(line);
    if (request.operation === 'reset') respond({ok: true, observation: environment.reset(request.seed, request.alliance)});
    else if (request.operation === 'step') respond({ok: true, ...environment.step(request.actionId)});
    else if (request.operation === 'baselines') {
      // Non-learning comparison on the same held-out seeds; answered asynchronously, one request at a time.
      evaluatePolicies({seeds: request.seeds, alliance: request.alliance})
        .then(results => respond({ok: true, baselines: Object.fromEntries(Object.entries(results).map(([name, value]) =>
          [name, {meanMargin: value.meanMargin, margins: value.matches.map(match => match.finalScore.margin)}]))}))
        .catch(error => respond({ok: false, error: error.message}));
    } else if (request.operation === 'close') {
      respond({ok: true});
      input.close();
    } else throw Error('Unsupported tactical bridge operation');
  } catch (error) {
    respond({ok: false, error: error instanceof Error ? error.message : String(error)});
  }
});
