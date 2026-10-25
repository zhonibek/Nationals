'use strict';

const readline = require('node:readline');
const {TacticalEnvironment} = require('../tactical_environment');
const {evaluatePolicies, demonstrate} = require('../tactical_baselines');

const environment = new TacticalEnvironment();
const input = readline.createInterface({input: process.stdin, crlfDelay: Infinity});

function respond(value, callback) {
  process.stdout.write(JSON.stringify(value) + '\n', callback);
}

let closing = false;
input.on('line', line => {
  if (closing) return;
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
    } else if (request.operation === 'demonstrate') {
      // The non-learning tactic's choice in the current state, for behavior cloning; read-only, null when none.
      const policy = request.policy ?? 'loader-cycle';
      const candidate = demonstrate(environment, policy);
      respond({ok: true, policy, actionId: candidate ? candidate.id : null, candidate});
    } else if (request.operation === 'close') {
      // input.close() alone does not end the process while the trainer keeps stdin open: it runs inside
      // stdin's 'data' event, so the stream's read-ahead restarts the pipe read right after the pause
      // stopped it, and that read keeps the event loop alive until EOF. Exit once the answer is flushed.
      closing = true;
      input.close();
      respond({ok: true}, error => process.exit(error ? 1 : 0));
    } else throw Error('Unsupported tactical bridge operation');
  } catch (error) {
    respond({ok: false, error: error instanceof Error ? error.message : String(error)});
  }
});
