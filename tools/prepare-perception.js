'use strict';

const fs = require('node:fs');
const {prepareProposal} = require('../roboproof/perception');

function main(argumentsValue) {
  const fixture = argumentsValue[0] === '--fixture';
  const filenames = fixture ? argumentsValue.slice(1) : argumentsValue;
  if (filenames.length !== 1) throw Error('Usage: node tools/prepare-perception.js [--fixture] input.json');
  const descriptor = fs.openSync(filenames[0], 'r');
  let input;
  try {
    const size = fs.fstatSync(descriptor).size;
    if (size < 1 || size > 65536) throw Error('Input must fit within 64 KiB');
    const buffer = Buffer.alloc(size + 1);
    const length = fs.readSync(descriptor, buffer, 0, buffer.length, 0);
    if (length > 65536 || length !== size) throw Error('Input changed or exceeded the input budget');
    input = JSON.parse(buffer.subarray(0, length).toString('utf8'));
  } finally { fs.closeSync(descriptor); }
  if (fixture !== (input.grounding?.source?.kind === 'fixture')) throw Error('Fixtures require --fixture; do not use that flag for real imported output');
  const proposal = prepareProposal(input, fixture ? {now: Date.parse(input.grounding.frame.capturedAt)} : {});
  proposal.validationClock = fixture ? 'fixture capture time; not a live camera freshness check' : 'current wall clock';
  process.stdout.write(`${JSON.stringify(proposal, null, 2)}\n`);
}

if (require.main === module) {
  try { main(process.argv.slice(2)); }
  catch (error) { console.error(error.message); process.exitCode = 1; }
}

module.exports = {main};
