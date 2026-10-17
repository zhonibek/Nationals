'use strict';

const fs = require('node:fs');
const path = require('node:path');
const {createHash} = require('node:crypto');
const {baseline, runGvf, replay} = require('../roboproof/motion');
const {suite, aggregate} = require('../roboproof/motion-evaluation');
const {loadHeadless} = require('../simulator/headless');

function evaluate(directory) {
  if (process.platform !== 'linux' || process.env.ROBOPROOF_APPROVED_REMOTE !== '1') throw Error('Run this simulation campaign only on an explicitly approved remote Linux server');
  if (!directory) throw Error('An unused evidence directory is required');
  const corpus = suite(2);
  const identity = loadHeadless().identity;
  fs.mkdirSync(directory, {recursive: false});
  const write = (name, value) => fs.writeFileSync(path.join(directory, name), JSON.stringify(value, null, 2) + '\n', {flag: 'wx'});
  write('worlds.json', {schemaVersion: 1, kind: 'gvf-v2-fixed-development-comparison', corpusSha256: corpus.corpusSha256,
    cases: corpus.cases, scope: 'All frozen v2 worlds declared before execution; development exposure, not a fresh independent final'});
  write('protocol.json', {schemaVersion: 1, identity, cases: corpus.cases.length, controller: 'unchanged scripted baseline',
    candidate: 'Pedro GVF v2, fixed defaults, same C++ cascade', learnedPolicyEvaluated: false,
    noParameterSearch: true, preserveFailures: true, physicsChanged: false});
  const originals = [], fields = [], cases = [];
  for (const [index, world] of corpus.cases.entries()) {
    const original = baseline(world.seed, world.task, world.configuration);
    const candidate = runGvf(world.seed, world.task, world.configuration);
    replay(structuredClone(original));
    replay(structuredClone(candidate));
    write(`case-${String(index).padStart(2, '0')}.json`, {world, original, candidate, exactReplayVerified: true});
    originals.push(original);fields.push(candidate);
    cases.push({...world, controller: {reason: original.reason, metrics: original.metrics}, gvf: {reason: candidate.reason, metrics: candidate.metrics}});
    process.stdout.write(JSON.stringify({case: world.id, controller: original.reason, gvf: candidate.reason}) + '\n');
  }
  const result = {schemaVersion: 1, identity, corpusSha256: corpus.corpusSha256, cases,
    controller: aggregate(originals), gvf: aggregate(fields),
    controllerSuccessRegressions: cases.filter(row => row.controller.reason === 'success' && row.gvf.reason !== 'success').map(row => row.id),
    contactRegressions: cases.filter(row => row.gvf.metrics.contactSeconds > row.controller.metrics.contactSeconds).map(row => row.id),
    motionPolicyTrained: false, learnedImprovementVerified: false, independentQualificationPerformed: false,
    interpretation: 'Fixed development comparison only. All failures included. GVF is analytic; no neural policy, game scoring, GPU training or physical validation.'};
  write('summary.json', result);
  const files = fs.readdirSync(directory).sort().map(name => {
    const bytes = fs.readFileSync(path.join(directory, name));
    return {path: name, bytes: bytes.length, sha256: createHash('sha256').update(bytes).digest('hex')};
  });
  write('result-manifest.json', {schemaVersion: 1, files});
  return result;
}

if (require.main === module) {
  const result = evaluate(process.argv[2]);
  process.stdout.write(JSON.stringify({controller: result.controller, gvf: result.gvf,
    controllerSuccessRegressions: result.controllerSuccessRegressions, learnedImprovementVerified: false}) + '\n');
}

module.exports = {evaluate};
