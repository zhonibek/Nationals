'use strict';

const {FEATURE_COUNT} = require('../../roboproof/tactical_environment');
const {POLICY_SCHEMA, MODEL_ID} = require('../../roboproof/tactical_policy');

// A tiny exported-policy record (not trained weights). By default hidden1 copies the first `width` features,
// hidden2 is the identity, and the policy head reads hidden unit `prefer`, so the score is tanh(tanh(x[prefer])) + bias.
function fixturePolicyRecord({width = 16, prefer = 0, bias = 0.5, overrides = {}} = {}) {
  const identity = (rows, columns) => Array.from({length: rows}, (_, row) => Array.from({length: columns}, (__, column) => row === column ? 1 : 0));
  return {schema: POLICY_SCHEMA, model: MODEL_ID, featureCount: FEATURE_COUNT, width, activation: 'tanh',
    algorithm: 'fixture weights for tests (not trained)',
    layers: {hidden1: {weight: identity(width, FEATURE_COUNT), bias: Array(width).fill(0)},
      hidden2: {weight: identity(width, width), bias: Array(width).fill(0)},
      policy: {weight: Array.from({length: width}, (_, index) => index === prefer ? 1 : 0), bias}},
    provenance: {runId: 'fixture-run', trainingSeed: 1, checkpointSha256: null},
    evaluation: {meanScoreMargin: 1, baselineMeanScoreMargin: {'loader-cycle': 2}},
    learnedImprovementVerified: false, independentQualificationPerformed: false, ...overrides};
}

module.exports = {fixturePolicyRecord};
