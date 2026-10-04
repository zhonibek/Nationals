'use strict';

const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const Control = require('./control-runtime');
const {MotionEpisode} = require('./motion');
const root = path.resolve(__dirname, '..');
const hash = bytes => crypto.createHash('sha256').update(bytes).digest('hex');

function loadHeadless() {
  const manifest = JSON.parse(fs.readFileSync(path.join(__dirname, 'control-build.json'), 'utf8').replace(/^\uFEFF/, ''));
  for (const [filename, digest] of Object.entries(manifest.sources)) {
    const source = fs.readFileSync(path.join(root, filename), 'utf8').replace(/^\uFEFF/, '').replace(/\r\n/g, '\n');
    if (hash(source) !== digest) throw Error(`Stale production controller source: ${filename}`);
  }
  const bytes = fs.readFileSync(path.join(__dirname, 'control.wasm'));
  if (hash(bytes) !== manifest.wasm) throw Error('Stale production control.wasm');
  const compiled = new WebAssembly.Module(bytes);
  const filenames = ['engine.js', 'motion.js', 'policy.js', 'headless.js', 'control-runtime.js', 'robot-config.js', 'override.js', 'override-geometry.js', 'override-dynamics.js', 'override-autonomy.js'];
  const sources = Object.fromEntries(filenames.map(filename => [filename,
    hash(fs.readFileSync(path.join(__dirname, filename), 'utf8').replace(/\r\n/g, '\n'))]));
  sources['config/robot.json'] = hash(fs.readFileSync(path.join(root, 'config/robot.json'), 'utf8').replace(/\r\n/g, '\n'));
  const identity = {engine: 'original VexRobotSimulator', canonicalSimulator: true,
    runtime: {node: process.version, v8: process.versions.v8, platform: process.platform, arch: process.arch},
    controllerWasmSha256: manifest.wasm, sources, simulatedFidelity: 'approximate, not physical validation'};
  return {identity, createEpisode: () => new MotionEpisode(() => Control.fromModule(compiled))};
}

module.exports = {loadHeadless};
