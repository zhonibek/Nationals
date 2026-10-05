'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {spawnSync} = require('node:child_process');
const root = path.resolve(__dirname, '..');

test('project pins ARM-compatible C and C++ standards before common.mk defaults', () => {
  const project = fs.readFileSync(path.join(root, 'Makefile'), 'utf8');
  assert.match(project, /^C_STANDARD\s*:=\s*gnu11\s*$/m);
  assert.match(project, /^CXX_STANDARD\s*:=\s*gnu\+\+20\s*$/m);
  assert(project.indexOf('CXX_STANDARD :=') < project.indexOf('-include ./common.mk'));
  assert.doesNotMatch(project, /gnu\+\+26|gnu23/);
});

test('GNU Make emits supported flags even when the environment suggests unsupported standards', context => {
  const executable = process.env.ROBOTAI_MAKE || 'make';
  const environment = {...process.env, C_STANDARD: 'gnu23', CXX_STANDARD: 'gnu++26'};
  const probe = spawnSync(executable, ['--version'], {encoding: 'utf8', timeout: 10000, env: environment});
  if (probe.error?.code === 'ENOENT') {
    context.skip('GNU Make unavailable; run with ROBOTAI_MAKE pointing to the installed PROS make');
    return;
  }
  assert.equal(probe.status, 0, probe.stderr);
  assert.match(probe.stdout, /GNU Make/);
  const result = spawnSync(executable, ['--no-print-directory', '-f', 'Makefile', '-f', '-',
    '--dry-run', 'robotai-standard-check', ...(process.env.ROBOTAI_MAKE_SHELL ? [`SHELL=${process.env.ROBOTAI_MAKE_SHELL}`] : [])],
    {cwd: root, env: environment, encoding: 'utf8', timeout: 15000,
      input: 'robotai-standard-check:\n\t@echo $(CFLAGS)\n\t@echo $(CXXFLAGS)\n'});
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /--std=gnu11(?:\s|$)/);
  assert.match(result.stdout, /--std=gnu\+\+20(?:\s|$)/);
  assert.doesNotMatch(result.stdout, /--std=gnu\+\+26|--std=gnu23/);
});
