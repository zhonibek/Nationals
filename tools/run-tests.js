#!/usr/bin/env node
'use strict';

// Runs every Node test file in tests/ on any platform: `node tools/run-tests.js` (or `npm test`).
const fs = require('node:fs');
const path = require('node:path');
const {spawnSync} = require('node:child_process');

const directory = path.resolve(__dirname, '../tests');
const files = fs.readdirSync(directory).filter(name => name.endsWith('.test.js')).sort().map(name => path.join(directory, name));
const result = spawnSync(process.execPath, ['--test', ...files], {stdio: 'inherit'});
process.exit(result.status ?? 1);
