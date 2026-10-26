#!/usr/bin/env node
'use strict';

// One-command launcher for Windows, macOS and Linux: `node tools/start.js` (or `npm start`).
// A single RoboProof server serves the simulator, the portal and the RoboProof dashboard.
//   --port <n>   port to listen on (default 8766)
//   --no-open    do not open a browser

const fs = require('node:fs');
const path = require('node:path');
const {spawn} = require('node:child_process');

const ROOT = path.resolve(__dirname, '..');
const [major, minor] = process.versions.node.split('.').map(Number);
if (major < 20 || (major === 20 && minor < 3)) {
  console.error(`Node.js 20.3 or newer is required (found ${process.versions.node}). Install it from https://nodejs.org.`);
  process.exit(1);
}

function option(name) {
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] : undefined;
}
const port = Number(option('--port') ?? 8766);
if (!Number.isInteger(port) || port < 1024 || port > 65535) {
  console.error('--port must be an integer from 1024 to 65535');
  process.exit(1);
}
const openBrowser = !process.argv.includes('--no-open');
const base = `http://127.0.0.1:${port}`;
const portal = `${base}/simulator/portal.html`;

function open(url) {
  const [command, args] = process.platform === 'win32' ? ['cmd', ['/c', 'start', '', url]]
    : process.platform === 'darwin' ? ['open', [url]] : ['xdg-open', [url]];
  try {
    const child = spawn(command, args, {detached: true, stdio: 'ignore'});
    child.on('error', () => console.log(`Open ${url} in your browser.`));
    child.unref();
  } catch {
    console.log(`Open ${url} in your browser.`);
  }
}

function nemotronStatus() {
  const keyFile = path.join(ROOT, '.cache/robotai-nemotron/nebius-api-key.txt');
  if (process.env.NEBIUS_API_KEY || process.env.ROBOTAI_NEMOTRON_API_KEY) return 'Nebius Token Factory key from the environment';
  if (fs.existsSync(keyFile) && fs.readFileSync(keyFile, 'utf8').trim()) return 'Nebius Token Factory key from .cache/';
  return 'no Nebius key yet: click the Nebius badge in the portal, or run tools/connect-nebius.ps1 (Windows) / set NEBIUS_API_KEY';
}

function banner(alreadyRunning) {
  console.log('Nationals Physical AI');
  console.log(`  Portal:     ${portal}${alreadyRunning ? '   (server was already running)' : ''}`);
  console.log(`  Simulator:  ${base}/simulator/index.html`);
  console.log(`  RoboProof:  ${base}/`);
  console.log(`  Nemotron:   ${nemotronStatus()}`);
  if (!alreadyRunning) console.log('Press Ctrl+C to stop.');
}

const {createServer} = require(path.join(ROOT, 'roboproof/server.js'));
const server = createServer();
server.on('error', async error => {
  if (error.code !== 'EADDRINUSE') {
    console.error(error.message);
    process.exit(1);
  }
  // Something already listens on the port: reuse it only if it is a RoboProof server.
  try {
    const response = await fetch(`${base}/api/nemotron/status`);
    if (!response.ok) throw Error(`HTTP ${response.status}`);
    banner(true);
    if (openBrowser) open(portal);
  } catch {
    console.error(`Port ${port} is in use by another program. Try: node tools/start.js --port 8770`);
    process.exit(1);
  }
});
server.listen(port, '127.0.0.1', () => {
  banner(false);
  if (openBrowser) open(portal);
});
process.on('SIGINT', () => server.close(() => process.exit(0)));
