'use strict';

const fs = require('node:fs');
const path = require('node:path');
const {execFile} = require('node:child_process');
const {evidenceTools} = require('./analysis');
const {writeJson} = require('./core');
const instructions = `You are a robotics diagnosis agent. Separate measured observations from hypotheses.
Never claim physical validation or PID-term evidence that tools do not expose.
Choose one tool call at a time. Reply with JSON only:
{"type":"tool","name":"get_scenario","arguments":{}}
or {"type":"final","observations":["..."],"hypotheses":["..."],"evidence":[0,1],"limitations":["..."]}.
Evidence indices refer to recorded tool results. Do not change files or propose a safety certification.
Available tools: get_scenario(), inspect_controller(), find_first_divergence(),
get_telemetry_range({start,end,limit<=100}), run_scenario(),
inspect_pid_terms(), compare_runs({scenario}),
inspect_source({filename,start,count<=150}) with filename from
include/subsystems/control/Cascade.hpp, src/subsystems/control/HolonomicMotion.cpp,
src/lemlib/chassis/odom.cpp,
run_parameter_sweep({parameter,values:[up to 12 values within the scenario bounds]}).
Observe a failure, inspect source/telemetry, and request a counterfactual sweep before concluding.`;

async function invoke(command, context) {
  if (!Array.isArray(command) || !command.length || command.some(part => typeof part !== 'string' || !part)) throw Error('Model command must be a nonempty argv array');
  const input = JSON.stringify(context);
  const child = execFile(command[0], command.slice(1), {encoding: 'utf8', timeout: 45000, maxBuffer: 256 * 1024, windowsHide: true});
  child.stdin.on('error', () => {});
  child.stdin.end(input);
  const result = await new Promise((resolve, reject) => {
    let stdout = '', stderr = '';
    child.stdout.on('data', value => { stdout += value; });
    child.stderr.on('data', value => { stderr += value; });
    child.once('error', reject);
    child.once('close', (code, signal) => {
      if (code !== 0 || signal) reject(Error(`Model command failed (${signal || code}): ${stderr.slice(0, 1000)}`));
      else resolve(stdout);
    });
  });
  return JSON.parse(result);
}

async function investigate(result, command, {maxTurns = 12, provider = invoke} = {}) {
  if (!Number.isInteger(maxTurns) || maxTurns < 1 || maxTurns > 20) throw Error('maxTurns must be 1..20');
  const tools = evidenceTools(result);
  const requested = [];
  let simulations = 0;
  for (let turn = 0; turn < maxTurns; turn++) {
    const response = await provider(command, {instructions, failure: {scenario_id: result.scenario.scenario_id, categories: result.categories, metrics: result.metrics}, history: tools.log});
    if (response.type === 'final') {
      for (const field of ['observations', 'hypotheses', 'limitations']) if (!Array.isArray(response[field]) || response[field].some(value => typeof value !== 'string')) throw Error(`Agent final requires ${field}: string[]`);
      if (!Array.isArray(response.evidence) || response.evidence.some(index => !Number.isInteger(index) || index < 0 || index >= tools.log.length)) throw Error('Agent cited missing tool evidence');
      if (!response.evidence.length || !requested.includes('inspect_controller') || !requested.includes('run_parameter_sweep') || !requested.some(name => ['find_first_divergence', 'get_telemetry_range'].includes(name))) throw Error('Agent must inspect source, telemetry and a counterfactual before concluding');
      const deterministic = command.length === 2 && path.resolve(command[1]) === path.join(__dirname, 'local_agent.js');
      return {mode: deterministic ? 'deterministic-scripted-tool-agent' : 'external-model-tool-agent',
        llm_status: deterministic ? 'NOT CONNECTED; fixed-rule bridge' : 'Configured external bridge; inspect its provider metadata',
        verification: 'Evidence references validated; natural-language hypotheses require review and do not prove causation', diagnosis: response, tool_log: tools.log};
    }
    if (response.type !== 'tool' || typeof response.name !== 'string' || !response.arguments || typeof response.arguments !== 'object' || Array.isArray(response.arguments)) throw Error('Invalid agent tool response');
    const cost = response.name === 'run_parameter_sweep' ? response.arguments.values?.length || 0 : ['run_scenario', 'compare_runs'].includes(response.name) ? 1 : 0;
    if (simulations + cost > 36) throw Error('Agent simulation budget exhausted');
    simulations += cost;
    tools.call(response.name, response.arguments);
    requested.push(response.name);
  }
  throw Error('Agent reached its turn budget without a supported diagnosis');
}

if (require.main === module) {
  const [file, commandFile, output = 'roboproof/runs/agent-diagnosis.json'] = process.argv.slice(2);
  if (!file || !commandFile) { console.error('Usage: node roboproof/agent.js counterexample.json model-command.json [out.json]'); process.exitCode = 1; }
  else {
    Promise.resolve().then(() => investigate(JSON.parse(fs.readFileSync(file, 'utf8')), JSON.parse(fs.readFileSync(commandFile, 'utf8'))))
      .then(diagnosis => { writeJson(output, diagnosis); console.log(`Wrote ${output}`); })
      .catch(error => { console.error(error.message); process.exitCode = 1; });
  }
}

module.exports = {investigate, invoke};
