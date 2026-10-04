'use strict';

async function main() {
  const chunks = [];
  for await (const chunk of process.stdin) {
    chunks.push(chunk);
  }
  const input = Buffer.concat(chunks).toString('utf8');
  if (!input.trim()) {
    console.error('Local Agent: No input received on stdin');
    process.exit(1);
  }

  const context = JSON.parse(input);
  const failure = context.failure || {};
  const history = context.history || [];

  const calledTools = history.map(item => item.tool);

  // Turn 1: Inspect controller source code if not done
  if (!calledTools.includes('inspect_controller')) {
    console.log(JSON.stringify({
      type: 'tool',
      name: 'inspect_controller',
      arguments: {}
    }));
    return;
  }

  // Turn 2: Find the exact first divergence point
  if (!calledTools.includes('find_first_divergence')) {
    console.log(JSON.stringify({
      type: 'tool',
      name: 'find_first_divergence',
      arguments: {}
    }));
    return;
  }

  // Turn 3: Inspect telemetry range around divergence
  if (!calledTools.includes('get_telemetry_range')) {
    const divItem = history.find(h => h.tool === 'find_first_divergence');
    const divTime = divItem?.result?.time ?? 2.0;
    const start = Math.max(0, divTime - 0.5);
    const end = divTime + 0.8;
    console.log(JSON.stringify({
      type: 'tool',
      name: 'get_telemetry_range',
      arguments: { start, end, limit: 15 }
    }));
    return;
  }

  // Turn 4: Run counterfactual parameter sweep
  if (!calledTools.includes('run_parameter_sweep')) {
    const categories = failure.categories || [];
    let sweepParam = 'friction';
    let sweepValues = [0.15, 0.45, 0.85, 1.05];

    if (categories.includes('HEADING_INSTABILITY') || categories.includes('OSCILLATION')) {
      sweepParam = 'encoder_latency';
      sweepValues = [0.0, 0.01, 0.02, 0.04];
    } else if (categories.includes('MOTOR_SATURATION') || categories.includes('TIMEOUT')) {
      sweepParam = 'battery_voltage';
      sweepValues = [9.0, 10.0, 11.0, 12.0];
    }

    console.log(JSON.stringify({
      type: 'tool',
      name: 'run_parameter_sweep',
      arguments: { parameter: sweepParam, values: sweepValues }
    }));
    return;
  }

  // Turn 5: Synthesize observations, causal inference, and evidence
  const divergenceEntry = history.find(h => h.tool === 'find_first_divergence');
  const telemetryEntry = history.find(h => h.tool === 'get_telemetry_range');
  const sweepEntry = history.find(h => h.tool === 'run_parameter_sweep');

  const divTime = divergenceEntry?.result?.time;
  const metrics = failure.metrics || {};
  const categories = failure.categories || [];

  const observations = [
    `Failure categories triggered: [${categories.join(', ')}].`,
    divTime !== undefined
      ? `First state divergence detected at t = ${divTime.toFixed(2)}s.`
      : `No path/localization threshold crossing was observed.`,
    `Endpoint error reached ${(metrics.endpoint_error ?? 0).toFixed(3)}m against pass threshold.`,
    `Maximum tracking error: ${(metrics.max_tracking_error ?? 0).toFixed(3)}m, localization error: ${(metrics.max_localization_error ?? 0).toFixed(3)}m.`
  ];

  if (telemetryEntry?.result?.length) {
    const samples = telemetryEntry.result;
    const maxSat = Math.max(...samples.map(s => Math.max(...(s.applied || []).map(Math.abs))));
    const maxSlip = Math.max(...samples.map(s => {
      const vTrans = Math.hypot(s.velocity?.[0] || 0, s.velocity?.[1] || 0);
      const wAvg = ((s.wheel_speed || []).reduce((a, b) => a + Math.abs(b), 0)) / 4;
      return Math.abs(wAvg - vTrans);
    }));
    observations.push(`Telemetry analysis around divergence shows peak wheel command ${maxSat.toFixed(1)}V and dynamic slip index ${maxSlip.toFixed(3)}.`);
  }

  const hypotheses = [];
  if (categories.includes('LOCALIZATION_DIVERGENCE') || categories.includes('ENDPOINT_FAILURE')) {
    hypotheses.push(
      'Reduced traction may contribute to tracking or localization error; the recorded sweep does not prove slip or a physical root cause.'
    );
  }
  if (categories.includes('MOTOR_SATURATION') || categories.includes('TIMEOUT')) {
    hypotheses.push(
      'Voltage limits may contribute to command saturation or delayed settling; confirm against recorded applied commands and paired results.'
    );
  }
  if (categories.includes('HEADING_INSTABILITY') || categories.includes('OSCILLATION')) {
    hypotheses.push(
      'Feedback delay is a candidate sensitivity to test; these observations alone do not establish instability or its cause.'
    );
  }
  if (!hypotheses.length) {
    hypotheses.push('No specific causal mechanism has been established by this deterministic tool sequence.');
  }

  const limitations = [
    'This bridge uses fixed rules and templates; it performs no neural, LLM or calibrated causal inference.',
    'Reduced-order planar plant models wheel dynamics via backward-Euler; physical tire-field compliance may exhibit non-linear stick-slip not captured here.',
    'Encoder dead-reckoning is unobservable under uniform slip without external absolute positioning (GPS/MCL/VEX Field Code).',
    'Analysis conducted in deterministic software simulation; physical robot validation on VEX V5 hardware is required.'
  ];

  const evidence = history.map((_, index) => index);

  console.log(JSON.stringify({
    type: 'final',
    observations,
    hypotheses,
    evidence,
    limitations
  }));
}

main().catch(err => {
  console.error(`Local Agent error: ${err.message}`);
  process.exit(1);
});
