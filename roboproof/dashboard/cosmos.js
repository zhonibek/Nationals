'use strict';

(() => {
  const element = id => document.getElementById(`cosmos-${id}`);
  const refreshButton = element('refresh');
  const status = element('status');
  const results = element('results');
  const policy = element('hardware-policy');
  if (!refreshButton || !status || !results || !policy) return;
  const modelId = 'nvidia/Cosmos-Reason2-2B';
  const revision = '9ce19a195e423419c349abfc86fd07178b230561';
  const uuid = value => typeof value === 'string' && /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/.test(value);
  const hash = value => typeof value === 'string' && /^[0-9a-f]{64}$/.test(value);
  const object = value => value !== null && typeof value === 'object' && !Array.isArray(value);
  const text = (value, maximum) => typeof value === 'string' && value.trim().length > 0 && value.length <= maximum;
  const integer = (value, maximum) => Number.isInteger(value) && value >= 1 && value <= maximum;
  const nonnegative = value => Number.isFinite(value) && value >= 0;
  const positive = value => Number.isFinite(value) && value > 0;
  let busy = false;

  function validClaims(record) {
    const negative = new Set(['robotactionsexecuted', 'trainingperformed', 'simulatormeasurementsmodified',
      'hardwareverified', 'hardwareauthenticityverified', 'commentaryverified', 'livemodelverified', 'liveinferenceverified']);
    const affirmative = new Set(['humanapprovalrequired', 'unverifiedmodelcommentary']);
    const pending = [{value: record, depth: 0}];
    let visited = 0;
    while (pending.length) {
      const {value, depth} = pending.pop();
      if (++visited > 20000 || depth > 32) return false;
      if (value === null || typeof value !== 'object') continue;
      for (const [key, child] of Object.entries(value)) {
        const flag = key.toLowerCase().replace(/[^a-z0-9]/g, '');
        if ((negative.has(flag) && child !== false) || (affirmative.has(flag) && child !== true)) return false;
        pending.push({value: child, depth: depth + 1});
      }
    }
    return true;
  }

  function validRecord(record) {
    if (!object(record) || record.schemaVersion !== 1 || !uuid(record.id) || !text(record.createdAt, 40)
      || !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?(?:Z|[+-](?:[01]\d|2[0-3]):[0-5]\d)$/.test(record.createdAt)
      || !Number.isFinite(Date.parse(record.createdAt)) || !validClaims(record) || !['huggingface-zerogpu', 'colab-cuda'].includes(record.provider)
      || !text(record.prompt, 4000) || !object(record.source) || !object(record.result)) return false;
    const calendar = new Date(`${record.createdAt.slice(0, 19)}Z`);
    if (!Number.isFinite(calendar.getTime()) || calendar.toISOString().slice(0, 19) !== record.createdAt.slice(0, 19)) return false;
    const {source, result} = record;
    const {hardware, media} = result;
    return source.kind === 'original-simulator-pose-visualization' && uuid(source.sessionId) && hash(source.sessionSha256)
      && hash(source.inputSha256) && result.status === 'generated' && text(result.commentary, 100000)
      && result.evidence_kind === 'unverified_model_commentary' && result.model_id === modelId
      && result.requested_revision === revision && result.resolved_revision === revision
      && typeof result.device === 'string' && /^cuda(?::(?:0|[1-9]\d*))?$/.test(result.device)
      && ['float16', 'bfloat16'].includes(result.dtype) && object(hardware) && text(hardware.gpu_name, 128)
      && hardware.device === result.device && hardware.dtype === result.dtype && positive(hardware.total_gpu_bytes)
      && nonnegative(hardware.free_gpu_bytes_at_admission) && hardware.free_gpu_bytes_at_admission <= hardware.total_gpu_bytes
      && object(media) && media.input_sha256 === source.inputSha256 && ['image', 'video'].includes(media.kind)
      && integer(media.sampled_frames, 64) && Array.isArray(media.sampled_timestamps_seconds)
      && media.sampled_timestamps_seconds.length <= 64 && media.sampled_timestamps_seconds.every(nonnegative)
      && (media.duration_seconds === null || nonnegative(media.duration_seconds))
      && (media.source_fps === null || positive(media.source_fps))
      && (media.processor_timestamp_basis === null || typeof media.processor_timestamp_basis === 'string')
      && result.robot_actions_executed === false && result.training_performed === false && result.human_approval_required === true
      && result.simulator_measurements_modified === false && typeof result.possibly_truncated === 'boolean'
      && integer(result.input_tokens, 8192) && integer(result.generated_tokens, 4096) && integer(result.output_token_limit, 4096)
      && result.generated_tokens <= result.output_token_limit && nonnegative(result.elapsed_seconds);
  }

  function validResponse(response) {
    return object(response) && response.schemaVersion === 1 && response.modelId === modelId && response.revision === revision
      && response.unverifiedModelCommentary === true && response.robotActionsExecuted === false && response.trainingPerformed === false
      && ['available', 'not_generated', 'invalid'].includes(response.status) && Array.isArray(response.results)
      && response.results.length <= 16 && (response.status === 'available' ? response.results.length > 0 : response.results.length === 0)
      && response.results.every(validRecord) && new Set(response.results.map(record => record.id)).size === response.results.length;
  }

  function append(parent, tag, content) {
    const child = document.createElement(tag);
    child.textContent = content;
    parent.appendChild(child);
    return child;
  }

  function render(record) {
    const card = document.createElement('article');
    card.className = 'inference';
    const {result, source} = record;
    const {hardware, media} = result;
    append(card, 'h3', `Saved Cosmos answer · ${record.createdAt}`);
    append(card, 'p', `Unverified model commentary · result ${record.id}. Not simulator measurements or a robot command.`);
    append(card, 'h4', 'Prompt');
    append(card, 'pre', record.prompt);
    append(card, 'h4', 'Model commentary (unverified)');
    append(card, 'pre', result.commentary);
    append(card, 'p', `Model: ${result.model_id} · requested revision: ${result.requested_revision} · reported resolved revision: ${result.resolved_revision}`);
    append(card, 'p', `Reported provider: ${record.provider} · reported GPU: ${hardware.gpu_name} · ${hardware.device} · ${hardware.dtype}. Reported GPU memory: ${hardware.total_gpu_bytes} bytes total; ${hardware.free_gpu_bytes_at_admission} bytes free at admission.`);
    append(card, 'p', `Sampled media: ${media.kind} · ${media.sampled_frames} frame(s) · timestamps (seconds): ${media.sampled_timestamps_seconds.length ? media.sampled_timestamps_seconds.join(', ') : 'not reported'} · duration (seconds): ${media.duration_seconds ?? 'not reported'} · source FPS: ${media.source_fps ?? 'not reported'} · processor timestamp basis: ${media.processor_timestamp_basis ?? 'not reported'}. This is sampled evidence, not continuous motion verification.`);
    append(card, 'p', `Original Simulator pose visualization, not camera footage. Source session SHA-256: ${source.sessionSha256} · input SHA-256: ${source.inputSha256}`);
    const link = append(card, 'a', 'Open source session in the original Simulator');
    link.href = `/simulator/index.html?motion=${encodeURIComponent(source.sessionId)}`;
    append(card, 'p', `${result.input_tokens} input tokens · ${result.generated_tokens}/${result.output_token_limit} output tokens · ${result.elapsed_seconds} seconds reported.`);
    if (result.possibly_truncated || result.generated_tokens >= result.output_token_limit) {
      append(card, 'p', 'Possibly truncated: this saved answer may have reached its output token limit. It may be incomplete.');
    }
    append(card, 'p', 'Saved provenance is not independent proof of GPU authenticity or commentary correctness. Human approval is required for any proposed action. No robot actions, training or simulator measurement changes are recorded.');
    results.appendChild(card);
  }

  async function refresh() {
    if (busy || !['http:', 'https:'].includes(location.protocol)) return;
    busy = true;
    refreshButton.disabled = true;
    results.replaceChildren();
    status.textContent = 'Reading saved Cosmos answers only. This does not run a model.';
    try {
      const response = await fetch('/api/cosmos/results', {method: 'GET', cache: 'no-store', redirect: 'error', signal: AbortSignal.timeout(5000)});
      if (!response.ok) throw Error();
      const saved = await response.json();
      if (!validResponse(saved) || saved.status === 'invalid') throw Error();
      if (saved.status === 'not_generated') {
        status.textContent = 'No real Cosmos answers are saved. Model availability or loading is not a generated answer.';
        return;
      }
      for (const record of saved.results) render(record);
      status.textContent = `${saved.results.length} saved Cosmos answer(s). Unverified model commentary only; no inference or training was started.`;
    } catch {
      results.replaceChildren();
      status.textContent = 'Saved Cosmos answers are unavailable or invalid. No answers are displayed. Refresh only reads saved files; it does not run a model.';
    } finally {
      busy = false;
      refreshButton.disabled = false;
    }
  }

  policy.textContent = 'Cosmos requires a separately approved NVIDIA CUDA GPU; CPU fallback is not accepted. Nemotron is separately T4-only; no T4 connection is verified by this panel. GPU and provider details are reported by saved files, not independently authenticated. Loading or refreshing this panel never calls a cloud model, executes robot actions or starts inference or training.';
  results.replaceChildren();
  refreshButton.addEventListener('click', refresh);
  if (['http:', 'https:'].includes(location.protocol)) void refresh();
  else {
    refreshButton.disabled = true;
    status.textContent = 'Offline view: no real Cosmos answers have been loaded. Open the local RoboProof server to read saved answers. No model or training is started.';
  }
})();
