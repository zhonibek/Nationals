'use strict';

const {createHash} = require('node:crypto');
const {normalizeTask} = require('../../simulator/motion');
const specification = require('./locateanything.json');

const MAX_FRAME_AGE_MS = 5000;
const MAX_ANSWER_BYTES = 32768;
const MAX_DETECTIONS = 32;
const FIELD_FRAME = 'nationals-field-inches-v1';
const FRAME_KEYS = ['id', 'cameraId', 'cameraPoseId', 'coordinateSpaceId', 'imageSha256', 'width', 'height', 'capturedAt'];

function keys(value, allowed, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw Error(`${label} must be an object`);
  if (Object.keys(value).some(key => !allowed.includes(key))) throw Error(`Unsupported ${label} field`);
}

function bounded(value, minimum, maximum, label) {
  if (!Number.isFinite(value) || value < minimum || value > maximum) throw Error(`${label} must be in [${minimum}, ${maximum}]`);
  return value;
}

function identifier(value, label) {
  if (typeof value !== 'string' || !/^[a-zA-Z0-9][a-zA-Z0-9._:-]{0,127}$/.test(value)) throw Error(`Invalid ${label}`);
  return value;
}

function timestamp(value, label) {
  if (typeof value !== 'string' || !Number.isFinite(Date.parse(value)) || new Date(value).toISOString() !== value) throw Error(`${label} must be a canonical UTC timestamp`);
  return Date.parse(value);
}

function sha256(value) {
  return createHash('sha256').update(value).digest('hex');
}

function normalizeFrame(frame) {
  keys(frame, FRAME_KEYS, 'frame');
  const normalized = {};
  for (const key of ['id', 'cameraId', 'cameraPoseId', 'coordinateSpaceId']) normalized[key] = identifier(frame[key], `frame.${key}`);
  if (typeof frame.imageSha256 !== 'string' || !/^[a-f0-9]{64}$/.test(frame.imageSha256)) throw Error('frame.imageSha256 must identify the source image');
  normalized.imageSha256 = frame.imageSha256;
  for (const key of ['width', 'height']) {
    bounded(frame[key], 1, 2560, `frame.${key}`);
    if (!Number.isInteger(frame[key])) throw Error(`frame.${key} must be an integer`);
    normalized[key] = frame[key];
  }
  timestamp(frame.capturedAt, 'frame.capturedAt');
  normalized.capturedAt = frame.capturedAt;
  return normalized;
}

function parseGrounding(input) {
  keys(input, ['schemaVersion', 'frame', 'prompt', 'answer', 'truncated', 'source'], 'grounding');
  if (input.schemaVersion !== 1) throw Error('Unsupported grounding schema');
  const frame = normalizeFrame(input.frame);
  if (typeof input.prompt !== 'string' || !input.prompt.trim() || input.prompt.length > 2000) throw Error('A bounded grounding prompt is required');
  if (input.truncated !== false) throw Error('Truncated or unknown-completion output cannot prepare a proposal');
  if (typeof input.answer !== 'string' || Buffer.byteLength(input.answer) > MAX_ANSWER_BYTES) throw Error('Grounding answer exceeds its text budget');
  keys(input.source, ['kind', 'model', 'revision'], 'source');
  if (input.source.model !== specification.model) throw Error('Only the declared LocateAnything model is supported');
  if (!['fixture', 'imported-model-output'].includes(input.source.kind)) throw Error('Unsupported source kind; live inference is not implemented');
  if (input.source.kind === 'fixture' ? input.source.revision !== null : typeof input.source.revision !== 'string' || !/^[a-f0-9]{40}$/.test(input.source.revision)) throw Error('Imported model output requires an immutable model revision; fixtures use null');
  const source = {...input.source};
  const detections = [];
  const expression = /<box>((?:<\d+>){2}|(?:<\d+>){4})<\/box>/g;
  const remainder = input.answer.replace(expression, (block, coordinates) => {
    if (detections.length >= MAX_DETECTIONS) throw Error('Grounding detection budget exhausted');
    const normalized = [...coordinates.matchAll(/<(\d+)>/g)].map(match => bounded(Number(match[1]), 0, 1000, 'normalized coordinate'));
    const pixels = normalized.map((value, index) => value / 1000 * (index % 2 === 0 ? frame.width : frame.height));
    if (pixels.length === 4 && (pixels[2] <= pixels[0] || pixels[3] <= pixels[1])) throw Error('Grounding box must have positive width and height');
    detections.push({index: detections.length, type: pixels.length === 2 ? 'point' : 'box', normalized, pixels, confidence: null});
    return '';
  });
  if (/<\/?box/i.test(remainder)) throw Error('Malformed or unsupported native box/point tokens');
  return {schemaVersion: 1, model: specification.model, source, frame, prompt: input.prompt, answer: input.answer,
    answerSha256: sha256(input.answer), status: detections.length ? 'grounded' : 'no_geometry', detections,
    inferencePerformedByRoboProof: false, sourceVerified: false};
}

function project(homography, pixel) {
  const [firstX, firstY, firstOffset, secondX, secondY, secondOffset, divisorX, divisorY, divisorOffset] = homography;
  const divisor = divisorX * pixel.x + divisorY * pixel.y + divisorOffset;
  if (!Number.isFinite(divisor) || Math.abs(divisor) < 1e-9) throw Error('Calibration has a projective singularity');
  return {xIn: (firstX * pixel.x + firstY * pixel.y + firstOffset) / divisor,
    yIn: (secondX * pixel.x + secondY * pixel.y + secondOffset) / divisor};
}

function validateCalibration(calibration, grounding, now) {
  keys(calibration, ['schemaVersion', 'id', 'kind', 'cameraId', 'cameraPoseId', 'coordinateSpaceId', 'width', 'height',
    'fieldFrame', 'homography', 'pixelCoverage', 'validFrom', 'expiresAt', 'reviewedBy', 'reviewedAt', 'validation'], 'calibration');
  if (calibration.schemaVersion !== 1 || calibration.kind !== 'rectified-fixed-plane' || calibration.fieldFrame !== FIELD_FRAME) throw Error('A rectified fixed-camera field-plane calibration is required');
  identifier(calibration.id, 'calibration.id');
  identifier(calibration.reviewedBy, 'calibration.reviewedBy');
  for (const key of ['cameraId', 'cameraPoseId', 'coordinateSpaceId', 'width', 'height']) {
    if (calibration[key] !== grounding.frame[key]) throw Error(`Calibration/frame ${key} mismatch`);
  }
  const validFrom = timestamp(calibration.validFrom, 'calibration.validFrom');
  const expiresAt = timestamp(calibration.expiresAt, 'calibration.expiresAt');
  const reviewedAt = timestamp(calibration.reviewedAt, 'calibration.reviewedAt');
  const capturedAt = Date.parse(grounding.frame.capturedAt);
  if (validFrom > capturedAt || now >= expiresAt || reviewedAt < validFrom || reviewedAt > capturedAt) throw Error('Calibration is expired, unreviewed for this frame, or not yet valid');
  keys(calibration.validation, ['kind', 'sampleCount', 'rmsErrorIn', 'maxErrorIn'], 'calibration.validation');
  const expectedKind = grounding.source.kind === 'fixture' ? 'fixture' : 'held-out-physical-measurements';
  if (calibration.validation.kind !== expectedKind) throw Error('Fixture calibration cannot validate imported model output');
  if (!Number.isInteger(calibration.validation.sampleCount) || calibration.validation.sampleCount < 4) throw Error('At least four held-out calibration measurements are required');
  bounded(calibration.validation.sampleCount, 4, 1000000, 'calibration sample count');
  bounded(calibration.validation.rmsErrorIn, 0, 0.5, 'calibration RMS error in inches');
  bounded(calibration.validation.maxErrorIn, calibration.validation.rmsErrorIn, 1, 'calibration maximum error in inches');
  if (!Array.isArray(calibration.homography) || calibration.homography.length !== 9) throw Error('Calibration homography must contain nine coefficients');
  const matrix = calibration.homography.map(value => bounded(value, -1e6, 1e6, 'homography coefficient'));
  const scale = Math.max(...matrix.map(Math.abs));
  if (scale === 0) throw Error('Singular calibration homography');
  const normalized = matrix.map(value => value / scale);
  const determinant = normalized[0] * (normalized[4] * normalized[8] - normalized[5] * normalized[7])
    - normalized[1] * (normalized[3] * normalized[8] - normalized[5] * normalized[6])
    + normalized[2] * (normalized[3] * normalized[7] - normalized[4] * normalized[6]);
  if (Math.abs(determinant) < 1e-12) throw Error('Singular or ill-conditioned calibration homography');
  const homography = matrix.map(value => value / scale);
  const coverage = calibration.pixelCoverage;
  if (!Array.isArray(coverage) || coverage.length !== 4) throw Error('Calibration pixelCoverage must be a rectangle');
  coverage.forEach((value, index) => bounded(value, 0, index % 2 === 0 ? calibration.width : calibration.height, 'calibration coverage'));
  if (coverage[2] <= coverage[0] || coverage[3] <= coverage[1]) throw Error('Calibration coverage must have positive area');
  const corners = [{x: coverage[0], y: coverage[1]}, {x: coverage[2], y: coverage[1]},
    {x: coverage[2], y: coverage[3]}, {x: coverage[0], y: coverage[3]}];
  let sign = 0;
  for (const corner of corners) {
    const divisor = homography[6] * corner.x + homography[7] * corner.y + homography[8];
    if (sign && Math.sign(divisor) !== sign) throw Error('Calibration crosses a projective singularity inside coverage');
    sign = Math.sign(divisor);
    const field = project(homography, corner);
    bounded(field.xIn, -60, 60, 'calibrated field xIn');
    bounded(field.yIn, -60, 60, 'calibrated field yIn');
  }
  return {homography, coverage};
}

function prepareProposal(input, {now = Date.now()} = {}) {
  keys(input, ['grounding', 'calibration', 'selection', 'start', 'goalHeadingDeg', 'deadlineSeconds'], 'perception proposal');
  if (!Number.isFinite(now)) throw Error('Invalid validation clock');
  const grounding = parseGrounding(input.grounding);
  const capturedAt = Date.parse(grounding.frame.capturedAt);
  if (now < capturedAt || now - capturedAt > MAX_FRAME_AGE_MS) throw Error('Frame is stale or from the future; acquire and review a new frame');
  if (!grounding.detections.length) throw Error('No grounded geometry; object absence is not proven');
  const selection = input.selection;
  keys(selection, ['detectionIndex', 'frameId', 'imageSha256', 'plane', 'pixelPoint', 'reviewedBy', 'reviewedAt'], 'selection');
  if (!Number.isInteger(selection.detectionIndex) || !grounding.detections[selection.detectionIndex]) throw Error('Explicit detection selection is required');
  if (selection.frameId !== grounding.frame.id || selection.imageSha256 !== grounding.frame.imageSha256) throw Error('Selection belongs to another frame/image');
  if (selection.plane !== 'field-floor') throw Error('Only a reviewed field-floor point can become a goal; object centers are not ground positions');
  identifier(selection.reviewedBy, 'selection.reviewedBy');
  const reviewedAt = timestamp(selection.reviewedAt, 'selection.reviewedAt');
  if (reviewedAt < capturedAt || reviewedAt > now) throw Error('Selection review must follow capture and precede validation');
  keys(selection.pixelPoint, ['x', 'y'], 'selection.pixelPoint');
  const pixel = {x: bounded(selection.pixelPoint.x, 0, grounding.frame.width, 'pixel x'),
    y: bounded(selection.pixelPoint.y, 0, grounding.frame.height, 'pixel y')};
  const detection = grounding.detections[selection.detectionIndex];
  const coordinates = detection.pixels;
  if (detection.type === 'point' ? Math.hypot(pixel.x - coordinates[0], pixel.y - coordinates[1]) > 1e-9
    : pixel.x < coordinates[0] || pixel.y < coordinates[1] || pixel.x > coordinates[2] || pixel.y > coordinates[3]) throw Error('Reviewed floor point must lie in the selected geometry');
  const {homography, coverage} = validateCalibration(input.calibration, grounding, now);
  if (pixel.x < coverage[0] || pixel.y < coverage[1] || pixel.x > coverage[2] || pixel.y > coverage[3]) throw Error('Pixel is outside validated calibration coverage; extrapolation is disabled');
  const field = project(homography, pixel);
  const task = normalizeTask({start: input.start, goal: {...field, headingDeg: input.goalHeadingDeg}, deadlineSeconds: input.deadlineSeconds});
  return {schemaVersion: 1, status: 'proposal_only', task, grounding, selection: structuredClone(selection),
    calibration: structuredClone(input.calibration), preparedAt: new Date(now).toISOString(),
    simulationOnly: true, requiresSeparateSimulatorApproval: true, inferencePerformedByRoboProof: false,
    execution: 'not run; offline preparation does not authorize movement', motionPolicyTraining: 'not implemented',
    provenanceVerification: 'caller-supplied image identity, model output and calibration review; not independently verified'};
}

function readiness() {
  return {...structuredClone(specification), inferenceAvailable: false, liveInferenceVerified: false,
    cameraConnected: false, calibrationInstalled: false, movementPolicyTrained: false,
    capabilities: ['Native box/point parser', 'Reviewed fixed-plane pixel-to-field proposal gate', 'Shared Simulator task schema'],
    execution: 'disabled; no perception execution endpoint'};
}

module.exports = {parseGrounding, prepareProposal, readiness, MAX_FRAME_AGE_MS, MAX_ANSWER_BYTES, MAX_DETECTIONS, FIELD_FRAME};
