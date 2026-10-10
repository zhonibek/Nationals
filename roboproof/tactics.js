'use strict';

const fs = require('node:fs');
const path = require('node:path');
const OverrideGame = require('../simulator/override');
const Snapshot = require('../simulator/tactics-snapshot');
const {sha256} = require('./agent-skills');

function rules() {
  const filenames = ['simulator/override.js', 'simulator/override-geometry.js', 'simulator/override-dynamics.js'];
  const constants = OverrideGame.constants;
  return {game: 'V5RC Override', season: '2026-2027', manualVersion: '2.0',
    sources: {manual: 'https://www.vexrobotics.com/override-manual', officialPdf: 'https://link.vex.com/docs/26-27/v5rc/game-manual',
      checkedAt: '2026-10-06', scope: 'Reviewed simulator subset; public complement checked, normative PDF not fetched in this update'},
    engineHashes: Object.fromEntries(filenames.map(filename => [filename, sha256(fs.readFileSync(path.join(__dirname, '..', filename), 'utf8').replace(/\r\n/g, '\n'))])),
    timing: {autonomousSeconds: constants.AUTO_SECONDS, matchSeconds: constants.MATCH_SECONDS, endgameSeconds: constants.ENDGAME_SECONDS},
    points: {...constants.POINTS, cupIndependentPoints: 0, tiedAutonomousBonus: constants.POINTS.autonomousBonus / 2},
    goalColumns: ['id', 'xIn', 'yIn', 'alliance', 'heightIn'],
    goals: OverrideGame.layouts.goals.map(goal => [goal.id, goal.x - constants.FIELD_HALF,
      goal.y - constants.FIELD_HALF, goal.alliance, goal.height]),
    constraints: [
      {id: 'SC2/SC3', fact: 'Placement, nesting and visible halves determine Pin scoring; a Cup alone has no point value.'},
      {id: 'SC4/SC5', fact: 'Yellow Pin value depends on ownership, including eligible Toggle state and placement region.'},
      {id: 'SC6/SC7', fact: 'Midfield robot points are end-of-match criteria, not autonomous score.'},
      {id: 'SG6', fact: 'Possession is limited to one Pin and one Cup per robot.'},
      {id: 'SG7/SG8/SG9/SG10', fact: 'Autonomous-line, opponent-goal and neutral-stack protections need checks before execution.'},
      {id: 'SG12', fact: 'Midfield-goal placement is unavailable during the final ten seconds of a match.'}
    ],
    roles: {tactics: 'Nemotron read-only adviser; no dispatch', motion: 'Original Simulator / iraLIB; experimental PPO not verified',
      manipulation: 'Existing scripted lift/grasp/place mechanics; not a trained sub-AI', perception: 'LocateAnything future preparation; no live camera',
      referee: 'Existing rules engine plus human adjudication; incomplete automatic legality checks'},
    limitations: 'Rules plus observations do not prove a winning or executable tactic. Missing rules, physics calibration and opponent behavior remain gates.'};
}

function gameEvidence(snapshot, now = Date.now()) {
  const state = Snapshot.validate(snapshot, now);
  return {snapshotSha256: sha256(JSON.stringify(state)), state,
    provenance: 'Explicit user-supplied original-Simulator snapshot; structure validated, not independently authenticated or replayed',
    interpretation: 'Reported score and state at capture, not future points, complete field visibility or current state after inference',
    scope: 'Head-to-head/practice subset; at most six nearby Pins and four Cups', executableTask: false};
}

module.exports = {rules, gameEvidence};
