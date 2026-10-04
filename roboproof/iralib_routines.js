'use strict';

/**
 * IRAlib Autonomous Routine Adapter for RoboProof
 * Connects the real VEX V5 competition routines from src/main.cpp
 * and simulator/simulator.js directly into RoboProof's verification engine.
 */

const INCH_TO_METER = 0.0254;
const DEG_TO_RAD = Math.PI / 180;

const IRALIB_ROUTINES = {
  cascadeDemo: {
    id: 'cascade-demo',
    name: 'IRAlib Competition Autonomous (Cascade Demo)',
    description: 'Autonomous routine 0 from main.cpp: Spline to (24", 24", 90°) then autoPose return to (0, 0, 0°)',
    segments: [
      {
        type: 'spline',
        start: [0, 0, 0],
        goal: [24 * INCH_TO_METER, 24 * INCH_TO_METER, 90 * DEG_TO_RAD],
        maxSpeed: 0.45,
        maxAccel: 0.8
      },
      {
        type: 'pose',
        start: [24 * INCH_TO_METER, 24 * INCH_TO_METER, 90 * DEG_TO_RAD],
        goal: [0, 0, 0],
        maxSpeed: 0.45,
        maxAccel: 0.8
      }
    ],
    task: {
      start: [0, 0, 0],
      goal: [24 * INCH_TO_METER, 24 * INCH_TO_METER, 90 * DEG_TO_RAD],
      dt: 0.01,
      duration: 10,
      localization: 'encoders',
      thresholds: {
        endpoint: 0.06,
        path: 0.12,
        heading: 0.12,
        localization: 0.08,
        saturation_fraction: 0.5,
        oscillations: 8,
        settle_speed: 0.04,
        settle_omega: 0.1,
        settle_seconds: 0.2
      }
    }
  },

  holonomicSkills: {
    id: 'holonomic-skills',
    name: 'IRAlib Holonomic Skills (4-Point Box Pattern)',
    description: 'Autonomous routine 1 from main.cpp: Diamond Skills sequence (0,24,0) -> (24,24,0) -> (0,24,0) -> (0,0,0)',
    waypoints: [
      [0, 0, 0],
      [0, 24 * INCH_TO_METER, 0],
      [24 * INCH_TO_METER, 24 * INCH_TO_METER, 0],
      [0, 24 * INCH_TO_METER, 0],
      [0, 0, 0]
    ],
    task: {
      start: [0, 0, 0],
      goal: [0, 0, 0],
      dt: 0.01,
      duration: 15,
      localization: 'encoders',
      thresholds: {
        endpoint: 0.06,
        path: 0.15,
        heading: 0.15,
        localization: 0.10,
        saturation_fraction: 0.6,
        oscillations: 10,
        settle_speed: 0.05,
        settle_omega: 0.15,
        settle_seconds: 0.2
      }
    }
  },

  pedroBezier: {
    id: 'pedro-bezier',
    name: 'IRAlib Pedro Bézier Diagnostic',
    description: 'Autonomous routine 2 from main.cpp / runDiagnostic(7): Cubic Bézier curve [0,0]->[0,16]->[24,8]->[24,24] with 0°->90° heading',
    controlPoints: [
      [0, 0],
      [0, 16 * INCH_TO_METER],
      [24 * INCH_TO_METER, 8 * INCH_TO_METER],
      [24 * INCH_TO_METER, 24 * INCH_TO_METER]
    ],
    startHeading: 0,
    endHeading: 90 * DEG_TO_RAD,
    task: {
      start: [0, 0, 0],
      goal: [24 * INCH_TO_METER, 24 * INCH_TO_METER, 90 * DEG_TO_RAD],
      dt: 0.01,
      duration: 12,
      localization: 'encoders',
      thresholds: {
        endpoint: 0.06,
        path: 0.12,
        heading: 0.12,
        localization: 0.08,
        saturation_fraction: 0.5,
        oscillations: 8,
        settle_speed: 0.04,
        settle_omega: 0.1,
        settle_seconds: 0.2
      }
    }
  }
};

/**
 * Creates a scenario pre-configured with an official IRAlib competition autonomous routine.
 */
function makeIralibScenario(routineKey = 'cascadeDemo', seed = 42, overrides = {}) {
  const routine = IRALIB_ROUTINES[routineKey];
  if (!routine) throw Error(`Unknown IRAlib routine: ${routineKey}. Supported: ${Object.keys(IRALIB_ROUTINES).join(', ')}`);
  
  const {makeScenario} = require('./core');
  const base = makeScenario(seed, {
    task: routine.task,
    ...overrides
  });
  base.iralib_routine = {
    key: routineKey,
    name: routine.name,
    description: routine.description
  };
  return base;
}

module.exports = {
  IRALIB_ROUTINES,
  makeIralibScenario,
  INCH_TO_METER,
  DEG_TO_RAD
};
