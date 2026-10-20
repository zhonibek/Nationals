'use strict';

function parseCompoundTacticalPrompt(prompt) {
  const p = prompt.trim();
  
  let startX = 0, startY = 0, startTheta = 0;
  let remaining = p;
  
  const startM = remaining.match(/(?:start(?:\s+at)?|старт(?:\s+в(?:\s+точке)?)?|from|начало(?:\s+на(?:\s+точке)?)?|pose)\s*[:=]?\s*\(?\s*([+-]?\d+(?:\.\d+)?)\s*[:,\s]\s*([+-]?\d+(?:\.\d+)?)(?:\s*[:,\s]\s*([+-]?\d+(?:\.\d+)?))?\s*\)?/i);
  if (startM) {
    startX = parseFloat(startM[1]);
    startY = parseFloat(startM[2]);
    if (startM[3] !== undefined) startTheta = parseFloat(startM[3]);
    remaining = remaining.replace(startM[0], ' ');
  }
  
  // Split into individual instruction clauses
  const rawClauses = remaining.split(/(?:;|\n|\r|\bthen\b|\bпосле чего\b|\bзатем\b|\bпотом\b|\bдалее\b|,(?![^\(]*\)))/i)
    .map(c => c.trim()).filter(c => c.length > 0);
  
  const state = { x: startX, y: startY, theta: startTheta };
  const allActions = [];
  const steps = [
    `Старт автономного плана в точке (${state.x.toFixed(1)}″, ${state.y.toFixed(1)}″, курс ${state.theta}°).`
  ];
  const waypoints = [
    { id: 1, x: state.x, y: state.y, theta: state.theta, type: 'start', speed: 1.0 }
  ];

  for (let i = 0; i < rawClauses.length; i++) {
    const clause = rawClauses[i];
    const c = clause.toLowerCase().trim();
    const rad = state.theta * Math.PI / 180;

    // 1. Compound: 'go forward by turning right/left N degrees' or 'turn right N and go forward D'
    const turnDriveM = c.match(/(?:go\s+forward|drive|ехать|вперед)[^\d\-]+(?:turning|повернув|поворот)\s+(?:to\s+the\s+)?(right|left|направо|налево)\s*(?:by\s*)?([+-]?\d+(?:\.\d+)?)\s*(?:deg|градус|°)?/i)
                    || c.match(/(?:turn|повернуть)\s+(right|left|направо|налево)\s*([+-]?\d+(?:\.\d+)?)\s*(?:deg|градус|°)?\s+(?:and|then|и|затем)\s+(?:go\s+forward|drive|ехать|вперед)\s*(?:by\s*)?(\d+(?:\.\d+)?)?/i);
    if (turnDriveM) {
      const dir = (turnDriveM[1].includes('right') || turnDriveM[1].includes('направо')) ? 1 : -1;
      const deg = parseFloat(turnDriveM[2]);
      const dist = parseFloat(turnDriveM[3] || '20');
      let newTheta = (state.theta + dir * deg) % 360;
      if (newTheta < 0) newTheta += 360;
      state.theta = Math.round(newTheta);
      const newRad = state.theta * Math.PI / 180;
      const nx = +(state.x + dist * Math.sin(newRad)).toFixed(1);
      const ny = +(state.y + dist * Math.cos(newRad)).toFixed(1);
      
      allActions.push({ type: 'turn', targetHeading: state.theta, desc: `Turn ${dir > 0 ? 'right' : 'left'} ${deg}° to ${state.theta}°` });
      allActions.push({ type: 'drive', targetInches: dist, heading: state.theta, targetX: nx, targetY: ny, desc: `Drive forward ${dist}" along ${state.theta}°` });
      steps.push(`Поворот ${dir > 0 ? 'вправо' : 'влево'} на ${deg}° (курс ${state.theta}°) и ускорение вперед на ${dist}″ в (${nx}″, ${ny}″).`);
      state.x = nx; state.y = ny;
      waypoints.push({ id: waypoints.length + 1, x: state.x, y: state.y, theta: state.theta, type: 'spline', speed: 1.2 });
      continue;
    }

    // 2. Strafe left / right N inches
    const strafeM = c.match(/(?:strafe|стрейф)\s*(?:to\s+the\s+)?(right|left|вправо|влево)?\s*(?:на\s*)?(\d+(?:\.\d+)?)\s*(?:in|inch|инч|дюйм|\")?/i);
    if (strafeM) {
      const isLeft = strafeM[1] && (strafeM[1].includes('left') || strafeM[1].includes('влево'));
      const dist = parseFloat(strafeM[2]);
      const sign = isLeft ? -1 : 1;
      const nx = +(state.x + sign * dist * Math.cos(rad)).toFixed(1);
      const ny = +(state.y - sign * dist * Math.sin(rad)).toFixed(1);
      allActions.push({ type: 'strafe', targetInches: sign * dist, heading: state.theta, targetX: nx, targetY: ny, desc: `Strafe ${isLeft ? 'left' : 'right'} ${dist}"` });
      steps.push(`Холономический стрейф ${isLeft ? 'влево' : 'вправо'} на ${dist}″ в точку (${nx}″, ${ny}″).`);
      state.x = nx; state.y = ny;
      waypoints.push({ id: waypoints.length + 1, x: state.x, y: state.y, theta: state.theta, type: 'strafe', speed: 1.0 });
      continue;
    }

    // 3. 'turn to point (X, Y)'
    const turnPointM = c.match(/(?:turn\s+to\s+point|повернут(?:ься)?\s+к\s+точке|look\s+at)\s*\(?\s*([+-]?\d+(?:\.\d+)?)\s*[:,\s]\s*([+-]?\d+(?:\.\d+)?)\s*\)?/i);
    if (turnPointM) {
      const tx = parseFloat(turnPointM[1]);
      const ty = parseFloat(turnPointM[2]);
      const dx = tx - state.x;
      const dy = ty - state.y;
      let bearing = Math.round(Math.atan2(dx, dy) * 180 / Math.PI);
      if (bearing < 0) bearing += 360;
      state.theta = bearing;
      allActions.push({ type: 'turn', targetHeading: bearing, desc: `Turn to point (${tx}, ${ty}) at ${bearing}°` });
      steps.push(`Прицеливание на точку (${tx}″, ${ty}″) — доворот шасси на курс ${bearing}°.`);
      waypoints.push({ id: waypoints.length + 1, x: state.x, y: state.y, theta: state.theta, type: 'turn', speed: 1.0 });
      continue;
    }

    // 4. 'go to (point) (X, Y)'
    const goPointM = c.match(/(?:go\s+to(?:\s+point)?|move\s+to|drive\s+to|ехать\s+в(?:\s+точку)?|в\s+точку|к\s+точке)\s*\(?\s*([+-]?\d+(?:\.\d+)?)\s*[:,\s]\s*([+-]?\d+(?:\.\d+)?)\s*\)?/i);
    if (goPointM) {
      const tx = parseFloat(goPointM[1]);
      const ty = parseFloat(goPointM[2]);
      const dx = tx - state.x;
      const dy = ty - state.y;
      let headingToPoint = Math.round(Math.atan2(dx, dy) * 180 / Math.PI);
      if (headingToPoint < 0) headingToPoint += 360;
      state.x = tx;
      state.y = ty;
      state.theta = headingToPoint;
      allActions.push({ type: 'drivePoint', targetX: tx, targetY: ty, targetHeading: headingToPoint, desc: `Move to point (${tx}, ${ty})` });
      steps.push(`Перемещение в целевую точку (${tx}″, ${ty}″) под углом ${headingToPoint}°.`);
      waypoints.push({ id: waypoints.length + 1, x: state.x, y: state.y, theta: state.theta, type: 'drive', speed: 1.3 });
      continue;
    }

    // 4. 'turn right/left N degrees' or 'turn to N degrees'
    const turnDegM = c.match(/(?:turn\s+(?:to\s+the\s+)?|повернуть\s+|разворот\s+на\s*)(right|left|направо|налево)\s*(?:by\s*)?([+-]?\d+(?:\.\d+)?)\s*(?:deg|градус|°)?/i)
                  || c.match(/(?:turn\s+to|повернуть\s+на|курс)\s*([+-]?\d+(?:\.\d+)?)\s*(?:deg|градус|°)?/i);
    if (turnDegM) {
      const isDirected = turnDegM[2] !== undefined;
      const dirStr = isDirected ? turnDegM[1] : '';
      const isRelRight = dirStr.includes('right') || dirStr.includes('направо');
      const isRelLeft = dirStr.includes('left') || dirStr.includes('налево');
      const deg = parseFloat(isDirected ? turnDegM[2] : turnDegM[1]);
      let newTheta = state.theta;
      if (isRelRight) newTheta = (state.theta + deg) % 360;
      else if (isRelLeft) newTheta = (state.theta - deg) % 360;
      else newTheta = deg % 360;
      if (newTheta < 0) newTheta += 360;
      state.theta = Math.round(newTheta);
      allActions.push({ type: 'turn', targetHeading: state.theta, desc: `Turn to ${state.theta}°` });
      steps.push(`Поворот ${isRelRight ? 'направо на ' + deg + '°' : isRelLeft ? 'налево на ' + deg + '°' : 'на курс ' + deg + '°'} (новый курс: ${state.theta}°).`);
      waypoints.push({ id: waypoints.length + 1, x: state.x, y: state.y, theta: state.theta, type: 'turn', speed: 1.0 });
      continue;
    }

    // 5. 'drive forward / backward N inches'
    const driveDistM = c.match(/(?:drive|go|move|проехать|ехать)?\s*(forward|backward|вперед|вперёд|назад)?\s*(\d+(?:\.\d+)?)\s*(?:in|inch|инч|дюйм|\")/i)
                    || c.match(/(?:drive|go|move|проехать|ехать)\s*(forward|backward|вперед|вперёд|назад)?\s*(\d+(?:\.\d+)?)/i);
    if (driveDistM) {
      const isBack = (driveDistM[1] && (driveDistM[1].includes('back') || driveDistM[1].includes('назад')));
      const dist = parseFloat(driveDistM[2]);
      const sign = isBack ? -1 : 1;
      const nx = +(state.x + sign * dist * Math.sin(rad)).toFixed(1);
      const ny = +(state.y + sign * dist * Math.cos(rad)).toFixed(1);
      allActions.push({ type: 'drive', targetInches: sign * dist, heading: state.theta, targetX: nx, targetY: ny, desc: `Drive ${isBack ? 'backward' : 'forward'} ${dist}"` });
      steps.push(`Прямолинейный разгон ${isBack ? 'назад' : 'вперед'} на ${dist}″ в точку (${nx}″, ${ny}″).`);
      state.x = nx; state.y = ny;
      waypoints.push({ id: waypoints.length + 1, x: state.x, y: state.y, theta: state.theta, type: 'drive', speed: 1.2 });
      continue;
    }

    // 6. Lift / Object Manipulation
    if (c.includes('lift') || c.includes('подъем') || c.includes('поднять') || c.includes('вилка') || c.includes('cup') || c.includes('pin') || c.includes('goal')) {
      const heightM = c.match(/(\d+(?:\.\d+)?)/);
      const h = heightM ? parseFloat(heightM[1]) : 12.0;
      allActions.push({ type: 'lift', height: h, desc: `Lift to ${h}"` });
      steps.push(`Подъем манипулятора на высоту ${h}″ для взаимодействия с игровым элементом.`);
      continue;
    }
  }

  return {
    startPose: { x: startX, y: startY, theta: startTheta },
    finalPose: { ...state },
    actions: allActions,
    steps,
    waypoints
  };
}

const testCases = [
  'start at 0:-48 drive forward 20 inches,  go to point (5, 5), turn right 48 degrees, turn to point(24,24), go forward by turning right 90 degrees',
  'начало на точке 30:30 поехать вперед на 20инчей',
  'старт -20, 20 стрейф вправо на 24 инча, затем поднять вилку на 14',
  'from (0,0) go to (0, 30), go to (30, 30), go to (30, 0), go to (0, 0)',
  'старт 10:10 drive forward 15, turn left 90, lift 8'
];

for (const tc of testCases) {
  console.log('====================================================');
  console.log('INPUT:', tc);
  const plan = parseCompoundTacticalPrompt(tc);
  console.log('Start:', plan.startPose, 'Final:', plan.finalPose);
  console.log('Actions count:', plan.actions.length, 'Waypoints:', plan.waypoints.length);
  plan.steps.forEach((s, idx) => console.log(`  [${idx}] ${s}`));
}
