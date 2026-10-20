function parseStart(p) {
  // Regex 1: Labeled coordinates (x=10, y=20)
  const labeled = p.match(/(?:start|старт|начало|начинаем|pose|spawn|from|исходная)[^\d\-]*?x\s*[:=]\s*([+-]?\d+(?:\.\d+)?)[^\d\-]*?y\s*[:=]\s*([+-]?\d+(?:\.\d+)?)(?:[^\d\-]*?(?:theta|heading|курс|угол|th)\s*[:=]\s*([+-]?\d+(?:\.\d+)?))?/i);
  if (labeled) {
    return {
      x: Math.max(-60, Math.min(60, parseFloat(labeled[1]))),
      y: Math.max(-60, Math.min(60, parseFloat(labeled[2]))),
      theta: labeled[3] ? parseFloat(labeled[3]) : 0,
      raw: labeled[0]
    };
  }

  // Regex 2: Flexible tuple matching:
  const regex = /(?:start(?:ing)?(?:\s+(?:at|from|in))?(?:\s+(?:the\s+)?(?:point|position|pose|coords|coordinates|точке|точки|позиции|координатах))?|старт(?:\s+(?:в|на|с|из)(?:\s+точк[а-я]*)?)?|начало(?:\s+(?:в|на|с|из)(?:\s+точк[а-я]*)?)?|нач(?:ать|инаем|иная)(?:\s+(?:в|на|с|из)(?:\s+точк[а-я]*)?)?|точка\s+старта|исходна(?:я|я\s+точка|я\s+позиция)|from(?:\s+(?:the\s+)?point)?|pose|spawn(?:\s+at)?)\s*[:=]?\s*\(?\s*(?:x\s*[:=]\s*)?([+-]?\d+(?:\.\d+)?)\s*[:,\s]\s*(?:y\s*[:=]\s*)?([+-]?\d+(?:\.\d+)?)(?:\s*[:,\s]\s*(?:(?:th|heading|курс|угол)\s*[:=]\s*)?([+-]?\d+(?:\.\d+)?))?\s*\)?/i;
  const m = p.match(regex);
  if (m) {
    return {
      x: Math.max(-60, Math.min(60, parseFloat(m[1]))),
      y: Math.max(-60, Math.min(60, parseFloat(m[2]))),
      theta: m[3] ? parseFloat(m[3]) : 0,
      raw: m[0]
    };
  }
  return null;
}

const tests = [
  'start at point (20, 30)',
  'start at point (20, -48)',
  'start at point (20, 30, 45)',
  'start at point (20:30)',
  'start at point 20, 30',
  'start at point 20:30',
  'start at point 0:-48',
  'start point (20, 30)',
  'starting at point (20, 30)',
  'start at (20, 30)',
  'start at 0:-48',
  'start from point (15, 25)',
  'старт в точке (20, 30)',
  'старт с точки (-10, 20)',
  'начало на точке 30:30',
  'начало на точке (30, 30)',
  'начало на точке 30:30 поехать вперед на 20инчей',
  'начало в точке 30, 30',
  'начало в точке (30, 30)',
  'начинаем в точке (10, 15)',
  'начать с точки (-5, -5)',
  'точка старта (5, 5)',
  'исходная позиция (-20, 40)',
  'pose: (10, 20, 90)',
  'start at point x=10, y=20',
  'start at point x: 10, y: 20, heading: 45',
  'start at point (0, -48) drive forward 20 inches',
  'start at point (-30, 30) go to point (0, 0)',
  'start at point (-48, 0)',
  'start at point (24, 24)',
  'start at point (-24, -24)',
  'start at point (0, 0)'
];

let allPassed = true;
for (const t of tests) {
  const res = parseStart(t);
  const ok = !!res;
  if (!ok) allPassed = false;
  console.log((ok ? '✔ PASS: ' : '✖ FAIL: ') + t + ' => ' + (res ? `(${res.x}, ${res.y}, th=${res.theta})` : 'null'));
}

console.log('All tests passed:', allPassed);
