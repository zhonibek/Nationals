'use strict';

const fs = require('node:fs');
const path = require('node:path');
const {randomUUID} = require('node:crypto');

function createStore(directory = path.join(__dirname, 'runs/nemotron')) {
  const validId = value => typeof value === 'string' && /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/.test(value);
  function save(record) {
    if (!validId(record.id)) throw Error('Invalid Nemotron session ID');
    fs.mkdirSync(directory, {recursive: true});
    const filename = path.join(directory, `${record.id}.json`);
    const temporary = `${filename}.${randomUUID()}.tmp`;
    fs.writeFileSync(temporary, JSON.stringify(record, null, 2) + '\n', {flag: 'wx'});
    fs.renameSync(temporary, filename);
    return record;
  }
  return {
    save,
    create: (prompt, plan) => save({...plan, id: randomUUID(), createdAt: new Date().toISOString(), prompt}),
    read(id) {
      if (!validId(id)) throw Error('Invalid Nemotron session ID');
      const filename = path.join(directory, `${id}.json`);
      if (!fs.existsSync(filename)) throw Error('Nemotron session not found');
      const stat = fs.statSync(filename);
      if (stat.size > 1024 * 1024) throw Error('Nemotron session exceeds size limit');
      const record = JSON.parse(fs.readFileSync(filename, 'utf8'));
      if (record.id !== id || record.schemaVersion !== 1) throw Error('Invalid saved Nemotron session');
      return record;
    },
    latest() {
      if (!fs.existsSync(directory)) return null;
      const files = fs.readdirSync(directory).filter(name => validId(name.replace(/\.json$/, '')) && name.endsWith('.json'));
      if (!files.length) return null;
      const latest = files.map(name => ({name, modified: fs.statSync(path.join(directory, name)).mtimeMs}))
        .sort((left, right) => right.modified - left.modified)[0];
      return this.read(latest.name.slice(0, -5));
    }
  };
}

module.exports = {createStore};
