'use strict';

const fs = require('node:fs');
const path = require('node:path');
const {randomUUID} = require('node:crypto');

function createMotionStore(directory = path.join(__dirname, 'runs/motion/sessions')) {
  const validId = value => typeof value === 'string' && /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/.test(value);
  function atomic(filename, value) {
    const bytes = JSON.stringify(value) + '\n';
    if (Buffer.byteLength(bytes) > 16 * 1024 * 1024) throw Error('Motion evidence exceeds 16 MiB');
    fs.mkdirSync(directory, {recursive: true});
    const temporary = `${filename}.${randomUUID()}.tmp`;
    fs.writeFileSync(temporary, bytes, {flag: 'wx'});
    fs.renameSync(temporary, filename);
  }
  function read(id) {
    if (!validId(id)) throw Error('Invalid motion session ID');
    const filename = path.join(directory, `${id}.json`);
    if (!fs.existsSync(filename) || fs.statSync(filename).size > 16 * 1024 * 1024) throw Error('Motion session unavailable or exceeds size limit');
    const record = JSON.parse(fs.readFileSync(filename, 'utf8'));
    if (record.id !== id || record.schemaVersion !== 1) throw Error('Invalid saved motion session');
    return record;
  }
  return {
    create(result) {
      const record = {schemaVersion: 1, id: randomUUID(), savedAt: new Date().toISOString(), ...result};
      atomic(path.join(directory, `${record.id}.json`), record);
      atomic(path.join(directory, 'latest.json'), {id: record.id});
      return record;
    },
    read,
    latest() {
      const filename = path.join(directory, 'latest.json');
      if (!fs.existsSync(filename)) return null;
      if (fs.statSync(filename).size > 1024) throw Error('Invalid motion pointer');
      return read(JSON.parse(fs.readFileSync(filename, 'utf8')).id);
    }
  };
}

module.exports = {createMotionStore};
