'use strict';

const fs = require('node:fs');
const path = require('node:path');
const {createHash} = require('node:crypto');

const DEFINITIONS = Object.freeze({
  'inspect-robot': ['get_robot_profile', 'finish_analysis'],
  'prepare-motion-experiment': ['get_motion_contract', 'prepare_reach_pose'],
  'diagnose-motion': ['get_saved_motion_evidence', 'finish_analysis'],
  'verify-motion-improvement': ['get_learning_summary', 'finish_analysis'],
  'plan-game-tactics': ['get_game_rules', 'get_game_snapshot', 'finish_analysis']
});
const MAX_SKILL_BYTES = 12 * 1024;
const MAX_LOADED_SKILLS = 2;
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');

function createRegistry(directory = path.join(__dirname, 'skills')) {
  function read(name) {
    if (!Object.hasOwn(DEFINITIONS, name)) throw Error('Unknown reviewed agent skill');
    const folder = path.join(directory, name);
    const filename = path.join(folder, 'SKILL.md');
    if (fs.lstatSync(folder).isSymbolicLink() || fs.lstatSync(filename).isSymbolicLink()) throw Error('Agent skill symlinks are not allowed');
    const stat = fs.statSync(filename);
    if (!stat.isFile() || stat.size > MAX_SKILL_BYTES) throw Error('Agent skill exceeds file budget');
    const bytes = fs.readFileSync(filename);
    const match = /^---\r?\n([\s\S]*?)\r?\n---\r?\n([\s\S]+)$/.exec(bytes.toString('utf8'));
    if (!match) throw Error('Agent skill requires frontmatter and instructions');
    const metadata = {};
    for (const line of match[1].split(/\r?\n/)) {
      const field = /^(name|description|compatibility|allowed-tools): ([^\r\n]+)$/.exec(line);
      if (!field || Object.hasOwn(metadata, field[1])) throw Error('Unsupported or duplicate skill frontmatter');
      metadata[field[1]] = field[2].trim();
    }
    if (metadata.name !== name || !metadata.description || metadata.description.length > 1024 ||
        (metadata.compatibility && metadata.compatibility.length > 500) || !match[2].trim()) throw Error('Invalid agent skill metadata');
    const allowedTools = metadata['allowed-tools']?.split(/\s+/) ?? [];
    if (JSON.stringify(allowedTools) !== JSON.stringify(DEFINITIONS[name])) throw Error('Skill tools do not match the reviewed allowlist');
    return {name, description: metadata.description, compatibility: metadata.compatibility,
      allowedTools, sha256: sha256(bytes), instructions: match[2].trim()};
  }
  function catalog() {
    return Object.keys(DEFINITIONS).map(name => {
      const {instructions, ...metadata} = read(name);
      return metadata;
    });
  }
  function session() {
    const available = catalog();
    const loaded = new Map();
    return {
      catalog: available,
      load(name) {
        if (!loaded.has(name) && loaded.size >= MAX_LOADED_SKILLS) throw Error('Agent skill activation budget exceeded');
        const skill = read(name);
        if (skill.sha256 !== available.find(entry => entry.name === name)?.sha256) throw Error('Agent skill changed during this session');
        loaded.set(name, skill);
        return skill;
      },
      permits: tool => [...loaded.values()].some(skill => skill.allowedTools.includes(tool)),
      provenance: () => [...loaded.values()].map(({instructions, ...metadata}) => metadata)
    };
  }
  return {catalog, session};
}

module.exports = {createRegistry, DEFINITIONS, MAX_SKILL_BYTES, MAX_LOADED_SKILLS, sha256};
