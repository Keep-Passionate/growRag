// Reuses the project's local Graphviz renderer; no network or model calls.
const fs = require('node:fs');
const path = require('node:path');
const { createRequire } = require('node:module');
const packageRoot = process.argv[2];
if (!packageRoot) throw new Error('Pass the installed local Node package directory.');
const localRequire = createRequire(path.join(path.resolve(packageRoot), '_resolver.cjs'));
(async () => {
  const viz = await localRequire('@viz-js/viz').instance();
  for (const name of ['architecture', 'paired_attribution']) {
    const svg = viz.renderString(fs.readFileSync(path.join(__dirname, name + '.dot'), 'utf8'), { format: 'svg' });
    fs.writeFileSync(path.join(__dirname, name + '.svg'), svg);
    await localRequire('sharp')(Buffer.from(svg), { density: 145 }).png().toFile(path.join(__dirname, name + '.png'));
  }
  process.stdout.write(JSON.stringify({graphviz: viz.graphvizVersion, outputs: ['architecture', 'paired_attribution']}));
})().catch(error => { process.stderr.write(String(error)); process.exitCode = 1; });
