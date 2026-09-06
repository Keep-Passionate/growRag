// Exact editable graph, using existing local packages; no network calls.
const fs = require('node:fs');
const path = require('node:path');
const { createRequire } = require('node:module');
const packageRoot = process.argv[2];
if (!packageRoot) throw new Error('Pass the local Node package directory.');
const localRequire = createRequire(path.join(path.resolve(packageRoot), '_resolver.cjs'));
(async () => {
  const viz = await localRequire('@viz-js/viz').instance();
  const svg = viz.renderString(fs.readFileSync(path.join(__dirname, 'code_architecture.dot'), 'utf8'), { format: 'svg' });
  fs.writeFileSync(path.join(__dirname, 'code_architecture.svg'), svg);
  await localRequire('sharp')(Buffer.from(svg), { density: 160 }).png().toFile(path.join(__dirname, 'code_architecture.png'));
  process.stdout.write(JSON.stringify({graphviz: viz.graphvizVersion, outputs: ['code_architecture.svg', 'code_architecture.png']}));
})().catch(error => { process.stderr.write(String(error)); process.exitCode = 1; });
