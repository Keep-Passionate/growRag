// Local Graphviz + Sharp; no network, API key, or generated image service.
const fs = require('node:fs');
const path = require('node:path');
const { createRequire } = require('node:module');
const packageRoot = process.argv[2];
if (!packageRoot) throw new Error('Pass the existing local Node package directory.');
const localRequire = createRequire(path.join(path.resolve(packageRoot), '_resolver.cjs'));
(async () => {
  const viz = await localRequire('@viz-js/viz').instance();
  const svg = viz.renderString(fs.readFileSync(path.join(__dirname, 'architecture.dot'), 'utf8'), { format: 'svg' });
  fs.writeFileSync(path.join(__dirname, 'architecture.svg'), svg);
  await localRequire('sharp')(Buffer.from(svg), { density: 150 }).png().toFile(path.join(__dirname, 'architecture.png'));
  process.stdout.write(JSON.stringify({graphviz: viz.graphvizVersion, outputs: ['architecture.svg', 'architecture.png']}));
})().catch(error => { process.stderr.write(String(error)); process.exitCode = 1; });
