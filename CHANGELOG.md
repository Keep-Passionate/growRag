# Changelog

## 0.3.0 — 2026-09-06

- Add PRE-retrieval BASE/FRESH/REUSE comparison over the typed-card external RAG loop.
- Fix form/intent before card selection; use one shared paired prompt version.
- Validate independent adapters, declared configuration and provenance before execution.
- Separate gold-free runtime from post-run layered evaluation and all three contrasts.
- Track planned versus executed actions, failed arms, unknown evidence/usage and harm types.
- Add exclusive raw JSON, machine-readable report and escaped human-readable Markdown output.
- Add a standalone, explicitly scripted zero-API demonstration and regression tests.
- Preserve ordinary component failures without silently losing other comparison arms.

Research status: executable controlled comparison, NOT an automatic trusted-memory
selector or new evidence of model-quality gains. No new live API run or training.
POST-retrieval shared-prefix comparison and semantic applicability remain future work.

## 0.2.0 — 2026-09-06

Research snapshot tag: `v0.2.0-research.1` on `codex/query-card-actions-v1`.

- Add a bounded external RAG loop with separate original question and search query.
- Add explicit rewrite form/intent/source and typed paraphrase/expansion card bodies.
- Add diagnostic card views with multi-source isolation and versioned prompt handling.
- Preserve parent cards; changed representations restart as unvalidated candidates.
- Add an isolated local pre-retrieval QPP diagnostic, not an enabled reuse gate.
- Add editable architecture figures and implementation notes.
- Keep legacy text-memory prompt behavior; typed cards require the new generator.
- Add the `qpp` optional dependency group to CI.

Research status: interface and local diagnostic validation only. No claim that
typed memory improves answer quality; automated applicability routing remains pending.
