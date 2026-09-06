# Changelog

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
