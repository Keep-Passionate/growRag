# Changelog

## 0.4.1 — 2026-09-07

- Distinguish no-memory skips, execution failures, abstentions and unscored outputs.
- Add an offline-only source retrieval probe with archived BASE/FRESH replay checks.
- Preserve the original live run and document the zero-coverage outcome and competing
  explanations for the single source improvement; no additional model call or training.

Verification: 633 local tests pass, plus lint and format checks. Source replay
matches archived evidence; manual retrieval ablations carry no generated answer scores.

## 0.4.0 — 2026-09-06

- Add honest independent PRE source records and explicit source-provenance cards.
- Admit source-only candidate observations without fabricating target validation.
- Add fixed 32/16 train-only development manifests and a shared 5 CNY budget entry point.
- Connect actual source generation, procedural extraction, frozen candidate selection and target comparisons.
- Persist each completed arm before continuing; retain unavailable REUSE separately from BASE fallback.
- Add descriptive batch reports, fixed temperature configuration and offline contract tests.

Pre-run verification: 598 local and clean-export tests pass (101 added); lint and
format checks pass. Python 3.11/3.12 CI passed. Tag `v0.4.0-research.1` resolves
to the actual live-run code `8081af4`; main remains unchanged.

Research boundary: query-only lexical selection is a weak diagnostic baseline,
not an automatic trust gate. Live outcomes are reported separately after execution.

The live PRE pilot ran at `8081af4`: 32 sources yielded one candidate; all 16 targets
had no eligible memory. BASE EM 6/16, FRESH 5/16; REUSE was not executed, not 0/16.
145 real API calls cost an estimated 0.0209324 CNY. No memory-effect claim follows.
Post-run fixes distinguish unavailable memory, failed execution and abstention in
reports, and render an absent budget block as no block. Original artifacts are retained.

## 0.3.0 — 2026-09-06

Research snapshot tag: `v0.3.0-research.1` at `3b8da81`; 497 local/clean-export
tests pass, GitHub Python 3.11/3.12 CI passes. See the knowledge verification record.

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
