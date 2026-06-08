# Progress

Last updated: 2026-06-08

Status only. Plan: [PLAN.md](PLAN.md). Deletions: [ARCHIVE.md](ARCHIVE.md).
Project/tooling/rules: [README.md](README.md).
Raw pre-audit history: [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) (not citable). Run artifacts: `runs/`.

## Validity Break (2026-06-08)

A code audit found validity defects across data construction, training, and
evaluation. **Everything produced before 2026-06-08 — datasets, caches,
checkpoints, results — is invalid for semantic-generalization claims**,
including the archived headline numbers (50x 5,000/5,000; semantic-hard
v3 100/100; "reranker v1 saturated").

| # | Defect | Fix |
|---|--------|-----|
| 1 | Position features (`candidate_index_log`, `candidate_bucket`) leaked enumeration order | Off by default; opt-in `--with-position-features`, baked into specs/caches/checkpoints |
| 2 | Gold programs name-injected at index 1, unmarked | Off by default; `program.source` provenance per row |
| 3 | Checkpoint/score-mode selection on the eval set | Val split from train drives all selection |
| 4 | Eval candidate pools truncated using hidden labels | Label-aware truncation is train-only |
| 5 | In-fold teacher; z-scored distillation target; teacher leaked into inputs | OOF teacher, raw targets, feature/target decoupled |
| 6 | semantic_hard train/eval share identical semantics | Core-overlap validation fails by default; superseded by semantic_clean |
| 7 | New eval required retraining (CUDA non-determinism breaks dev/final) | `eval_checkpoint` on frozen self-describing checkpoints |
| 8 | Run-dir same-second collisions | Atomic creation |
| 9 | Grading/env bugs (`True == 1`, scoping, sampler bias, visible/hidden reuse) | Strict types, scope rules, sampler fix, disjoint tests |
| 10 | No step budget; silent hidden-test under-fill | Interpreter fuel + trace cap; under-fill reported |
| 11 | paircmp semantics recur across semantic_clean splits behind a replicate suffix | Validator strips it; exact/core waivers split; paircmp is a labeled control |

Root cause of non-detection: results were appended as raw log blocks with no
methodology review. Results now enter only via the README rules.

## Current Status

- Code: all defects fixed; 68 tests green; `eval_checkpoint` reports
  solved | covered per family.
- Data: `semantic_clean_{train,dev,final}` collected injection-free with
  per-row provenance (verified label-identical to the first pass).
  Held-out oracle ceiling: dev 46/80, final 44/80. paircmp is an
  in-distribution control, never in the headline. nested/twostat are beyond
  the enumerator — that headroom is the create-vs-elicit target.
- No model results exist under the new protocol yet.
- Stale pre-audit facts still cited in PLAN.md §1 (51% underdetermination,
  pilot 6/10–7/10) need re-measurement; PLAN §0's "no degenerate tasks" check
  does not exist yet.

## Next

PLAN.md Phase 0: exploit-grading battery, hack detectors, T2/T3 tokenizers,
serialization audit, env throughput benchmark.

## Results

None yet. One line + run-dir link per result, after passing the README rules.
