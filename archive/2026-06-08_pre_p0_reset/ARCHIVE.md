# Archive Record

Date: 2026-06-08. Everything listed here was deleted as pre-audit dead weight
(see PROGRESS.md "Validity Break"). All datasets are deterministically
regenerable from code + suite + seed via `minileet.collect_parallel`; the
result numbers survive in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) but are not
citable. Models are not regenerable and were invalid (position features /
gold injection / tainted data).

## Deleted

**Datasets (`data/`, ~17 GB)** — all pre-audit collections:
scaled_{train,final_eval}_50x (13 GB), cache_scaled_50x (3.8 GB),
semantic_hard v1/v2/v3 train/dev/final (9 files), std_{train,eval},
proc_{train,eval} variants, hard_* variants (train/eval/final/mixed/mined),
bridge/strict_bridge/first_exact/targeted_contrast/hard_plus_* variants,
the three `*.pre_provenance.bak` files (verified label-identical to
their recollected replacements before deletion), and
`{train_enum,eval_enum,hard_eval_100_enum}.jsonl` (briefly retained for a
strict-regrade measurement, then dropped: with all pre-audit data deleted,
regrading proves nothing the exploit battery and property tests don't prove
directly; the flip statistic is regenerable from git + seed if ever wanted).

**Run artifacts (`runs/`, ~550 MB)** — all pre-audit training/eval/analysis
runs and checkpoints: runs/train/{scaled_50x, semantic_hard_*, hard_split,
standard}, runs/eval/semantic_hard_*, runs/analysis/{semantic_hard_*,
hard_split, standard}, runs/smoke, runs/cache_build, sweeps/.

**Post-break reranker runs** — runs/{train,analysis}/semantic_clean_nogold
(2026-06-08 morning, position-blind, no-gold). Deleted because they are
reranker runs: the reranker gets no further investment, and these were not
protocol-grade (final measured by retraining; predate eval_checkpoint and the
paircmp-control convention). The single frozen-reranker handoff baseline is
produced once at P1 (train on dev, freeze, eval_checkpoint on final) and is
the only reranker artifact that will ever be kept.

## Kept

- `data/semantic_clean_{train_20x_visible_pass,dev_100_enum,final_100_enum}_nogold.jsonl`
  — the post-break datasets (clean, provenance-recorded). P1 training data.

**Code (reranker lineage, removed 2026-06-08)** — `minileet/{rerank,
transformer_rerank, structured_rerank, candidate_set_rerank,
candidate_set_compact, sweep_structured, sequence, features}.py` and their
tests. Reason: selector models over a frozen enumerator — capped at its
coverage by construction, featurized by hand-written regexes over the
enumerator's own templates (template-fingerprint matching, not program
reading), and human-engineered features are a forbidden prior inside the
zero-prior loop. No gate uses them; the value head's gates are ECE,
selection-vs-first-visible, and gap-to-oracle. Pre-audit versions are in git
history (commit 5bac904); generic row helpers and selection metrics were
extracted to `minileet/rows.py` and `minileet/selection.py` before removal.
