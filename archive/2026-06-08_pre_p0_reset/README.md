# ZeroCoder / MiniLeet

MiniLeet is the executable programming world for **ZeroCoder**: a from-scratch
coding system trained from generated tasks, exact verification, and
self-generated programs rather than language-model pretraining.

The research thesis is:

> In worlds with a procedural task generator and an exact verifier, experience
> can substitute for pretraining: a coder trained by `generate -> verify ->
> distill` can acquire non-trivial algorithmic programming competence from zero
> human-produced coding data.

This repository currently contains the MiniLeet environment, typed DSL,
interpreter, task generators, collection pipeline, reranker/value-network
lineage, split/audit tooling, and experiment harness. The next planned step is
the Phase-0 hardening sprint from [PLAN.md](PLAN.md): exploit tests, hack
detectors, generation-grade tokenizers, trace tokenization, serialization audit,
and environment throughput measurement.

## Project Shape

ZeroCoder is scoped around five questions:

- **Create vs. elicit:** what can search, verification, and curriculum create
  without a pretrained model?
- **World model:** does trace/execution prediction improve synthesis from
  scratch?
- **Diffusion repair:** does evidence-guided repair-as-denoising beat fresh
  resampling at matched budget?
- **Auto-curriculum:** can learned task proposal and adversarial tests sustain
  improvement beyond a hand-built ladder?
- **Discovery:** can correctness-gated efficiency rewards find programs better
  than reference solutions?

The current MiniLeet codebase supplies the "rules of the game": typed programs,
terminating execution, visible and hidden tests, shaped rewards, traces,
procedural tasks, enumeration search, replay-style JSONL collection, learned
rerankers, and split/audit infrastructure.

## Current Scope

The DSL supports:

- `int`, `bool`, and `list[int]`
- integer constants and variables
- list length and indexing
- arithmetic and comparisons
- assignment, return, `if`, and `for item in list`

The environment exposes:

- function signatures and visible examples
- hidden tests generated from deterministic samplers
- exact typed execution with traces
- fixed-budget candidate search
- visible-test selection and hidden-test evaluation
- pass rates, shaped rewards, and solved labels

The interpreter records:

- assignments
- branch decisions
- loop iterations
- returns
- runtime errors

## Implemented Tooling

Baselines and learned selectors:

- `template`: hand-written smoke baseline
- `enumerative`: deterministic typed-AST enumeration
- `random`: shuffled typed-AST enumeration under the same budget
- `minileet.rerank`: feature MLP predicting hidden pass rate
- `minileet.transformer_rerank`: sequence Transformer reranker
- `minileet.structured_rerank`: shared-encoder structured reranker
- `minileet.candidate_set_rerank`: cross-candidate reranker/value baseline

Data, validation, and diagnostics:

- `minileet.collect` / `minileet.collect_parallel`
- `minileet.audit`
- `minileet.filter_dataset`
- `minileet.mine_hard_negatives`
- `minileet.analyze_failures`
- `minileet.split_validation`
- `minileet.eval_checkpoint`
- compact candidate-set cache tooling
- run directories under `runs/`

Task suites include starter/generated/train/eval suites, procedural train/eval
suites, hard semantic suites, bridge/contrast suites, and the
`procedural_semantic_clean_*` benchmark family.

## Methodology Rules

Headline results are position-blind and injection-free:

- no `candidate_index_log`
- no `candidate_bucket:*` tokens
- no task-conditioned gold candidate injection
- no hidden-label-aware truncation at eval time
- validation, not final eval, drives checkpoint and score-mode selection
- final eval runs once through `minileet.eval_checkpoint` on a frozen
  `best_model.pt`

Use `--with-position-features` only for an explicitly labeled search-prior
variant. Use `--allow-task-conditioned-gold` only for coverage diagnostics or
positive-example generation, with provenance reported.

Datasets, checkpoints, and results produced before the 2026-06-08 validity
break are not citable as semantic-generalization results. Raw history is kept in
[EXPERIMENT_LOG.md](EXPERIMENT_LOG.md); current protocol details are in
[PROGRESS.md](PROGRESS.md).

## Run

```bash
python3 -m minileet.cli list
python3 -m minileet.cli list --suite generated
python3 -m minileet.cli list --suite procedural_train
python3 -m minileet.cli eval
python3 -m minileet.cli eval --suite eval
python3 -m minileet.cli eval --suite eval --baseline enumerative --budget 512
python3 -m minileet.collect --suite train --budget 512 --hidden-count 64 --out data/train_enum.jsonl
python3 -m minileet.collect --suite procedural_train --budget 512 --hidden-count 32 --out data/proc_train_enum.jsonl
python3 -m minileet.audit data/train_enum.jsonl
python3 -m minileet.rerank --train data/train_enum.jsonl --eval data/eval_enum.jsonl
python3 -m minileet.transformer_rerank --train data/proc_train_250_enum.jsonl --eval data/proc_eval_60_enum.jsonl --epochs 16 --batch-size 256 --max-len 512 --dropout 0.05 --seed 1
python3 -m minileet.structured_rerank --train data/std_train_visible_pass.jsonl --eval data/std_eval_enum.jsonl --epochs 8 --run-name std_structured_1m
python3 -m minileet.sweep_structured --train data/std_train_visible_pass.jsonl --eval data/std_eval_enum.jsonl --out-root sweeps/std_structured_1m --trials 8 --epochs 4
python3 -m minileet.eval_checkpoint --model runs/train/<dataset>/<name>/<run>/best_model.pt --eval data/final_eval.jsonl --run-name final_v1
python3 -m unittest
```

Semantic-clean training/evaluation may require explicit core-overlap waivers for
documented control families; use the split validator output as the source of
truth for what was waived.

## Roadmap

The plan is phased:

1. **P0 hardening:** exploit grading tests, hack detectors, T2/T3 tokenizers,
   serialization audit, and throughput benchmarks.
2. **P1 conditional BC:** train M0 on enumerator-solved programs and
   corruption-repair pairs.
3. **P2 flywheel:** run full expert iteration with hand curriculum and
   validation-gated selection.
4. **P3 ablations:** measure world-model co-training, repair modes, and
   grammar-masked sampling.
5. **P4 self-play curriculum:** learned task proposer and adversarial
   distinguishing-test generator.
6. **P5 DSL-2:** add bounded loops, arrays, index ops, and depth-capped
   recursion.
7. **P6 Python-subset world:** move to a static, fuel-bounded Python subset.
8. **P7 NL bridge:** add a discrete NL-to-spec parser as a quarantined
   interface layer.

See [PLAN.md](PLAN.md) for the full technical plan, gates, kill criteria,
risks, and publication sequence.
