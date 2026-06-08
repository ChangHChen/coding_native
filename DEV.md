# ZeroCoder — Build & Prove Manifest
### Exactly what gets built, and the proof that it works, per stage

**Definition of done (applies to every item below):** an item is done when (1) the artifact exists as a named module/dataset/config, (2) its proof runs automatically — in CI for correctness proofs, in the eval harness for capability proofs — and (3) the proof's output is logged to the run tree. No item is "done by inspection." Thresholds marked ◇ are pre-registered defaults: adjustable before the phase starts, frozen once its results exist.

---

## P0 — Environment hardening & substrate (2–3 wk)

### Build
1. **Env soundness fixes** (assumed largely done): ForEach zero-iteration-sound type merge; If intersection merge; operator whitelist validated at construction; type-strict grading (`bool` never grades as `int`; per-case type-sensitive `passed`); single typecheck per evaluation; signature-vs-task validation.
2. **Task-suite repairs:** degeneracy filter at generation time (hidden-output entropy floor — no constant-output tasks); disjoint visible/hidden input sampling (no recycling); semantic-hard eval regenerated with core-name separation from train; `_contrast_list` length-target fix.
3. **Honest enumeration mode** in search: name-conditioned oracle candidates excluded from all baseline runs; oracle programs retained only as tagged reference solutions.
4. **Methodology rails:** checked-in three-tier split manifests (train / validation / frozen-eval) with core-name dedup; core-name tier in split_validation; one shared `UNSCORED` constant; one denominator convention (all tasks) across rerank / compact / analyze_failures; task-contiguity guard in audit and cache build; `--overwrite` standardized everywhere.
5. **CI exploit battery** (`tests/exploits/`): adversarial programs per known hole (bool-return, ForEach-on-empty, op-string abuse, constant-output, …), each required to grade as *fail*.
6. **Hack detectors** (`detectors.py`): constant-output passer detector, visible/hidden divergence spike, zero-trace passer, per-generation audit hook. Incidents quarantine the program from replay and emit a report.
7. **T2 tokenizer** (`ast_tokens.py`): arity-typed pre-order AST linearization; `encode`/`decode`; per-slot grammar-mask function `valid_tokens(partial_state)`; canonical int-literal and variable-slot tokens.
8. **T3 tokenizer** (`trace_tokens.py`): event-typed trace tokens (assign / branch / loop-step) with exact values and variable identity; bounded divergence-window excerptor.
9. **Evidence builder** (`evidence.py`): EvalResult → EVIDENCE tokens — visible pass bit-vector; per failing test: args, expected, observed-or-typed-error; divergence pointer + local state; `trace_len` vs reference.
10. **serialization.py** audited against the schema checklist; **throughput bench** (program·test execs/sec) and a written go/no-go on porting the interpreter hot loop.

### Prove
- Exploit battery: **100%** of adversarial programs grade fail, in CI.
- Soundness property test: 10⁶ generator-random well-typed programs run on random valid inputs with **zero** type crashes; ForEach-empty regression green.
- Grading strictness: differential re-grade of all existing collected rows; every label flip belongs to a known leniency class; datasets relabeled.
- Suite hygiene (audit over regenerated suites): **0** constant-hidden-output tasks; visible→hidden input recycle rate **= 0**; core-name overlap train↔frozen-eval **= 0**.
- T2: `decode(encode(p)) == p` on 100% of suite reference programs **and** 10⁶ random ASTs. Grammar-mask soundness: every reference program accepted token-by-token. Grammar-mask completeness: 10⁵ mask-guided random rollouts → **100%** parse + typecheck.
- T3: token-rendered trace matches interpreter trace event-for-event on 10⁵ (program, input) pairs; excerpt always contains the divergence step.
- Evidence: golden-file tests; size ≤ ◇160 tokens on 100% of collected failures.
- Throughput: measured rate vs ◇target; P2 generation wall-clock projection documented and within budget.

---

## P1 — Conditional BC: generation-0 distillation (3–4 wk)

### Build
1. **Model M0** (`model.py`): masked absorbing-state diffusion transformer, 10–25M params, ctx 512; segment embeddings (SPEC / PROGRAM / EVIDENCE / TRACE); value head; config-as-dataclass checkpointing.
2. **Training objective:** masking-ratio-sampled CE on PROGRAM; value regression/BCE on env labels.
3. **Sampler** (`sample.py`): confidence-based parallel unmasking, temperature; v0 validity = typecheck-reject (grammar-masked path implemented behind a flag, ablated in P3).
4. **Data builders:** (spec → program) pairs from honest-enumeration solved rows, AST-hash deduped; **corruption-repair generator** (`corrupt.py`): op flip / constant shift / comparison swap on solved programs, *executed* to harvest real evidence → (spec ⊕ evidence ⊕ corrupted → original) pairs.
5. **Eval harness:** pass@k runner (sample → typecheck → visible gate → hidden grade; train/val only); value-head ECE; selection metrics on the established conventions. Frozen-eval physically gated behind a separate command + flag.
6. **Replay schema v1** (`buffer.py`): (task, candidate, labels, evidence, parent-link) rows — parent-link is the repair-chain hook.

### Prove
- Wiring/capacity: memorize 100 tasks to pass@1 ≥ **99%**.
- Stability: val masked-CE decreases to plateau, no NaN, across **3 seeds**.
- **pass@64 ≥ ◇90% of honest-enumeration coverage** on train-distribution validation tasks.
- **> 0 solves** on held-out hard validation families (exact count logged — this number is the first evidence of generalization).
- Invalid-sample rate **< ◇20%** under typecheck-reject; failure-type histogram logged.
- Corruption-repair fix-rate ≥ **◇60%** at 1 round, ≥ ◇80% at 3 rounds (stretch), on held-out corruptions.
- Value head: ECE **< ◇0.15** on validation candidates; AUC vs the frozen reranker logged — this is the handoff baseline.
- Sampling throughput (samples/sec/GPU) measured; P2 cost projection updated.

---

## P2 — Flywheel: expert iteration (6–8 wk)

### Build
1. **Generation driver** (`exit_loop.py`): SAMPLE (K fresh + R repair rounds) → VERIFY → SELECT (env-solved only; near-winner path present but OFF) → DISTILL → REPLAY → MEASURE; temperature schedule; config-driven budgets.
2. **Repair v0** (heuristic-localized): divergence step → AST node → T2 span mapping; remask ± one neighbor; re-denoise with EVIDENCE clamped; AST-hash cycle guard; **value-head submit threshold** (calibrated stop rule).
3. **Replay v2:** mine_hard_negatives policy as a library call; staleness window W; per-task caps; **hindsight-pair extraction** from every solved repair chain.
4. **Instrumentation:** detectors + audit hooks per generation; generation-curve logger; **capability-per-FLOP ledger** (train FLOPs + sampling FLOPs + env executions, one accounting module).
5. **Curriculum bands** with [60%, 90%] promotion rule.
6. **Frozen-eval runner** — one invocation per phase review, access-gated.

### Prove
- **Flywheel:** ≥ **3 consecutive generations** of improved held-out `solved@select` on validation (the curve is the artifact).
- **Beat honest enumeration at matched program-evaluation budget** on validation-hard.
- Value head ECE **< ◇0.10**; **handoff proof:** value-head selection ≥ frozen-reranker selection for 2 consecutive generations before it becomes primary (both logged forever after).
- Repair v0 has **positive marginal solves per unit compute** vs fresh sampling (else escalate to P3 arm-(b) priority).
- **Label purity = 100%** in every distillation batch (assertion: only env-solved), near-winner path verified OFF.
- Detector incidents = 0 unexplained; any incident → quarantine + written report (the report is a deliverable, not a failure).
- **Reproducibility:** generation *k* re-run from (seed, config, buffer snapshot) reproduces the selection set byte-identically, or nondeterminism is bounded and documented.
- Stall playbook executable: coverage-vs-ranking decomposition (analyze_failures over model candidates) runs per generation.

---

## P3 — Ablations: the two bets + decoding (4 wk, overlaps P2)

### Build
1. **Trace co-task:** TRACE-canvas denoising given (program, input); mixing-ratio config (4:1 / 1:1 / 1:4).
2. **Outcome-prediction co-task:** per-test pass-bit prediction head/task.
3. **Repair v1** (evidence-conditioned): clamp SPEC + EVIDENCE, whole program at low noise, confidence decides the edit — no external localizer.
4. **Grammar-masked unmasking** wired into the production sampler.
5. **Ablation harness:** fixed-seed grid — {WM on/off} × {fresh / repair-a / repair-b} × {grammar / reject} — matched-budget accounting, ≥ 3 seeds per cell.

### Prove
- **World model (Q2):** WM co-training ≥ **◇1.5×** sample efficiency to a fixed validation solved-rate, in ≥ 2 of 3 seeds; full curves for both arms published either way.
- **Repair (Q3):** best repair arm ≥ **◇1.3×** solves at matched budget vs fresh; (b) vs (a) delta reported.
- **Grammar masks:** invalid rate → ~0 **and** net wall-clock per solve ≤ rejection baseline (masks must pay for themselves); solve-rate non-inferior.
- **Execution-prediction competence** (reported regardless): next-event accuracy and final-value exact-match ≥ ◇90% on validation programs within length bound — the standalone world-model metric.
- Nulls documented with identical rigor; thesis text updated per the P3 kill rule.

---

## P4 — Self-play curriculum (6 wk)

### Build
1. **Task proposer** (`proposer.py`): emits spec-language tasks (signature template + oracle program + sampler params); validity pipeline — oracle typechecks and executes, hidden-output entropy floor, novelty distance, dedup; **learnability-band reward** (solver success in [20%, 80%]).
2. **Distinguishing-test generator** (`distinguisher.py`): given two visible-equivalent candidates, propose an input where outputs differ; oracle adjudicates; successful tests appended to the task's visible set (capped).
3. **Curriculum manager:** mixes hand ladder + proposed pool; promotion logic; pool telemetry (difficulty histogram, diversity, age).

### Prove
- Proposer yield: ≥ **◇50%** of proposals pass all validity checks; **0** entropy-floor violations enter the pool; pool novelty (mean min-distance) above floor.
- **Transfer gate:** training on the proposer curriculum ≥ hand ladder on the frozen hard splits.
- **Distinguisher gate:** effective underdetermination on the train pool **51% → < ◇25%** (fraction of first-visible-passers that still fail hidden after augmented visible sets); per-attempt distinction success rate logged.
- Closed-loop stability: ≥ **5 generations** with the proposer in the loop, validation curve non-decreasing, no degenerate-task flood.

---

## P5 — DSL-2 (8–10 wk)

### Build
1. **Language:** fuel-bounded `while`, arrays + index/slice ops, depth-capped recursion; typechecker + interpreter + TraceEvent extensions; termination via fuel (proved by construction).
2. **Generators:** new families (search, two-pointer, prefix-sum, in-DSL sorting); **efficiency tasks with deliberately suboptimal references**; empirical growth-rate fitter (steps vs n).
3. **M1** (50–100M, ctx 1–2k); T2/T3 extended to the new constructs.
4. P0 battery regenerated for DSL-2 (exploits, degeneracy, splits, round-trips).

### Prove
- **All P0 proofs re-pass on DSL-2** (round-trips at 10⁶, soundness property test, exploit battery, suite hygiene).
- **P2 gates re-pass at M1 in DSL-2:** flywheel ≥ 3 generations; beat honest enumeration at matched budget.
- **Efficiency:** model matches-or-beats reference `trace_len` on ≥ **◇30%** of suboptimal-reference tasks; ≥ 1 verified strict improvement with a better fitted growth class = stretch (the discovery ticket, fast-tracked if it lands).
- **Length health:** invalid-rate and solve-rate vs program-length curves; degradation past threshold triggers the block-diffusion contingency, decision documented.

---

## P6 — Python-subset world (10–12 wk)

### Build
1. **PyWorld:** whitelisted static Python subset (AST-node whitelist; no imports/IO; fuel-bounded loops; lists/dicts/strings); **sandboxed traced interpreter** — per-episode container, deterministic reset, resource caps (the InterCode template); result-set and **state-diff grading** for effectful tasks; spec format unchanged.
2. **T2-py / T3-py** tokenizers for the subset.
3. **Task corpus:** curated formal-spec LeetCode-easy/medium equivalents with hidden suites + property tests; split manifests.
4. **M2** (100–300M, ctx 4k); multi-GPU ExIt infrastructure.

### Prove
- **Sandbox:** escape-attempt battery fails to escape; resource caps enforced; reset determinism (seed → identical episode); containerized throughput ≥ ◇target.
- Tokenizer round-trips on the full corpus + 10⁶ random subset ASTs.
- **Flywheel turns in PyWorld** at M2: ≥ 3 generations of validation improvement.
- **Headline gate (one frozen-eval shot):** ≥ **50%** solved@select on LC-easy equivalents; ≥ **20%** on medium; pass@64 alongside.
- **Capability-per-FLOP curve across {M0, M1, M2}** assembled and published — the Q1 deliverable exists whether or not the headline gate passes.

---

## P7 — NL bridge (8–10 wk, parallel from P5)

### Build
1. **Spec-language v2:** constraint slots, resource bounds, property assertions — every slot with an executable checker.
2. **Template NL generator** (pure pair factory) and **paraphrase pipeline** — impure datasets versioned and flagged; a **purity ledger** mapping every training run to its datasets.
3. **NL→spec parser** (separate weights); round-trip validators (predicted spec typechecks; its examples execute consistently).
4. **Contrastive dual encoder** (NL ↔ spec): retrieval + auxiliary requirement-match score.
5. **End-to-end harness:** NL → parser → spec → coder → submit, with failure attribution.

### Prove
- Spec v2: every slot machine-checkable (unit tests; property-assertion runner over oracles).
- Parser: ≥ **◇95%** exact-match spec parse on held-out templates; ≥ **◇99%** of predicted specs pass the executability/consistency validators — i.e., parser errors are *detectable*, which is the design claim of the discrete interlingua.
- **End-to-end gate:** NL→solve within **◇5 points** of spec→solve on templated NL.
- **Separability proof:** 100% of end-to-end failures attributable to parse-fail vs synth-fail by the harness — the argument for the architecture, demonstrated, not asserted.
- Contrastive: retrieval R@1 ≥ ◇target on held-out pairs; match-score ensembling with the value head adds measurable selection value (or the null is reported).
- **Purity audit in CI:** coder training runs provably consumed only pure-tagged datasets.

---

## Dependency spine

P0 → P1 → P2 → P3 form the critical path; P4 forks after P2; P7 forks after P5's spec-v2 work begins; P5 → P6 sequential. Every phase's *Prove* list is its exit interview: the phase review consists of running the list and reading the logs, nothing else.