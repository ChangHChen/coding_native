# ZeroCoder — Project Plan
### A from-scratch, pure-RL, diffusion-native coding world model

**Working name:** ZeroCoder (internal). Public framing avoids "AlphaZero" in titles; the lineage is claimed in the text, not the headline.
**One-sentence thesis:** In worlds that have a task generator and an exact verifier, experience can substitute for pretraining — a transformer-diffusion coder trained purely by *generate → verify → distill*, with zero human-produced data, can reach broad, non-trivial competence on algorithmic programming, and the computer world is the one domain where this claim gets a fair trial.

---

## 0. Positioning, ground rules, and the questions being answered

**Positioning (the 2×2).** Narrow + zero-prior is done (AlphaDev, AlphaTensor: per-instance search, superhuman artifacts, model discarded). Broad + human-prior is done and hot (FunSearch, AlphaEvolve, Absolute Zero, R1-Zero: pretrained LLM as the proposal engine). **Broad + zero-prior is empty.** Both neighbors explain why: AlphaDev needed narrowness to make search tractable without a prior; AlphaEvolve needed the pretrained prior to make breadth tractable without unbounded search. Our substitute for the prior is the **procedural task generator + free exact verification**: experience is manufactured, not collected. The generator is the load-bearing structural innovation, not scaffolding.

**Definition of "zero" (pre-registered).** *No human-produced data anywhere in the coder.* Allowed, AlphaZero-consistently: the rules of the game (DSL grammar, type system, interpreter), the simulator, and supervised objectives on **self-generated** targets (search outcomes, env labels, traces). Expert iteration is cross-entropy on our own verified wins — that is the AlphaZero recipe itself, not a violation. One quarantined impurity is permitted **only at the NL interface** (Phase 7), exactly as vision-language systems accept human captions at the perception boundary while the policy never sees human data.

**Scientific questions (each phase answers at least one):**

- **Q1 — Create vs. elicit.** What is the capability-per-FLOP of search + verification + curriculum with *no base model to elicit from*? (Direct, unconfounded evidence for the live RLVR debate.)
- **Q2 — World model.** Does co-training execution prediction (trace/world-model objectives) improve synthesis sample-efficiency *from scratch*? (CWM showed it with pretraining at scale; from-scratch is open.)
- **Q3 — Diffusion as improvement operator.** Does trace-divergence-guided **repair-as-denoising** beat independent resampling at matched budget? (The diffusion-specific bet.)
- **Q4 — Auto-curriculum.** Can a learned task proposer + adversarial test generator sustain improvement beyond the hand-built ladder? (The self-play substitute; open-endedness constituency.)
- **Q5 — Discovery (AlphaDev inheritance).** With efficiency-graded rewards, does the loop find solutions *better than the references* on cost axes? (Keeps a lottery ticket on a discovery headline at near-zero marginal cost.)

**Methodological discipline (assumed in force throughout).** All environment-integrity and evaluation-methodology issues from the code audit are fixed: sound typechecking (ForEach zero-iteration merge, If intersection merge), type-strict grading (`bool ≢ int`), validated operators, no degenerate (constant-output) tasks, no visible→hidden input recycling, honest enumeration baseline (name-conditioned oracle candidates excluded from all baselines), task-level validation split driving **all** selection (checkpoints, score modes, hyperparameters), frozen final-eval splits touched once per milestone, one denominator convention codebase-wide, core-name (profile-stripped) overlap checking, single shared UNSCORED protocol, contiguity-guarded caches. Every later number assumes this foundation; **in expert iteration a grading hole is not a metrics bug — it gets distilled into the policy.**

---

## 1. Phase 0 — What is already built (asset inventory)

The existing MiniLeet codebase is the AlphaZero anatomy, minus the learned policy:

| Component (built) | Role in this program |
|---|---|
| Typed, terminating DSL + interpreter with TraceEvents | Rules of the game + free, exact, instrumented simulator |
| Visible/hidden tests + shaped reward (0.2/0.8, solved bonus) | Proxy + ground-truth objective; dense shaping for cold start |
| Procedural task generators (families × operators × profiles), held-out dev/final hard splits, semantic compositions, bridge/contrast suites | Curriculum engine + the generalization instrument |
| Enumeration search | Generation-0 policy ("random play") and the matched-budget baseline |
| collect / collect_parallel / serialization | Replay-buffer infrastructure (task-contiguous JSONL, trace-bearing rows) |
| Feature + sequence tokenizers; MLP / flat-transformer / structured / candidate-set rerankers; compact memmap cache | **Value-network lineage** — learned verifiers predicting hidden outcome from visible evidence |
| audit / analyze_failures / mine_hard_negatives / filter / split_validation / runs | Loop diagnostics, prioritized replay, dataset curation, split hygiene, experiment bookkeeping |

**Measured facts that drive the design (baseline numbers to beat / exploit):**

- Visible tests underdetermine the task ~**51%** of the time (first visible-passer fails hidden) → justifies the learned verifier *and* the adversarial distinguishing-test game (Phase 4).
- On the small pilot eval: first-visible **6/10** solved, learned candidate-set reranker **7/10** (mean hidden 0.956), oracle-over-visible-passers **10/10** → the value-network signal is real; the oracle gap is the headroom.

**Phase-0 remaining work (hardening sprint, ~2–3 weeks):**

1. Adversarial grading test battery: a directory of known-exploit programs (bool-return, ForEach-on-empty, op-string abuse, etc.) that must all grade as failures, run in CI.
2. **Hack detectors as first-class instrumentation**: per-generation audits for constant-output programs scoring reward, visible/hidden divergence spikes, trace anomalies. Hacking incidents are *data* (safety constituency), not just bugs.
3. **T2 tokenizer** — generation-grade, lossless AST linearization (see §2). **T3 tokenizer** — faithful trace encoding (see §2). The existing scoring tokenizer (T1) is kept for the external reranker only.
4. serialization.py audited against the schema checklist; env throughput benchmark (programs·tests/sec) — if the pure-Python interpreter is the bottleneck, port the hot loop (Rust/Cython) or batch via process pools. AlphaZero spent most compute on self-play; we will too.

---

## 2. The model

**Architecture.** Masked discrete-diffusion transformer (absorbing-state, MDLM/LLaDA-style training objective: random masking ratio per example, cross-entropy on masked positions), bidirectional attention, pre-norm, RoPE. Conditioning is the simplest correct mechanism: the **spec segment is never masked** — the denoiser attends to it as fixed context. Canvas layout:

```
[SPEC: signature ⊕ visible examples (⊕ description tokens later)] [PROGRAM canvas] [EVIDENCE: execution feedback (clamped when present)] [optional TRACE canvas (co-training)]
```

**Evidence segment (the iteration interface).** Synthesis is the depth-0 case of a sequential process whose state is *(spec, current candidate, execution evidence)*. When a candidate fails, the outer loop clamps an EVIDENCE segment containing: the visible pass/fail bit-vector; for each failing test, the input args, expected output, and observed output **or typed error token**; a **divergence pointer** — the first trace step where behavior went wrong, with the local variable state at that step (a bounded T3 excerpt, not the full trace); and, on efficiency tasks, `trace_len` vs. reference. Evidence is kept **Markov** — only the current candidate's evidence rides the canvas, with an AST-hash memory of failed attempts held in the outer loop to prevent cycling — so context stays bounded across repair rounds.

**Heads.**
- **Denoiser** (token logits) — the policy.
- **Value head** (predict hidden pass-rate + solved from spec ⊕ program, pooled) — the in-model verifier, trained on env labels every generation. The external candidate-set reranker is retained as an independent selection cross-check and for calibration comparison.
- **Trace head / trace co-task** — execution prediction (Q2), implemented as denoising the TRACE canvas given program+input (preferred: same objective, one model) with a dedicated segment embedding.

**Three tokenizations (strict separation of concerns):**
- **T1 (exists):** scoring/verifier encoding — bucketed values, compressed; fine for the reranker, *disqualified* for generation and world-model training.
- **T2 (build):** generation-grade program encoding — canonical pre-order AST linearization with **arity-typed tokens** (each node token implies its child count and child types), exact small-int literal tokens, named-slot variable tokens. Lossless: `decode(encode(p)) == p`. Vocabulary estimate: 300–2,000 tokens. The arity-typing is what makes per-slot **grammar masks** computable at sampling time.
- **T3 (build):** faithful trace encoding — event-typed tokens (assign/branch/loop-step) with exact values and variable identities; no `:other` collapse, no bucketing. This is world-model supervision, not a feature.

**Sampling.** Confidence-based parallel unmasking (LLaDA/Dream recipe) with a remasking schedule; temperature for exploration. **v0 validity:** sample → typecheck-reject (the typechecker is free — a luxury text dLLMs don't have; *measure* the invalid rate as a diagnostic). **v1 validity:** grammar-masked unmasking — per-position valid-token masks derived from the arity-typed linearization. Precedent says this matters: grammar-encoded models are the documented ingredient that made from-scratch RL program synthesis work without pretraining (CSG result). **Contingency for length scaling:** block-diffusion hybrid (SDAR-style — AR over blocks, diffusion within blocks) if global diffusion degrades on long programs in Phases 5–6.

**Why diffusion (the claim, falsifiable in Phase 3):** (a) infilling makes **repair a native action** — remask the implicated span, re-denoise conditioned on everything else; (b) parallel decode throughput suits a sample-hungry zero-RL regime; (c) the value-model pairing is the current best practice for dLLM RL (TraceRL's diffusion value model — structurally the component we already built).

**Scale ladder (deliberately small first — the loop is the experiment, not the parameter count):**

| Model | Params | Context | World | Rationale |
|---|---|---|---|---|
| M0 | 10–25M | 512 | MiniLeet v1 | Hours-scale iteration; vocabulary is tiny; saturating MiniLeet is expected and fine |
| M1 | 50–100M | 1–2k | DSL-2 (while/arrays/recursion) | First real generalization pressure |
| M2 | 100–300M | 4k | Python-subset world | The stated proof-of-concept scale, earned not assumed |

Scaling curves across {M0-small, M0, M1} × {data-generation budgets} are themselves a deliverable (Q1).

---

## 3. The learning algorithm

**Primary loop — Expert Iteration (the faithful AlphaZero recipe; no likelihood-ratio RL required):**

```
for generation g = 1, 2, ...:
  SAMPLE   : per task, K fresh samples from π_{g-1} (temperature-scheduled)
             + R repair rounds on the best failing candidates
  VERIFY   : run env (visible + hidden on train tasks); label everything
  SELECT   : winners = env-solved; near-winners = top verifier-ranked visible-passers
             (near-winner inclusion is OFF by default; switched on only with
              calibration evidence — distilling unverified programs is the
              main self-poisoning risk)
  DISTILL  : masked-diffusion CE on winners (spec → program)
             + value regression/BCE on ALL graded samples
             + trace co-task on sampled (program, input) executions
  REPLAY   : mine_hard_negatives-style prioritized buffer; staleness window
             of W generations; per-task candidate caps; dedup by AST hash;
             log full interaction trajectories — every solved repair chain
             decomposes into hindsight (failing program + evidence → improved
             program) pairs for the repair co-task
  MEASURE  : validation gates; frozen-eval untouched until milestone review
```

- **Repair operator (Q3) — two designs, staged:**
  *(a) Heuristic-localized (v0):* run the failing candidate; locate the **first trace divergence**; map the implicated statement(s) to T2 spans; remask those spans (± one neighbor); re-denoise with the EVIDENCE segment clamped.
  *(b) Evidence-conditioned learned repair (v1):* clamp spec + EVIDENCE, present the whole failing program at low noise, and let confidence decide what changes — the model *reads* the interpreter feedback and chooses its own edit; no external localizer.
  **Repair training data is free and pure:** (i) *hindsight near-pairs* — AST-diff failing/passing siblings of the same task in the replay buffer (the enumerator's dense candidate sets make near-pairs abundant); mask the failer's diff span, clamp its real evidence, target the passer's span; (ii) *corruption curriculum* — mutate a solved program (op flip, constant shift, comparison swap), **execute it** to harvest genuine interpreter evidence, train to restore. Denoising lifted into program space with real execution feedback; available from P1.
  **Stopping rule (submit decision):** the value head is the submit policy — repair until predicted P(solved) crosses a calibrated threshold or the round budget exhausts, then submit the value-max candidate. (No learned "submit" action needed; interactive-coding agents demonstrably waste turns on that decision.)
  Budget split fresh-vs-repair is tuned; the *measurement* is marginal solves per unit compute: fresh vs. (a) vs. (b).
- **Exploration:** temperature + mask-order randomization; optional novelty bonus on AST sketch hashes if mode collapse appears (diagnosed via per-task candidate diversity in audit).
- **Reward (with the AlphaDev axis from day one, weight ramped):**
  `reward = 1[solved] · (1 + λ · efficiency)`, `efficiency = clip((trace_len_ref − trace_len)/trace_len_ref, −1, 1)`; correctness gates, efficiency scores. A seeded subset of tasks ships with **deliberately suboptimal references** (Q5). In Phase 5+, fit empirical step-count-vs-n growth (the interpreter is free) for asymptotic-class scoring.
- **Phase-2 on-policy RL (only after ExIt plateaus; ranked options):**
  1. **TraceRL-style** trajectory-aware policy optimization with the diffusion value model (current best-performing dLLM RL; our value head is that component).
  2. **Coupled-GRPO** (DiffuCoder's complementary-masking estimator) if trajectory logprobs are too noisy.
  3. **Likelihood-free contrastive** (DiffusionNFT-style positive/negative finetuning) as the robust fallback — it degenerates gracefully toward ExIt-with-negatives.
- **World-model co-training (Q2):** multi-task batches mixing (spec→program) with (program+input→trace/output) at swept ratios (4:1, 1:1, 1:4). Secondary world-model task: **outcome prediction** (given spec, program, inputs — predict pass/fail per test), which doubles as self-evaluation skill and a third verifier signal.
- **Curriculum:**
  - *Phase ≤3:* the existing hand ladder (starter → procedural → semantic-hard), with per-band promotion when solve-rate enters [60%, 90%].
  - *Phase 4 (self-play substitutes — code has no two-player game; these are the principled replacements):*
    **(i) Learned task proposer** (Absolute-Zero-style propose-and-solve, but proposing into our spec language): reward = learnability band (solver success in [20%, 80%]) + validity + novelty (embedding distance to replay tasks). Proposals are *checked* — typecheck, oracle-executability, non-degeneracy (hidden-output entropy floor) — before entering the pool. UED framing (PAIRED-style regret) is the upgrade path.
    **(ii) Adversarial distinguishing-test generator** — directly attacks the measured 51% underdetermination: given two visible-equivalent candidates, propose an input on which they differ; the oracle adjudicates instantly; reward for successful distinction. Generated distinguishing tests are appended to tasks' visible sets, *shrinking the proxy-truth gap over time* — the environment hardens itself.

---

## 4. Measurement protocol (pre-registered)

**Split discipline.** Three tiers everywhere: train / validation (drives *all* selection: checkpoints, score modes, hyperparameters, curriculum promotion) / frozen final-eval (dev-hard + final-hard + semantic-hard-eval with core-name de-duplication), touched once per phase review. Task-level (core-name) separation enforced by the split validator.

**Primary metrics.**
- `solved@select` — one submission per task via value-head-gated selection (the deployment metric).
- `pass@k` curves (k ∈ {1, 8, 64, 256}) — separates prior quality from selection quality.
- **Sample efficiency** — env interactions (program·test executions) to reach X% on validation.
- **Generation curve** — held-out solved-rate per ExIt generation (the flywheel plot; the single most important figure of the program).
- **Capability-per-FLOP** — total accounting: train FLOPs + sampling FLOPs + env executions (Q1 headline).
- **Generalization gap** — in-distribution vs. held-out families/operators/profiles, reported separately (never averaged away).
- **Hack incidence** — detector hits per generation (Q-safety; reported even when zero).
- **Discovery count** — tasks where the model's solution beats the reference on trace_len / fitted growth class (Q5).
- **Verifier calibration** — ECE of the value head and external reranker on validation; gates near-winner distillation.

**Baselines (fixed before Phase 1, never renegotiated after results exist).**
1. **Honest enumeration** at matched program-evaluation budget (oracle-leak candidates removed).
2. First-visible selection; verifier-gated selection (existing stack).
3. Random-init model sampling (floor).
4. *Impure reference anchor (clearly marked non-zero, reported separately):* a small pretrained code LM fine-tuned on the same spec format — this quantifies exactly what pretraining buys, which **strengthens** the Q1 story regardless of which side wins.

---

## 5. Roadmap — phases, gates, kill criteria

Gate numbers below are pre-registered defaults; they may be re-justified *before* a phase starts, never after its results exist.

| Phase | Duration | Content | Exit gate (validation) | Kill/pivot trigger |
|---|---|---|---|---|
| **P0** Hardening | 2–3 wk | §1 checklist; T2/T3; detectors; CI exploit battery; env throughput | All exploit programs grade as failures; T2 round-trip lossless; throughput ≥ target | — (blocking work) |
| **P1** Conditional BC | 3–4 wk | Train M0 on enumerator-solved data **+ corruption-repair pairs** (mutate solved programs, harvest real interpreter evidence, train restoration) — gen-0 distillation; pure | pass@64 ≥ 90% of enumeration coverage on train-dist; **>0 held-out hard solves**; invalid-sample rate < 20% (v0); corruption-repair fix-rate ≥ 60% at 1 round | Architecture can't fit the distribution → revisit T2/arch before any RL |
| **P2** Flywheel | 6–8 wk | Full ExIt loop, hand curriculum | ≥3 consecutive generations of held-out improvement; **beat honest enumeration at matched budget on dev-hard**; value-head ECE < 0.1 | Generation curve flat for 5 gens after diagnosis (see §6) → pivot to instrument + negative-measurement paper (still publishable: Q1 answered "no, and here's the decomposition") |
| **P3** Ablations | 4 wk (overlaps P2) | WM co-training on/off; **fresh resample vs. heuristic-localized repair vs. evidence-conditioned repair**; grammar-mask vs. rejection | Report regardless. *Confirmation thresholds:* WM ≥ **1.5×** sample efficiency; best repair arm ≥ **1.3×** solves at matched budget | Both bets null → diffusion/world-model claims dropped from thesis; program continues as ExIt measurement (Q1, Q4, Q5 stand) |
| **P4** Self-play curriculum | 6 wk | Learned proposer + distinguishing-test generator | Proposer curriculum ≥ hand ladder on transfer to frozen hard splits; effective underdetermination on train pool **51% → <25%** with generated tests | Proposer collapses to degenerate/duplicate tasks despite checks → keep distinguisher (independently valuable), hand curriculum stays |
| **P5** DSL-2 | 8–10 wk | Bounded `while` (fuel), arrays + index ops, depth-capped recursion; new generators; M1 scale; expanded efficiency tasks (in-DSL sorting/searching → discovery candidates) | Re-pass P2 gates in DSL-2; ≥1 verified discovery (model beats reference cost on some task) is the stretch goal, not a gate | Flywheel doesn't transfer to DSL-2 at M1 → scale/curriculum study before proceeding |
| **P6** Python-subset world | 10–12 wk | Static Python subset (no imports/IO; fuel-bounded loops; lists/dicts/strings), traced interpreter, same spec format; M2 scale | **"Decent pass rate," defined:** ≥50% solved@select on a curated formal-spec set of LeetCode-easy-equivalents; ≥20% on medium-equivalents; pass@64 reported alongside | Misses by wide margin → publish the scaling/capability-per-FLOP curve as the result (Q1) and stop or seek scale |
| **P7** NL bridge | 8–10 wk, parallel from P5 | NL→spec **discrete parser** (primary); contrastive NL/spec dual encoder (auxiliary verifier signal + encoder pretraining); spec-language v2 (constraint & complexity slots). Data ladder: templated (pure) → paraphrase (quarantined impurity, explicit) → real LC statements with pre-extracted stubs/examples | ≥95% spec-parse exact-match on held-out templates; end-to-end NL→solve within 5 pts of spec→solve on templated NL | Free-form NL from zero is **out of scope by design** — the parser is the interface; failures here never gate P1–P6 |

**Critical path:** P0 → P1 → P2 → P3; P4 and P7 are parallelizable; P5 → P6 sequential. Total: **~9–12 months** to a P6 verdict with P1–P4 results banked along the way.

---

## 6. Risks and contingencies

- **Flywheel stalls (P2).** First diagnose with the existing instruments: analyze_failures decomposes misses into *coverage* (right program never sampled → policy/exploration problem: more repair rounds, higher temperature, sketch-beam over AST skeletons, denser shaping via per-test + trace partial credit) vs. *ranking* (sampled but not selected → verifier problem: calibration, harder negatives, distinguisher tests). Curriculum re-banding second. Only after both: declare the negative result — which, with clean FLOP accounting, is the field's first real measurement of why pretraining is load-bearing.
- **Self-poisoning via near-winner distillation.** Default OFF; gate on verifier ECE; cap unverified fraction per batch; track label purity per generation.
- **dLLM RL instability (P2-onpolicy).** Stay on ExIt — it is sufficient for the thesis; the on-policy phase is an upgrade, not a dependency. Fallback order: TraceRL → coupled-GRPO → likelihood-free contrastive.
- **Length scaling (P5–P6).** Block-diffusion hybrid; hierarchical sketch-then-fill (two-pass diffusion: skeleton canvas, then body); function-level decomposition in the Python world.
- **Proposer degeneracy (P4).** Hard validity/novelty/entropy checks before pool entry; learnability-band reward; PAIRED-regret upgrade; hand ladder always retained as floor.
- **Reward hacking.** Detectors run every generation; incidents are quarantined from replay *and* written up — the controlled-hack-emergence record is a deliverable for the safety community, not an embarrassment.
- **Env throughput bottleneck.** Profile in P0; Rust/Cython interpreter port budgeted as a P2-era task if executions/sec caps the loop (self-play is where AlphaZero's compute went; plan for the same).
- **Toy-domain discount (reception risk).** Mitigations are structural: benchmark release (P1), transfer-framed method papers (P3), measurement-framed headline (Q1), discovery lottery ticket (Q5).
- **Compute envelope (order-of-magnitude).** M0 phases: 1–2 GPUs (consumer/A100-class), generations in hours. P5: 2–4 GPUs. P6: 4–8 GPUs for weeks. The program is deliberately runnable by a small team; if a result demands scale, that's a *finding* that justifies seeking it.

---

## 7. Deliverables and publication sequence

1. **P1 era — "MiniLeet" benchmark + verifier study.** The environment/ladder release plus the underdetermination measurement (51%) and the learned-verifier results (first-visible vs. learned vs. oracle). Establishes the instrument (Crafter/Procgen precedent: toy worlds become standards when they operationalize a question and are a pleasure to instrument).
2. **P2/P3 era — the flywheel paper (Q1 headline).** From-scratch diffusion coder via expert iteration; generation curves; capability-per-FLOP vs. honest enumeration and vs. the impure pretrained anchor; scaling grid. Framed as the unconfounded create-vs-elicit measurement.
3. **P3 — world-model ablation note (Q2)** and **repair-as-denoising methods paper (Q3)** — the latter explicitly demonstrated as transferable to pretrained dLLMs (the escape hatch from the toy ghetto).
4. **P4 — self-play in verifiable worlds (Q4):** learned proposer + adversarial distinguishing tests; the environment that hardens its own reward.
5. **P5/P6 — scale-up results;** any verified discovery (Q5) is opportunistically fast-tracked.
6. **Ongoing — reward-hacking observatory note** (safety workshop): catalogued hack emergences with full ground truth.
7. **Open source:** env + generators + harness + ladder (P1); training code (P2+); model checkpoints per phase.

---

## Appendix A — Spec language v2 (sketch, for P5/P7)

Current spec = signature + visible examples. V2 adds machine-checkable slots that NL statements carry and examples cannot: input constraints (`n ≤ 10^5`, value ranges), resource bounds (step-count class: O(n), O(n log n) — checked empirically via fitted growth), property assertions (sortedness, permutation-of-input, idempotence — checked by property tests the env generates), and tie-breaking rules. Every slot is *executable or checkable*; the parser's output is verifiable end-to-end, which is the entire argument for the discrete-interlingua bridge.

## Appendix B — Method references (what we're borrowing, and why)

| Method / work | What we take | Where |
|---|---|---|
| AlphaZero / Expert Iteration | Search-as-improvement-operator + distill on self-generated targets; "rules of the game" allowance | §3 loop |
| AlphaDev | Correctness-gated + efficiency-scored reward; latency axis → trace_len; discovery via suboptimal references | §3 reward, Q5 |
| MDLM / LLaDA / Dream | From-scratch masked-diffusion objective; confidence-based parallel unmasking | §2 |
| SDAR / block diffusion | Length-scaling contingency (AR-over-blocks, diffusion-within) | §2, §6 |
| TraceRL + diffusion value model (dLLM-RL) | Phase-2 on-policy recipe; validates the value-model pairing we already built | §3 |
| DiffuCoder / coupled-GRPO | Lower-variance dLLM logprob estimation (fallback 2) | §3 |
| DiffusionNFT | Likelihood-free positive/negative finetuning (fallback 3; graceful degradation to ExIt) | §3 |
| Code World Model (CWM) | Execution-trace training improves coding; here as from-scratch co-training (Q2) | §3 |
| Absolute Zero Reasoner | Propose-and-solve self-play with learnability reward — into our spec language, with validity checks | §3, P4 |
| PAIRED / UED / POET | Regret-based proposer upgrade path; open-endedness framing | P4 |
| Grammar-constrained from-scratch RL (CSG result) | Grammar encoding as the decisive ingredient absent pretraining → arity-typed T2 + grammar masks | §2 |
| Karel grammar+RL line | Held-out-test generalization reward design; cautionary: their RL still needed supervised init — our substitute is gen-0 BC on enumeration | §3 |
| STaR / rejection-sampling distillation | Train-on-verified-wins hygiene; near-winner caution | §3 |
| LLaVA / BLIP-2 / CLIP | NL-interface recipe: contrastive encoder pretraining + bridge; impurity quarantined at perception | P7 |
| InterCode | POMDP interactive-coding interface (code=action, execution feedback=observation); evidence that feedback rounds yield large gains even with naive retry; execution-result / state-diff reward recipes for effectful programs; containerized cheap-reset env template. *Anti-pattern for us:* raw-text observations (assumes pretrained readers) — our EVIDENCE encoding is designed, not piped | §2 evidence, §3 repair, P6 |