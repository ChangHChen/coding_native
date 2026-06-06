# MiniLeet Progress Log

Date: 2026-06-06

## Research Direction

MiniLeet is a controlled environment for testing whether programming behavior
can be learned from executable formal environments without language-model
pretraining.

Current framing:

```text
task spec + typed AST program + visible tests + execution traces
        -> predictor / policy / world model
        -> candidate selection or AST edits
        -> hidden-test generalization
```

The environment is custom, but follows standard program-synthesis evaluation
practice: typed DSL programs, input-output examples, executable tests, hidden
fuzz tests, fixed budgets, and enumerative/random baselines.

## Implemented

- Typed DSL with `int`, `bool`, `list[int]`, expressions, assignments, `if`,
  `for`, and `return`.
- Static type checker.
- Exact interpreter with execution traces.
- Visible/hidden test environment.
- Generated task suites:
  - `starter`
  - `generated`
  - `train`
  - `eval`
  - `procedural_train`
  - `procedural_eval`
- Baselines:
  - template baseline
  - deterministic enumerative typed-AST search
  - random shuffled typed-AST search
- JSONL trajectory collector.
- JSONL audit tool.
- Feature-based PyTorch reranker.
- Transformer sequence reranker.
- Shared-encoder structured reranker with run logging, plots, and sweep support.

## Current Data

Small generated split:

```text
train_enum.jsonl:
  rows/programs:        3,185
  tasks:                37
  visible executions:   15,925
  hidden executions:    203,840
  file size:            33 MB

eval_enum.jsonl:
  rows/programs:        991
  tasks:                13
  visible executions:   4,955
  hidden executions:    63,424
  file size:            9.8 MB
```

Medium procedural split:

```text
proc_train_250_enum.jsonl:
  rows/programs:        46,771
  tasks:                250
  visible executions:   233,855
  hidden executions:    1,496,672
  file size:            307 MB

proc_eval_60_enum.jsonl:
  rows/programs:        11,319
  tasks:                60
  visible executions:   56,595
  hidden executions:    362,208
  file size:            74 MB
```

Data files are ignored by git via `.gitignore`.

## Results

Small split reranker:

```text
Command:
  python3 -m minileet.rerank \
    --train data/train_enum.jsonl \
    --eval data/eval_enum.jsonl \
    --epochs 300 \
    --seed 0

Result:
  first_visible: selected=13/13 solved=9/13  mean_hidden=0.858 visible_overfit=4
  learned:       selected=13/13 solved=10/13 mean_hidden=0.941 visible_overfit=3
  oracle:        selected=13/13 solved=13/13 mean_hidden=1.000 visible_overfit=0
```

Medium procedural split reranker:

```text
Command:
  python3 -m minileet.rerank \
    --train data/proc_train_250_enum.jsonl \
    --eval data/proc_eval_60_enum.jsonl \
    --epochs 120 \
    --hidden 128 \
    --max-tokens 512 \
    --seed 0

Result:
  first_visible: selected=60/60 solved=21/60 mean_hidden=0.813 visible_overfit=39
  learned:       selected=60/60 solved=24/60 mean_hidden=0.875 visible_overfit=36
  oracle:        selected=60/60 solved=60/60 mean_hidden=1.000 visible_overfit=0
```

Medium procedural split Transformer reranker:

```text
Command:
  python3 -m minileet.transformer_rerank \
    --train data/proc_train_250_enum.jsonl \
    --eval data/proc_eval_60_enum.jsonl \
    --epochs 16 \
    --batch-size 256 \
    --max-len 512 \
    --max-vocab 4096 \
    --d-model 128 \
    --layers 2 \
    --heads 4 \
    --dropout 0.05 \
    --seed 1

Result:
  first_visible: selected=60/60 solved=21/60 mean_hidden=0.813 visible_overfit=39
  transformer:   selected=60/60 solved=26/60 mean_hidden=0.860 visible_overfit=34
  oracle:        selected=60/60 solved=60/60 mean_hidden=1.000 visible_overfit=0
```

Interpretation:

```text
The visible-only learned reranker improves held-out hidden-test generalization
over raw enumeration on both the small and medium splits. The Transformer
sequence reranker improves solved count beyond the feature MLP, but the feature
MLP still has better mean hidden pass rate on the current medium split.
```

This is an early but useful signal that visible execution traces and program
structure contain learnable information about hidden robustness.

## Current Limitations

- The learned model is still feature-based, not a real code-native neural
- The Transformer sequence model is implemented, but it is still a reranker,
  not an action-conditioned dynamics model or diffusion repair model.
- The DSL is narrow: mostly list scans, predicates, aggregations, and simple
  comparisons.
- The enumerator can express many current tasks, so the oracle-visible-gated
  score is artificially high.
- Current training is candidate reranking, not AST-edit RL.
- PyTorch is installed, but CUDA was not visible in the current shell.

## Next Plan

### 1. Neural Program/Trace Encoder

Replace hand-engineered features with a small neural encoder:

```text
task tokens + program tokens + visible execution traces
        -> embedding / sequence encoder
        -> hidden pass-rate prediction
        -> rerank visible-passing candidates
```

Initial architecture:

```text
token vocabulary from JSONL rows
embedding dim: 128
encoder: GRU or small Transformer
head: regression to hidden_pass_rate
loss: MSE
selection: highest predicted score among visible-passing candidates
```

Compare against:

```text
first_visible
feature_mlp
neural_encoder
oracle_visible_gated
```

Initial implementation:

```text
minileet.transformer_rerank
  task/program/visible-trace tokens
  small Transformer encoder
  hidden_pass_rate regression head
```

Current status:

```text
feature_mlp: solved=24/60, mean_hidden=0.875
transformer: solved=26/60, mean_hidden=0.860
```

The Transformer improves solved count but not mean hidden pass rate. The next
modeling change should be one of:

```text
shared encoder over task/program/test segments
register-token fusion
numeric feature connector
regression + solved-probability + pairwise ranking losses
```

Implemented:

```text
minileet.structured_rerank
  one shared BERT-style encoder from Hugging Face transformers
  task/program/per-test segments encoded separately with shared weights
  learned register tokens for fusion
  numeric connector
  hidden-pass, solved-probability, and ranking heads
  run-dir logging: config.json, history.jsonl, summary.json, loss.png

minileet.sweep_structured
  randomized hyperparameter trials
  per-trial run directories
  sweep.csv and best.json
```

Success criterion for the next version:

```text
The neural encoder beats the feature MLP on both solved count and mean hidden
pass rate on proc_eval_60.
```

### 2. Larger Procedural Data

If the neural encoder works, collect a larger split:

```text
train: 500-1,000 procedural tasks
eval: 100-200 procedural tasks
hidden-count: 32 or 64
```

Goal:

```text
100k-250k candidate rows first, then 1M+ rows if useful.
```

### 3. Stronger Baselines

Add:

```text
genetic programming
beam search over AST sketches
best-visible-pass heuristic
nearest-task/template retrieval baseline
```

### 4. Real World-Model Targets

Train predictors for:

```text
program + input -> output/error
program + visible tests -> pass vector
program + trace prefix -> next trace event
program + failed test -> likely repair family
```

### 5. Move Toward Agentic Editing

Introduce an edit-loop benchmark:

```text
start from empty or broken AST
agent chooses typed AST edit
environment runs visible tests
agent observes traces/errors
fixed edit/test budget
hidden tests used only for final evaluation
```

This is the first step toward the original non-LLM coding-world-model/RL idea.
