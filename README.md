# MiniLeet-v0

MiniLeet-v0 is a small programming-world scaffold for testing non-LLM coding agents.
The environment uses typed AST programs, an exact interpreter, visible examples,
hidden fuzz tests, execution traces, rewards, and simple baselines.

The first research question is narrow:

> Can an agent learn programming behavior from executable environment interaction
> over structured program states, without language-model pretraining?

This repository currently contains the environment and a non-learning template
baseline. The next step is to add model-free RL and world-model training on top
of the same API.

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
python3 -m minileet.rerank --train data/proc_train_250_enum.jsonl --eval data/proc_eval_60_enum.jsonl --epochs 120 --hidden 128 --max-tokens 512
python3 -m minileet.transformer_rerank --train data/proc_train_250_enum.jsonl --eval data/proc_eval_60_enum.jsonl --epochs 16 --batch-size 256 --max-len 512 --dropout 0.05 --seed 1
python3 -m minileet.structured_rerank --train data/proc_train_250_enum.jsonl --eval data/proc_eval_60_enum.jsonl --epochs 8 --run-dir runs/structured_1m
python3 -m minileet.sweep_structured --train data/proc_train_250_enum.jsonl --eval data/proc_eval_60_enum.jsonl --out-dir sweeps/structured_1m --trials 8 --epochs 4
python3 -m unittest
```

## Current Scope

The DSL supports:

- `int`, `bool`, and `list[int]`
- integer constants and variables
- list length and indexing
- arithmetic and comparisons
- assignment, return, `if`, and `for item in list`

The task API exposes:

- function signatures
- visible examples
- hidden tests generated from deterministic samplers
- an oracle used only by the evaluator
- pass rate and reward under a fixed candidate budget

Baselines:

- `template`: a small hand-written baseline used for smoke checks
- `enumerative`: deterministic typed-AST program enumeration
- `random`: shuffled typed-AST enumeration under the same candidate budget

Baselines choose candidates using visible tests only. Hidden tests are evaluated
only for the selected candidate so overfitting to visible examples is visible in
the reported hidden pass rate.

The first learned baseline is `minileet.rerank`, a small PyTorch MLP that learns
to predict hidden pass rate from task/program/visible-trace features and selects
among visible-passing candidates.

`minileet.transformer_rerank` is the first neural sequence baseline. It tokenizes
task specs, rendered programs, visible test outcomes, and traces, then trains a
small Transformer encoder to predict hidden pass rate.

`minileet.structured_rerank` is a ~1M parameter shared-encoder reranker built
with Hugging Face `transformers`: one BERT-style encoder is reused for task,
program, and visible-test trace segments, followed by register-token fusion,
numeric feature connection, regression/BCE/ranking heads, per-epoch monitoring,
JSONL history, summary JSON, and loss plots.

Task suites:

- `starter`: 5 hand-written smoke-test tasks
- `generated`: procedural semantic-family tasks
- `train`: deterministic generated train split
- `eval`: deterministic generated held-out split
- `all`: starter plus generated
- `procedural_train`: medium-scale generated train split
- `procedural_eval`: medium-scale generated held-out split

The interpreter records traces for:

- assignments
- branch decisions
- loop iterations
- returns
- runtime errors

## Intended Experiment Path

1. Keep the benchmark tiny and deterministic.
2. Add stronger dumb baselines, including genetic programming.
3. Add a model-free RL agent over AST edit actions.
4. Add self-supervised world-model losses from interpreter traces.
5. Compare world model + planning/RL against the baselines.

See [PROGRESS.md](PROGRESS.md) for current results, collected data sizes, and
the next experimental plan.
