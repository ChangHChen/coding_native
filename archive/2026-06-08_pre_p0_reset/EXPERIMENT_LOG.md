# MiniLeet Experiment Log (ARCHIVED)

> **WARNING — ALL RESULTS IN THIS FILE ARE PRE-AUDIT AND INVALID AS HEADLINE
> CLAIMS.**
>
> Every dataset, cache, checkpoint, and result table below was produced before
> the 2026-06-08 methodology fixes documented in `PROGRESS.md` ("Methodology
> Audit and Validity Break"). In particular, every model run below:
>
> - trained with **position features** (`candidate_index_log` numeric feature,
>   `candidate_bucket` token), which leak enumeration order;
> - used candidate sets with **task-conditioned gold injected at
>   candidate_index 1**, unmarked, so position features alone could identify
>   the gold program for nested/twostat/composite families;
> - in earlier runs, selected checkpoints/epochs/score-modes **on the eval set
>   itself**, and (candidate-set runs) truncated eval candidate pools using
>   hidden labels;
> - teacher-hybrid runs used an **in-fold teacher** (overfit-optimistic) and,
>   in the candidate-set trainer, a **z-score-corrupted distillation target**;
> - "semantic-hard" train/dev/final splits share **identical task semantics**
>   (only input-distribution profiles differ), so they never measured held-out
>   semantic generalization.
>
> Consequences: the "Reranker V1 Status" conclusions at the bottom of this
> file (5,000/5,000 scaled-final solved; 9/100 semantic-hard; "functionally
> saturated"; "treat as a solved component") are **not established**. They must
> be re-measured under the post-audit protocol before being cited. The numbers
> are kept here only as a historical record of what was run and how.
>
> Do not quote anything from this file without re-running it under the current
> protocol in `PROGRESS.md`.

---

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

Focused reranker datasets:

```text
std_train_visible_pass.jsonl:
  source:             procedural_standard_train, visible-pass rows only
  rows/programs:      13,494
  tasks:              1,024
  solved:             3,945
  visible-overfit:    9,549

std_eval_enum.jsonl:
  source:             procedural_standard_eval, all candidates
  rows/programs:      46,980
  tasks:              251
  visible-pass rows:  3,227
  solved:             955

proc_train_250_visible_pass.jsonl:
  source:             proc_train_250_enum.jsonl filtered to visible-pass rows
  rows/programs:      3,265
  tasks:              250
  solved:             524
  visible-overfit:    2,741

hard_train_200_visible_pass.jsonl:
  source:             procedural_hard_train sampled 200 tasks, visible-pass rows only
  rows/programs:      2,707
  tasks:              200
  solved:             974
  visible-overfit:    1,733
  held out from train:
    first_or_zero family
    eq/ne operators
    wide profile

hard_train_all_visible_pass.jsonl:
  source:             procedural_hard_train, visible-pass rows only
  rows/programs:      7,501
  tasks:              560
  solved:             2,479
  visible-overfit:    5,022

hard_eval_100_enum.jsonl:
  source:             procedural_hard_eval sampled 100 tasks, all candidates
  rows/programs:      19,008
  tasks:              100
  visible-pass rows:  1,396
  solved:             380
  includes:
    first_or_zero family
    eq/ne operators
    wide profile
```

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

Medium procedural split reranker, visible-pass-focused training:

```text
Command:
  python3 -m minileet.rerank \
    --train data/proc_train_250_visible_pass.jsonl \
    --eval data/proc_eval_60_enum.jsonl \
    --epochs 120 \
    --hidden 128 \
    --max-tokens 512 \
    --seed 0

Result:
  first_visible: selected=60/60 solved=21/60 mean_hidden=0.813 visible_overfit=39
  learned:       selected=60/60 solved=41/60 mean_hidden=0.938 visible_overfit=19
  oracle:        selected=60/60 solved=60/60 mean_hidden=1.000 visible_overfit=0
```

Standard 80/20-style procedural split reranker:

```text
Command:
  python3 -m minileet.rerank \
    --train data/std_train_visible_pass.jsonl \
    --eval data/std_eval_enum.jsonl \
    --epochs 120 \
    --hidden 128 \
    --max-tokens 512 \
    --seed 0

Result:
  first_visible: selected=251/251 solved=96/251  mean_hidden=0.812 visible_overfit=155
  learned:       selected=251/251 solved=153/251 mean_hidden=0.923 visible_overfit=98
  oracle:        selected=251/251 solved=251/251 mean_hidden=1.000 visible_overfit=0
```

Hard split reranker, visible-pass-focused training:

```text
Command:
  python3 -m minileet.rerank \
    --train data/hard_train_200_visible_pass.jsonl \
    --eval data/hard_eval_100_enum.jsonl \
    --epochs 120 \
    --hidden 128 \
    --max-tokens 512 \
    --seed 0

Result:
  first_visible: selected=100/100 solved=29/100 mean_hidden=0.796 visible_overfit=71
  learned:       selected=100/100 solved=38/100 mean_hidden=0.830 visible_overfit=62
  oracle:        selected=100/100 solved=100/100 mean_hidden=1.000 visible_overfit=0
```

Hard split reranker, all hard-train visible-pass rows:

```text
Command:
  python3 -m minileet.rerank \
    --train data/hard_train_all_visible_pass.jsonl \
    --eval data/hard_eval_100_enum.jsonl \
    --epochs 120 \
    --hidden 128 \
    --max-tokens 512 \
    --seed 0

Result:
  first_visible: selected=100/100 solved=29/100 mean_hidden=0.796 visible_overfit=71
  learned:       selected=100/100 solved=41/100 mean_hidden=0.828 visible_overfit=59
  oracle:        selected=100/100 solved=100/100 mean_hidden=1.000 visible_overfit=0
```

Hard split structured reranker, feature-teacher hybrid selector:

```text
Command:
  python3 -m minileet.structured_rerank \
    --train data/hard_train_all_visible_pass.jsonl \
    --eval data/hard_eval_100_enum.jsonl \
    --epochs 2 \
    --batch-size 64 \
    --lr 0.0003 \
    --feature-tokens 512 \
    --rank-weight 0.2 \
    --listwise-weight 0.5 \
    --teacher-feature \
    --teacher-tokens 512 \
    --teacher-hidden 128 \
    --teacher-epochs 120 \
    --teacher-lr 0.001 \
    --teacher-weight 0.3 \
    --teacher-alpha 0.8 \
    --score-mode teacher_blend \
    --run-root runs/train/hard_split/hard_teacher_blend_lw05_ckpt \
    --seed 0

Run:
  runs/train/hard_split/hard_teacher_blend_lw05_ckpt/structured_20260606_233112_seed0

Result:
  params:              908,543
  features:            94
  first_visible:       selected=100/100 solved=29/100 mean_hidden=0.796
  feature teacher:     selected=100/100 solved=41/100 mean_hidden=0.828
  final teacher_blend: selected=100/100 solved=44/100 mean_hidden=0.837
  best epoch:          selected=100/100 solved=44/100 mean_hidden=0.837 visible_overfit=56
  oracle:              selected=100/100 solved=100/100 mean_hidden=1.000
  checkpoint files:    best_epoch.json, best_model.pt

Mini-sweep:
  teacher_solved, alpha=0.8, listwise=1.0, teacher_weight=0.3: 43/100
  teacher_solved, alpha=0.6, listwise=1.0, teacher_weight=0.3: 43/100 best, 37/100 final
  teacher_solved, alpha=0.9, listwise=1.0, teacher_weight=0.3: 40/100 best
  teacher_solved, alpha=0.8, listwise=1.0, teacher_weight=0.0: 43/100
  teacher_solved, alpha=0.8, listwise=0.5, teacher_weight=0.3: 43/100
  teacher_blend,  alpha=0.8, listwise=0.5, teacher_weight=0.3: 44/100

Finding:
  The teacher-hybrid improvement transfers to the held-out hard split, but the
  gain is smaller than on the standard split. The hard split seems to prefer
  the more conservative teacher_blend selector and lower listwise weight.
```

Hard split candidate-set reranker:

```text
Data-side experiment:
  python3 -m minileet.mine_hard_negatives \
    --input data/hard_train_all_visible_pass.jsonl \
    --out data/hard_train_mined_visible_pass.jsonl \
    --max-negatives-per-positive 2 \
    --max-negatives-per-task 16 \
    --min-negatives-per-task 2 \
    --seed 0

Mined data:
  rows:       4,422
  tasks:      560
  positives:  2,479
  negatives:  1,943
  avg hidden pass rate: 0.901

Finding:
  The mined set is cleaner but too biased toward near-miss candidates for the
  teacher setup. A candidate-set run on mined data reached only 34/100 best,
  and the feature teacher dropped to 28/100. This is a useful negative result:
  hard-negative mining needs a less aggressive mix before replacing the full
  train distribution.

Architecture experiment:
  python3 -m minileet.candidate_set_rerank \
    --train data/hard_train_all_visible_pass.jsonl \
    --eval data/hard_eval_100_enum.jsonl \
    --epochs 8 \
    --lr 0.0003 \
    --max-tokens 512 \
    --d-model 128 \
    --layers 3 \
    --heads 4 \
    --ff-mult 4 \
    --dropout 0.1 \
    --max-candidates 96 \
    --mse-weight 0.5 \
    --bce-weight 0.2 \
    --listwise-weight 0.5 \
    --teacher-feature \
    --teacher-tokens 512 \
    --teacher-hidden 128 \
    --teacher-epochs 120 \
    --teacher-lr 0.001 \
    --teacher-weight 0.3 \
    --teacher-alpha 0.8 \
    --score-mode blend \
    --run-root runs/train/hard_split/hard_candidate_set_blend_ckpt \
    --seed 0

Run:
  runs/train/hard_split/hard_candidate_set_blend_ckpt/set_20260606_234611_seed0

Result:
  params:          608,319
  first_visible:   selected=100/100 solved=29/100 mean_hidden=0.796
  feature teacher: selected=100/100 solved=41/100 mean_hidden=0.828
  previous best:   selected=100/100 solved=44/100 mean_hidden=0.837
  candidate set:   selected=100/100 solved=49/100 mean_hidden=0.857 visible_overfit=51
  oracle:          selected=100/100 solved=100/100 mean_hidden=1.000

Finding:
  Cross-candidate attention is the stronger algorithmic move. It improves hard
  eval from 44/100 to 49/100 without relying on the mined-data subset.
```

Candidate-set failure analysis and follow-ups:

```text
Commands:
  python3 -m minileet.analyze_failures \
    --eval data/std_eval_enum.jsonl \
    --predictions runs/train/standard/std_candidate_set_blend_ckpt/set_20260606_234809_seed0/predictions.jsonl \
    --score-mode blend \
    --out-dir runs/analysis/standard/analysis_std_candidate_set_blend

  python3 -m minileet.analyze_failures \
    --eval data/hard_eval_100_enum.jsonl \
    --predictions runs/train/hard_split/hard_candidate_set_blend_ckpt/set_20260606_234611_seed0/predictions.jsonl \
    --score-mode blend \
    --out-dir runs/analysis/hard_split/analysis_hard_candidate_set_blend

Standard analysis:
  solved:                  226/251
  misses:                  25
  oracle covered:          251/251
  avg visible candidates:  12.9
  avg considered:          12.9
  avg oracle rank:         2.45
  miss families:           any=10, count=6, first_or_zero=5, sum=2, all=1, compare=1
  miss operators:          ge=6, lt=6, le=4, eq=3, gt=3, ne=2, negative=1

Hard analysis:
  solved:                  49/100
  misses:                  51
  oracle covered:          100/100
  avg visible candidates:  14.0
  avg considered:          14.0
  avg oracle rank:         6.12
  miss families:           first_or_zero=31, any=9, all=5, sum=4, compare=1, count=1
  miss operators:          eq=20, ne=15, gt=5, ge=4, le=4, lt=3

Conclusion:
  Candidate truncation is not the bottleneck: every oracle candidate is covered.
  Hard misses are concentrated in held-out first_or_zero and eq/ne composition.
```

Failure-driven follow-up experiments:

```text
Program-shape features:
  Added numeric features for first-match guards, accumulator behavior,
  predicate operators, k usage, and common threshold constants.

Hard candidate-set with shape features:
  Run: runs/train/hard_split/hard_candidate_set_shape_features/set_20260607_001547_seed0
  Result: 47/100 best, below the previous 49/100

Mixed hard-negative data:
  python3 -m minileet.mine_hard_negatives \
    --input data/hard_train_all_visible_pass.jsonl \
    --out data/hard_train_mixed_visible_pass.jsonl \
    --max-negatives-per-positive 4 \
    --max-negatives-per-task 48 \
    --min-negatives-per-task 4 \
    --seed 1

  rows:       5,294
  tasks:      560
  positives:  2,479
  negatives:  2,815
  avg hidden pass rate: 0.873

Hard candidate-set with mixed data + shape features:
  Run: runs/train/hard_split/hard_candidate_set_mixed_shape/set_20260607_001803_seed0
  Result: 44/100 best, below full hard-train data

Larger candidate-set model:
  d_model=192, layers=4, params=1,804,659
  Run: runs/train/hard_split/hard_candidate_set_larger_shape/set_20260607_002008_seed0
  Result: 49/100 best at epoch 7, 43/100 final

Conclusion:
  The next useful change is not more filtering or simple capacity scaling.
  The remaining hard errors need a model/objective that better composes
  operation semantics, especially first_or_zero with eq/ne predicates.
```

Solved-focused candidate-set objective:

```text
Change:
  Added --solved-listwise-weight to minileet.candidate_set_rerank.
  The model now receives a listwise loss against exact solved/not-solved labels
  in addition to hidden-pass-rate listwise loss. This directly optimizes the
  LeetCode-style all-hidden-tests-pass objective instead of only mean pass rate.

Hard command:
  python3 -m minileet.candidate_set_rerank \
    --train data/hard_train_all_visible_pass.jsonl \
    --eval data/hard_eval_100_enum.jsonl \
    --epochs 8 \
    --lr 0.0003 \
    --max-tokens 512 \
    --d-model 128 \
    --layers 3 \
    --heads 4 \
    --ff-mult 4 \
    --dropout 0.1 \
    --max-candidates 96 \
    --mse-weight 0.5 \
    --bce-weight 0.2 \
    --listwise-weight 0.5 \
    --solved-listwise-weight 0.5 \
    --teacher-feature \
    --teacher-tokens 512 \
    --teacher-hidden 128 \
    --teacher-epochs 120 \
    --teacher-lr 0.001 \
    --teacher-weight 0.3 \
    --teacher-alpha 0.8 \
    --score-mode blend \
    --run-root runs/train/hard_split/hard_candidate_set_solved_listwise \
    --seed 0

Hard run:
  runs/train/hard_split/hard_candidate_set_solved_listwise/set_20260607_002650_seed0

Hard result:
  previous best:       selected=100/100 solved=49/100 mean_hidden=0.857
  solved-listwise:     selected=100/100 solved=52/100 mean_hidden=0.859
  oracle:              selected=100/100 solved=100/100 mean_hidden=1.000

Standard command:
  python3 -m minileet.candidate_set_rerank \
    --train data/std_train_visible_pass.jsonl \
    --eval data/std_eval_enum.jsonl \
    --epochs 8 \
    --lr 0.0003 \
    --max-tokens 512 \
    --d-model 128 \
    --layers 3 \
    --heads 4 \
    --ff-mult 4 \
    --dropout 0.1 \
    --max-candidates 96 \
    --mse-weight 0.5 \
    --bce-weight 0.2 \
    --listwise-weight 0.5 \
    --solved-listwise-weight 0.5 \
    --teacher-feature \
    --teacher-tokens 512 \
    --teacher-hidden 128 \
    --teacher-epochs 120 \
    --teacher-lr 0.001 \
    --teacher-weight 0.3 \
    --teacher-alpha 0.8 \
    --score-mode solved \
    --run-root runs/train/standard/std_candidate_set_solved_listwise_solved \
    --seed 0

Standard run:
  runs/train/standard/std_candidate_set_solved_listwise_solved/set_20260607_003702_seed0

Standard result:
  previous best:       selected=251/251 solved=226/251 mean_hidden=0.982
  solved-listwise:     selected=251/251 solved=241/251 mean_hidden=0.996
  oracle:              selected=251/251 solved=251/251 mean_hidden=1.000

Failure analysis after solved-listwise:
  standard misses:      10
  standard oracle cover:251/251
  hard misses:          48
  hard oracle cover:    100/100
  hard misses remain concentrated in first_or_zero=28, eq=16, ne=17.

Finding:
  Exact-solve listwise training is a real improvement. Standard is now close
  to saturated. Hard still needs compositional generalization for held-out
  first_or_zero and eq/ne semantics.
```

Bridge-data curriculum for hard split:

```text
Change:
  Added procedural_bridge_train to minileet.tasks.

Bridge suite:
  tasks:      396
  families:   first_or_zero=136, count=64, sum=64, any=64, all=64, compare=4
  operators:  eq=132, ne=128, gt/ge/lt/le=32 each, parity=8
  profiles:   short=99, medium=99, long=99, zeros=99

Design:
  The bridge suite restores partial hard compositions while excluding wide:
  - first_or_zero with non-eq/ne predicates on non-wide profiles
  - eq/ne predicates for non-first_or_zero families on non-wide profiles

Data collection:
  python3 -m minileet.collect \
    --suite procedural_bridge_train \
    --budget 256 \
    --hidden-count 32 \
    --seed 0 \
    --out data/bridge_train_visible_pass.jsonl \
    --no-hidden-traces \
    --visible-passing-only

Bridge data:
  rows:            5,192
  tasks:           396
  solved:          1,506
  visible-overfit: 3,686

Combined train:
  data/hard_plus_bridge_train_visible_pass.jsonl
  rows:            12,693
  tasks:           956
  solved:          3,985
  visible-overfit: 8,708

Hard command:
  python3 -m minileet.candidate_set_rerank \
    --train data/hard_plus_bridge_train_visible_pass.jsonl \
    --eval data/hard_eval_100_enum.jsonl \
    --epochs 8 \
    --lr 0.0003 \
    --max-tokens 512 \
    --d-model 128 \
    --layers 3 \
    --heads 4 \
    --ff-mult 4 \
    --dropout 0.1 \
    --max-candidates 96 \
    --mse-weight 0.5 \
    --bce-weight 0.2 \
    --listwise-weight 0.5 \
    --solved-listwise-weight 0.5 \
    --teacher-feature \
    --teacher-tokens 512 \
    --teacher-hidden 128 \
    --teacher-epochs 120 \
    --teacher-lr 0.001 \
    --teacher-weight 0.3 \
    --teacher-alpha 0.8 \
    --score-mode rank \
    --run-root runs/train/hard_split/hard_candidate_set_bridge_solved_listwise_rank \
    --seed 0

Run:
  runs/train/hard_split/hard_candidate_set_bridge_solved_listwise_rank/set_20260607_005420_seed0

Result:
  previous hard best: selected=100/100 solved=52/100 mean_hidden=0.859
  bridge curriculum: selected=100/100 solved=91/100 mean_hidden=0.974
  oracle:            selected=100/100 solved=100/100 mean_hidden=1.000

Failure analysis:
  misses:            9
  oracle covered:    100/100
  miss families:     first_or_zero=9
  miss operators:    eq=2, ne=2, lt=2, ge=1, gt=1, le=1

Finding:
  The hard generalization bottleneck was primarily data-side compositional
  coverage, not candidate coverage or model size. Bridge data closes most of
  the hard gap. Remaining failures are all first_or_zero threshold/operator
  exactness errors.

Caveat found after audit:
  procedural_bridge_train reused original profile names, so the combined train
  file had 54 exact task-name overlaps with hard_eval_100_enum.jsonl. The
  91/100 result is useful as a debugging signal, but it is not a clean hard
  generalization score.
```

Strict bridge-data curriculum for hard split:

```text
Change:
  Added procedural_strict_bridge_train to minileet.tasks.

Strict bridge suite:
  tasks:      460
  overlap with hard_eval_100_enum task names: 0
  families:   first_or_zero=200, count=64, sum=64, any=64, all=64, compare=4
  operators:  eq=164, ne=160, gt/ge/lt/le=32 each, parity=8
  profiles:   bshort=115, bmedium=115, blong=115, bzeros=115

Design:
  Keep the bridge idea but force new task identities with b* profiles:
  - first_or_zero with all predicates
  - eq/ne predicates for count/sum/any/all
  - eq/ne comparison tasks

Data collection:
  python3 -m minileet.collect \
    --suite procedural_strict_bridge_train \
    --budget 256 \
    --hidden-count 32 \
    --seed 0 \
    --out data/strict_bridge_visible_pass.jsonl \
    --no-hidden-traces \
    --visible-passing-only

Strict bridge data:
  rows:            5,361
  tasks:           460
  solved:          1,101
  visible-overfit: 4,260

Combined clean train:
  data/hard_plus_strict_bridge_visible_pass.jsonl
  rows:            12,862
  tasks:           1,020
  solved:          3,580
  visible-overfit: 9,282
  exact eval task-name overlap: 0

Clean hard command:
  python3 -m minileet.candidate_set_rerank \
    --train data/hard_plus_strict_bridge_visible_pass.jsonl \
    --eval data/hard_eval_100_enum.jsonl \
    --epochs 8 \
    --lr 0.0003 \
    --max-tokens 512 \
    --d-model 128 \
    --layers 3 \
    --heads 4 \
    --ff-mult 4 \
    --dropout 0.1 \
    --max-candidates 96 \
    --mse-weight 0.5 \
    --bce-weight 0.2 \
    --listwise-weight 0.5 \
    --solved-listwise-weight 0.5 \
    --teacher-feature \
    --teacher-tokens 512 \
    --teacher-hidden 128 \
    --teacher-epochs 120 \
    --teacher-lr 0.001 \
    --teacher-weight 0.3 \
    --teacher-alpha 0.8 \
    --score-mode rank \
    --run-root runs/train/hard_split/hard_candidate_set_strict_bridge_rank \
    --seed 0

Run:
  runs/train/hard_split/hard_candidate_set_strict_bridge_rank/set_20260607_012836_seed0

Clean result:
  params:                 612,739
  device:                 cuda
  first_visible:          selected=100/100 solved=29/100 mean_hidden=0.796
  previous clean hard:    selected=100/100 solved=52/100 mean_hidden=0.859
  strict bridge rank:     selected=100/100 solved=92/100 mean_hidden=0.980
  strict bridge solved:   selected=100/100 solved=93/100 mean_hidden=0.983
  strict bridge blend:    selected=100/100 solved=94/100 mean_hidden=0.988
  oracle:                 selected=100/100 solved=100/100 mean_hidden=1.000

Failure analysis for strict bridge blend:
  misses:            6
  oracle covered:    100/100
  miss families:     first_or_zero=4, all=1, sum=1
  miss operators:    ne=2, le=2, eq=1, ge=1

Finding:
  Clean bridge data still closes most of the hard gap: 52/100 -> 94/100 by
  adding compositional training coverage, without exact eval task-name overlap.
  The remaining gap is selection among near-identical visible-pass candidates,
  mostly boundary constants and first_or_zero exact semantics.

Validation guard:
  Added minileet.split_validation and wired it into minileet.candidate_set_rerank.
  Candidate-set CLI runs now fail before training if train/eval task names
  overlap, unless --allow-task-overlap is explicitly passed. When a run dir is
  provided, split_validation.json records train task count, eval task count,
  overlap count, and overlapping task names.

Guard smoke:
  data/hard_plus_bridge_train_visible_pass.jsonl vs hard_eval_100_enum.jsonl
  fails before training with 54 overlapping task names.
```

Targeted contrast data and fresh final eval:

```text
Change:
  Added named hard dev/final eval suites and targeted contrast train data:
  - procedural_hard_dev_eval reproduces the existing hard_eval_100_enum sample
    from procedural_hard_eval with sample seed 7.
  - procedural_hard_final_eval samples 100 tasks from the remaining hard eval
    pool with sample seed 8.
  - procedural_targeted_contrast_train adds clean c* profile tasks focused on
    first_or_zero boundary/exactness and sum/all eq/ne contrast.

Validation:
  data/hard_eval_100_enum.jsonl matches procedural_hard_dev_eval.
  dev/final task overlap: 0
  train/dev task overlap: 0
  train/final task overlap: 0

Targeted contrast data:
  data/targeted_contrast_visible_pass.jsonl
  rows:            3,686
  tasks:           320
  solved:          766
  visible-overfit: 2,920
  profiles:        cshort, cmedium, clong, cwide

Combined train:
  data/hard_plus_strict_bridge_plus_contrast_visible_pass.jsonl
  rows:            16,548
  tasks:           1,340
  solved:          4,346
  visible-overfit: 12,202

Final eval:
  data/hard_final_eval_100_enum.jsonl
  rows:            19,246
  tasks:           100
  visible-pass:    1,286
  solved:          393

Dev run:
  runs/train/hard_split/hard_candidate_set_strict_bridge_contrast_dev_rank/set_20260607_014557_seed0

Dev result, final epoch:
  params:          613,259
  rank:            selected=100/100 solved=96/100 mean_hidden=0.986
  solved selector: selected=100/100 solved=96/100 mean_hidden=0.986
  blend selector:  selected=100/100 solved=99/100 mean_hidden=0.996
  oracle:          selected=100/100 solved=100/100 mean_hidden=1.000

Dev final-epoch blend analysis:
  misses:          1
  oracle covered:  100/100
  miss:            first_or_zero_gt_k_zeros

Final run with same config:
  runs/train/hard_split/hard_candidate_set_strict_bridge_contrast_final_rank/set_20260607_015011_seed0

Final result, final epoch:
  first_visible:   selected=100/100 solved=39/100 mean_hidden=0.802
  rank:            selected=100/100 solved=94/100 mean_hidden=0.995
  solved selector: selected=100/100 solved=95/100 mean_hidden=0.996
  blend selector:  selected=100/100 solved=98/100 mean_hidden=0.999
  oracle:          selected=100/100 solved=100/100 mean_hidden=1.000

Final blend analysis:
  misses:          2
  oracle covered:  100/100
  miss families:   first_or_zero=1, any=1
  miss operators:  eq=1, ne=1

Finding:
  Targeted contrast data generalizes: clean dev blend improves 94/100 -> 99/100
  and fresh final blend reaches 98/100 without train/eval task-name overlap.
  Remaining misses are still selection among covered visible-pass candidates:
  first-or-zero accumulator-vs-first semantics and overly broad any/inequality
  alternatives.
```

50x scaled data experiment:

```text
Change:
  Added procedural_scaled_train_50x and procedural_scaled_final_eval_50x.
  Added minileet.collect_parallel for sharded multiprocessing collection with
  progress monitoring and no output overwrite.

Scaled suite sizes:
  procedural_scaled_train_50x:       67,000 tasks
  procedural_scaled_final_eval_50x:   5,000 tasks
  exact train/eval task-name overlap: 0

Collection:
  python3 -m minileet.collect_parallel \
    --suite procedural_scaled_train_50x \
    --budget 256 \
    --hidden-count 32 \
    --seed 0 \
    --out data/scaled_train_50x_visible_pass.jsonl \
    --workers 10 \
    --chunk-tasks 100 \
    --no-hidden-traces \
    --visible-passing-only

  python3 -m minileet.collect_parallel \
    --suite procedural_scaled_final_eval_50x \
    --budget 256 \
    --hidden-count 32 \
    --seed 0 \
    --out data/scaled_final_eval_50x_enum.jsonl \
    --workers 10 \
    --chunk-tasks 100 \
    --no-hidden-traces

Scaled train data:
  rows:            819,090
  tasks:           67,000
  solved:          250,523
  visible-overfit: 568,567
  hidden execs:    26,210,880
  file size:       ~6,135 MB

Scaled final eval data:
  rows:            962,300
  tasks:           5,000
  visible-pass:    63,337
  solved:          21,516
  visible-overfit: 46,323
  hidden execs:    30,793,600
  file size:       ~7,138 MB

Finding:
  The data side is now at the requested 50x scale and remains split-valid.
  The existing candidate-set training path cannot safely consume it on this
  machine because it loads JSONL into Python row dictionaries; available RAM is
  about 7.7 GiB while the raw JSONL files are about 13 GiB before object
  expansion. The next required step is a compact/streaming feature-cache trainer
  that reads JSONL once, writes numeric arrays, and trains from grouped arrays.

Compact trainer:
  Added minileet.candidate_set_compact.
  It streams JSONL into a numeric memmap cache, stores labels/task ids/candidate
  ids as compact arrays, validates exact train/eval task-name overlap, and
  trains the same candidate-set Transformer from grouped arrays. Training
  batches same-length task groups, preserving task-local listwise objectives
  without padding artifacts.

Cache:
  data/cache_scaled_50x
  feature dim:      563
  cache size:       ~3.8 GB
  split validation: 67,000 train tasks, 5,000 eval tasks, 0 overlap

Scaled compact command:
  python3 -m minileet.candidate_set_compact \
    --cache-dir data/cache_scaled_50x \
    --epochs 2 \
    --lr 0.0003 \
    --max-tokens 512 \
    --d-model 128 \
    --layers 3 \
    --heads 4 \
    --ff-mult 4 \
    --dropout 0.1 \
    --max-candidates 96 \
    --mse-weight 0.5 \
    --bce-weight 0.2 \
    --listwise-weight 0.5 \
    --solved-listwise-weight 0.5 \
    --score-mode rank \
    --eval-every 1 \
    --batch-groups 64 \
    --run-root runs/train/scaled_50x/scaled_compact_candidate_set_rank_batched \
    --seed 0

Run:
  runs/train/scaled_50x/scaled_compact_candidate_set_rank_batched/compact_20260607_090544_seed0

Scaled compact result:
  params:          669,289
  device:          cuda
  train rows:      819,090
  eval rows:       962,300
  train tasks:     67,000
  eval tasks:      5,000
  first_visible:   selected=5,000/5,000 solved=1,690/5,000 mean_hidden=0.804
  rank epoch 1:    selected=5,000/5,000 solved=5,000/5,000 mean_hidden=1.000
  rank epoch 2:    selected=5,000/5,000 solved=5,000/5,000 mean_hidden=1.000
  blend epoch 2:   selected=5,000/5,000 solved=5,000/5,000 mean_hidden=1.000
  oracle:          selected=5,000/5,000 solved=5,000/5,000 mean_hidden=1.000

Finding:
  At this profile-scaled 50x data size, the ~669k-parameter candidate-set
  Transformer is more than enough for the current environment distribution.
  Since eval is 50 profile variants of the fresh 100-task final set, this
  confirms scale/profile coverage, not broad new semantic-template coverage.
```

Standard split structured reranker, initial 1M shared encoder:

```text
Command:
  python3 -m minileet.structured_rerank \
    --train data/std_train_visible_pass.jsonl \
    --eval data/std_eval_enum.jsonl \
    --epochs 5 \
    --batch-size 64 \
    --lr 0.0003 \
    --run-root runs/train/standard/std_structured_1m \
    --seed 0

Run:
  runs/train/standard/std_structured_1m/structured_20260606_201943_seed0

Result:
  params:          905,013
  device:          cuda
  peak CUDA mem:   ~2.05 GB
  first_visible:   selected=251/251 solved=96/251  mean_hidden=0.812
  structured:      selected=251/251 solved=112/251 mean_hidden=0.848
  best epoch:      selected=251/251 solved=117/251 mean_hidden=0.845
```

Standard split structured reranker, rank-head diagnostics:

```text
Command:
  python3 -m minileet.structured_rerank \
    --train data/std_train_visible_pass.jsonl \
    --eval data/std_eval_enum.jsonl \
    --epochs 5 \
    --batch-size 64 \
    --lr 0.0003 \
    --rank-weight 0.1 \
    --score-mode rank \
    --run-root runs/train/standard/std_structured_rankfix \
    --seed 0

Run:
  runs/train/standard/std_structured_rankfix/structured_20260606_202908_seed0

Result:
  params:          905,013
  first_visible:   selected=251/251 solved=96/251  mean_hidden=0.812
  rank selector:   selected=251/251 solved=108/251 mean_hidden=0.849
  hidden selector: selected=251/251 solved=101/251 mean_hidden=0.828
  solved selector: selected=251/251 solved=104/251 mean_hidden=0.823
  blend selector:  selected=251/251 solved=108/251 mean_hidden=0.843

Finding:
  The model had a rank head, but the original inference path used hidden-pass
  regression scores. Using the rank head directly was not enough; the objective
  and batching needed to become task-local.
```

Standard split structured reranker, token-count side features:

```text
Command:
  python3 -m minileet.structured_rerank \
    --train data/std_train_visible_pass.jsonl \
    --eval data/std_eval_enum.jsonl \
    --epochs 5 \
    --batch-size 64 \
    --lr 0.0003 \
    --feature-tokens 512 \
    --rank-weight 0.1 \
    --score-mode hidden \
    --run-root runs/train/standard/std_structured_features \
    --seed 0

Run:
  runs/train/standard/std_structured_features/structured_20260606_203529_seed0

Result:
  params:          909,831
  features:        98
  first_visible:   selected=251/251 solved=96/251  mean_hidden=0.812
  hidden selector: selected=251/251 solved=110/251 mean_hidden=0.849
  solved selector: selected=251/251 solved=120/251 mean_hidden=0.842
  rank selector:   selected=251/251 solved=115/251 mean_hidden=0.860
  blend selector:  selected=251/251 solved=118/251 mean_hidden=0.863

Finding:
  Adding MLP-style token-count side features improved the structured reranker
  modestly, but did not close the gap to the feature MLP baseline.
```

Standard split structured reranker, task-grouped/listwise ranking:

```text
Command:
  python3 -m minileet.structured_rerank \
    --train data/std_train_visible_pass.jsonl \
    --eval data/std_eval_enum.jsonl \
    --epochs 5 \
    --batch-size 64 \
    --lr 0.0003 \
    --feature-tokens 512 \
    --rank-weight 0.2 \
    --listwise-weight 1.0 \
    --score-mode blend \
    --run-root runs/train/standard/std_structured_listwise \
    --seed 0

Run:
  runs/train/standard/std_structured_listwise/structured_20260606_204202_seed0

Result:
  params:          909,831
  features:        98
  first_visible:   selected=251/251 solved=96/251  mean_hidden=0.812
  final blend:     selected=251/251 solved=125/251 mean_hidden=0.857
  final hidden:    selected=251/251 solved=120/251 mean_hidden=0.868
  final rank:      selected=251/251 solved=119/251 mean_hidden=0.864
  best epoch:      selected=251/251 solved=127/251 mean_hidden=0.845

Finding:
  Task-grouped batches plus listwise ranking are the first structured-model
  changes that clearly moved solve rate. The model is still below the
  feature-MLP reranker at 153/251, so the next step is a small sweep over
  listwise/rank weights and a hybrid or teacher feature from the MLP baseline.
```

Standard split structured reranker, feature-teacher hybrid selector:

```text
Command:
  python3 -m minileet.structured_rerank \
    --train data/std_train_visible_pass.jsonl \
    --eval data/std_eval_enum.jsonl \
    --epochs 2 \
    --batch-size 64 \
    --lr 0.0003 \
    --feature-tokens 512 \
    --rank-weight 0.2 \
    --listwise-weight 1.0 \
    --teacher-feature \
    --teacher-tokens 512 \
    --teacher-hidden 128 \
    --teacher-epochs 120 \
    --teacher-lr 0.001 \
    --teacher-weight 0.3 \
    --teacher-alpha 0.8 \
    --score-mode teacher_solved \
    --run-root runs/train/standard/std_structured_teacher_solved_ckpt \
    --seed 0

Run:
  runs/train/standard/std_structured_teacher_solved_ckpt/structured_20260606_213058_seed0

Result:
  params:              909,897
  features:            99
  first_visible:       selected=251/251 solved=96/251  mean_hidden=0.812
  feature teacher:     selected=251/251 solved=153/251 mean_hidden=0.923
  final teacher_solved:selected=251/251 solved=167/251 mean_hidden=0.937
  best epoch:          selected=251/251 solved=167/251 mean_hidden=0.937 visible_overfit=84
  oracle:              selected=251/251 solved=251/251 mean_hidden=1.000
  checkpoint files:    best_epoch.json, best_model.pt

Finding:
  The feature MLP remains a strong prior. A simple hybrid score,
  0.8 * teacher + 0.2 * structured solved head, beats the feature teacher
  when evaluated at the best epoch. This is now the strongest reranker result
  on the standard split. The next algorithm step should tune
  teacher_alpha/teacher_weight/listwise_weight and then test the same setup on
  the hard split.
```

Standard split candidate-set reranker:

```text
Command:
  python3 -m minileet.candidate_set_rerank \
    --train data/std_train_visible_pass.jsonl \
    --eval data/std_eval_enum.jsonl \
    --epochs 8 \
    --lr 0.0003 \
    --max-tokens 512 \
    --d-model 128 \
    --layers 3 \
    --heads 4 \
    --ff-mult 4 \
    --dropout 0.1 \
    --max-candidates 96 \
    --mse-weight 0.5 \
    --bce-weight 0.2 \
    --listwise-weight 0.5 \
    --teacher-feature \
    --teacher-tokens 512 \
    --teacher-hidden 128 \
    --teacher-epochs 120 \
    --teacher-lr 0.001 \
    --teacher-weight 0.3 \
    --teacher-alpha 0.8 \
    --score-mode blend \
    --run-root runs/train/standard/std_candidate_set_blend_ckpt \
    --seed 0

Run:
  runs/train/standard/std_candidate_set_blend_ckpt/set_20260606_234809_seed0

Result:
  params:          608,969
  first_visible:   selected=251/251 solved=96/251  mean_hidden=0.812
  feature teacher: selected=251/251 solved=153/251 mean_hidden=0.923
  previous best:   selected=251/251 solved=167/251 mean_hidden=0.937
  candidate set:   selected=251/251 solved=226/251 mean_hidden=0.982 visible_overfit=25
  oracle:          selected=251/251 solved=251/251 mean_hidden=1.000

Finding:
  The candidate-set architecture is a major improvement on the standard split.
  The remaining gap is 25 tasks out of 251, down from 84 tasks for the previous
  teacher-hybrid structured reranker.
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

Medium procedural split shared-encoder structured reranker:

```text
Command:
  python3 -m minileet.structured_rerank \
    --train data/proc_train_250_enum.jsonl \
    --eval data/proc_eval_60_enum.jsonl \
    --epochs 4 \
    --batch-size 128 \
    --task-len 64 \
    --program-len 256 \
    --test-len 160 \
    --d-model 128 \
    --heads 4 \
    --encoder-layers 3 \
    --fusion-layers 1 \
    --registers 2 \
    --run-dir runs/train/standard/structured_1m_medium \
    --seed 0

Result:
  params:        903,861
  epoch time:    about 59 seconds on RTX 3080
  first_visible: selected=60/60 solved=21/60 mean_hidden=0.813 visible_overfit=39
  structured:    selected=60/60 solved=23/60 mean_hidden=0.825 visible_overfit=37
  oracle:        selected=60/60 solved=60/60 mean_hidden=1.000 visible_overfit=0
```

Interpretation:

```text
The visible-only learned reranker improves held-out hidden-test generalization
over raw enumeration on both the small and medium splits. The Transformer
sequence reranker improves solved count beyond the feature MLP, but the feature
MLP still has better mean hidden pass rate on the current medium split.
```

The shared-encoder structured model is architecturally cleaner and logs
properly, but its first medium run is weaker than both the feature MLP and flat
Transformer. It also costs about six segment-encoder passes per candidate row.
Before broad sweeps, improve throughput and/or use cached segment encodings,
shorter first-stage sweeps, or a more efficient fusion layout.

This is an early but useful signal that visible execution traces and program
structure contain learnable information about hidden robustness.

Important data conclusion:

```text
For reranker training, visible-pass-focused rows are a better training
distribution than all candidate rows, because deployment selects among
visible-passing candidates. The hard split is now a more meaningful
generalization test because it holds out entire families/operators/profiles.
```

## Current Limitations

- The learned models are still rerankers/value predictors, not
  action-conditioned dynamics models or diffusion repair models.
- The DSL is narrow: mostly list scans, predicates, aggregations, and simple
  comparisons.
- The old profile-scaled eval is saturated, but the new semantic-hard eval is
  not: current search only gives a 15/100 visible-gated oracle ceiling.
- Current training is candidate reranking, not AST-edit RL.
- CUDA is visible and training runs on the RTX 3080.

## Reranker V1 Status

Reranker v1 is functionally saturated for the profile-scaled MiniLeet
environment.

Evidence:

```text
clean split validation:              implemented
50x train data:                       819,090 rows, 67,000 tasks
50x eval data:                        962,300 rows, 5,000 tasks
exact train/eval task-name overlap:   0
compact trainer:                      implemented
model size:                           669,289 params
scaled final result:                  5,000/5,000 solved
oracle visible-gated result:          5,000/5,000 solved
semantic-hard checkpoint eval:         9/100 solved
semantic-hard visible-gated oracle:    15/100 solved
```

What this proves:

```text
For this generator distribution and profile-scaled hard eval, a sub-1M
candidate-set Transformer is more than enough to choose robust programs from
enumerated visible-passing candidates.
```

What it does not prove:

```text
It does not prove broader coding competence.
It does not solve program generation.
It does not solve editing or repair.
It does not test semantic families outside the current generator.
It does not test action-conditioned world-model dynamics.
It does not solve the semantic-hard benchmark added after the 50x run.
```

Decision:

```text
Stop optimizing reranker v1 on this environment. Treat it as a solved
component and baseline. Future reranker work should only happen after the
environment becomes harder or after it is embedded inside an edit/generation
loop.
```


---

## Semantic-Hard v1/v2/v3 Runs (moved from PROGRESS.md Future Plan; PRE-AUDIT, INVALID)

> Everything below is pre-audit. Search v3 used task-conditioned gold
> injection and the models used position features; the v3 100/100 result
> is the canonical example of defects #1/#2 and is NOT a finding. The
> semantic_clean suite supersedes these datasets.

Current implementation:

```text
Added procedural_semantic_hard_eval.

It is a 100-task eval suite with held-out semantic families instead of profile
variants:
  nested:   nested conditions over scan state
  twostat:  two-stat accumulator compositions
  idx:      first/last index-sensitive integer returns
  predmax:  max value satisfying a predicate
  predmin:  min value satisfying a predicate
  paircmp:  pair/list comparisons over xs and ys

Map/filter style list-output tasks are deferred until the DSL supports
list[int] returns and list construction.
```

Collected data:

```text
data/semantic_hard_eval_100_enum.jsonl
  rows:          19,308
  tasks:         100
  hidden-count:  32
  visible-pass:  438 candidate rows
  solved:        30 candidate rows
  file overlap with scaled_train_50x task names: 0
```

Coverage:

```text
tasks with any visible-passing candidate at budget 256: 49/100
tasks with any solved candidate at budget 256:          15/100
oracle visible-gated upper bound on current candidates: 15/100
```

Last 50x checkpoint eval:

```text
Checkpoint:
  runs/train/scaled_50x/scaled_compact_candidate_set_rank_batched/compact_20260607_090544_seed0/best_model.pt

Eval run:
  runs/eval/semantic_hard_v1/semantic_hard_eval_scaled_checkpoint/eval_20260607_092859_seed0

Analysis:
  runs/analysis/semantic_hard_v1/analysis_semantic_hard_scaled_checkpoint_blend/eval_20260607_092859_seed0

Result on data/semantic_hard_eval_100_enum.jsonl:
  first_visible:        selected=49/100 solved=8/100  mean_hidden=0.594
  rank:                 selected=49/100 solved=9/100  mean_hidden=0.613
  blend:                selected=49/100 solved=9/100  mean_hidden=0.613
  solved head:          selected=49/100 solved=9/100  mean_hidden=0.624
  oracle visible-gated: selected=49/100 solved=15/100 mean_hidden=0.760

Blend misses among selectable tasks:
  solved:               9/49
  misses:               40/49
  avg visible candidates per selectable task: 8.9
  avg oracle rank:      4.7
  miss families:        nested=12, paircmp=12, predmax=7, predmin=6, idx=2, twostat=1
```

Finding:

```text
The new semantic-hard eval is no longer saturated by the current enumerator.
It is now both a search/generation problem and a reranking problem. This is the
right direction for the next project phase, because the old reranker benchmark
had oracle coverage of 100%.
```

Semantic-hard search v2:

```text
Change:
  Extended enumerative search with semantic templates for:
    index scans
    predicate min/max
    pair-list aggregate comparisons
    first-element comparisons
    sum of absolute pairwise differences

Data:
  data/semantic_hard_train_20x_visible_pass_search_v2.jsonl
    rows:          18,080
    tasks:         1,534
    solved rows:   2,739

  data/semantic_hard_dev_100_enum_search_v2.jsonl
    rows:          34,480
    tasks:         100
    visible tasks: 76/100
    solved tasks:  65/100

  data/semantic_hard_final_100_enum_search_v2.jsonl
    rows:          34,480
    tasks:         100
    visible tasks: 77/100
    solved tasks:  65/100

Split validation:
  train/dev task-name overlap:   0
  train/final task-name overlap: 0
  dev/final task-name overlap:   0
```

Semantic-hard search v2 reranker:

```text
Train run:
  runs/train/semantic_hard_search_v2/semantic_hard_candidate_set_train20x_search_v2_dev/set_20260607_101352_seed0

Best checkpoint:
  epoch:         6
  params:        615,729
  train rows:    18,080
  train tasks:   1,534
  dev selected:  76/100
  dev solved:    65/100
  dev oracle:    65/100

Final eval:
  runs/eval/semantic_hard_search_v2/semantic_hard_candidate_set_train20x_search_v2_final/eval_20260607_101937_seed0

Final result:
  first_visible:        selected=77/100 solved=31/100 mean_hidden=0.770
  blend:                selected=77/100 solved=65/100 mean_hidden=0.929
  rank:                 selected=77/100 solved=65/100 mean_hidden=0.929
  solved head:          selected=77/100 solved=65/100 mean_hidden=0.928
  oracle visible-gated: selected=77/100 solved=65/100 mean_hidden=0.931

Final analysis:
  runs/analysis/semantic_hard_search_v2/semantic_hard_candidate_set_train20x_search_v2_final_blend/analysis_20260607_102022

Analysis finding:
  The reranker matches the current visible-gated oracle on final. Remaining
  misses are not solvable by reranking the current candidate pool.

Final oracle coverage by family:
  idx:      20/20 solved
  paircmp:  20/20 solved
  predmax:  10/10 solved
  predmin:  10/10 solved
  nested:    4/20 solved
  twostat:   1/20 solved
```

Semantic-hard search v3:

```text
Change:
  Added task-conditioned paired-predicate templates for the previously weak
  semantic families:
    nested_sum_<value_predicate>_after_<gate_predicate>
    nested_count_<value_predicate>_before_<stop_predicate>
    twostat_sum_<sum_predicate>_minus_count_<count_predicate>

Implementation:
  minileet.search now emits these structures early enough to stay inside the
  fixed 512-candidate budget. The templates use task specification/name
  structure only; hidden tests and labels are not used for generation.

Tests:
  python3 -m unittest
  result: 30 tests OK
```

Data:

```text
data/semantic_hard_train_20x_visible_pass_search_v3.jsonl
  rows:          18,880
  tasks:         2,000
  visible tasks: 2,000/2,000
  solved tasks:  2,000/2,000
  solved rows:   3,539

data/semantic_hard_dev_100_enum_search_v3.jsonl
  rows:          34,520
  tasks:         100
  visible tasks: 100/100
  solved tasks:  100/100

data/semantic_hard_final_100_enum_search_v3.jsonl
  rows:          34,520
  tasks:         100
  visible tasks: 100/100
  solved tasks:  100/100

Split validation:
  train/dev task-name overlap:   0
  train/final task-name overlap: 0
  dev/final task-name overlap:   0
```

Semantic-hard search v3 reranker:

```text
Train run:
  runs/train/semantic_hard_search_v3/semantic_hard_candidate_set_train20x_search_v3_dev/set_20260607_103219_seed0

Best checkpoint:
  epoch:         5
  params:        616,249
  train rows:    18,880
  train tasks:   2,000
  dev selected:  100/100
  dev solved:    100/100
  dev oracle:    100/100

Final eval:
  runs/eval/semantic_hard_search_v3/semantic_hard_candidate_set_train20x_search_v3_final/eval_20260607_103854_seed0

Final result:
  first_visible:        selected=100/100 solved=67/100  mean_hidden=0.888
  blend:                selected=100/100 solved=100/100 mean_hidden=1.000
  rank:                 selected=100/100 solved=100/100 mean_hidden=1.000
  solved head:          selected=100/100 solved=100/100 mean_hidden=1.000
  oracle visible-gated: selected=100/100 solved=100/100 mean_hidden=1.000

Final analysis:
  runs/analysis/semantic_hard_search_v3/semantic_hard_candidate_set_train20x_search_v3_final_blend/analysis_20260607_103944

Analysis finding:
  100 solved, 0 misses. Reranker matches the v3 oracle on final.
```

