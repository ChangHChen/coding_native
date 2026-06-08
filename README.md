# ZeroCoder P0 Substrate

This repo has been reset to a P0-only build.

The previous workspace was archived at:

```text
archive/2026-06-08_pre_p0_reset/
```

P0 is the environment-hardening substrate for ZeroCoder. Nothing in this root
is a model, reranker, cache, or result log. The only goal here is to build and
prove the pieces needed before P1 conditional BC.

## P0 Build Items

- typed DSL, typechecker, and strict interpreter
- exact grading with `bool` distinct from `int`
- task validation and suite hygiene checks
- honest enumeration filtering
- exploit tests
- hack detectors
- T2 AST tokenizer
- T3 trace tokenizer
- evidence builder
- serialization schema audit
- throughput benchmark

## Run

```bash
python3 -m unittest discover -s tests
python3 -m minileet.bench
```

P1 is blocked until these tests and proof commands are green and their outputs
are logged.
