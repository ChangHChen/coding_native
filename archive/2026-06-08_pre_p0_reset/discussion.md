# Programming Model Idea

**User:** Anonymous  
**Created:** 6/6/2026 0:26:48  
**Updated:** 6/6/2026 0:52:37  
**Exported:** 6/6/2026 16:16:09  
**Link:** [https://chatgpt.com/c/6a22eab7-e2f4-83a3-80f7-e348443c9aec](https://chatgpt.com/c/6a22eab7-e2f4-83a3-80f7-e348443c9aec)  

## Prompt:
6/6/2026, 12:26:48 AM

I'm thinking a programing/coding native model, using diffusion, and pure RL (interaction with the digital environment, the interoretator, the feedback, etc.) like world models (which are hot currently)
I don't know if this is a good idea or just a dream

## Response:
6/6/2026, 12:26:49 AM

It is **not just a dream**. The good version of your idea is very plausible, and parts of it are already happening. But I would slightly reframe it:

**Bad version:** “Train a coding model from zero using only pure RL and interpreter feedback.”

**Good version:** “Build a coding-native model pretrained on code, docs, execution traces, tests, failures, patches, and environment interactions; then improve it with verifiable RL in sandboxes; optionally use diffusion for program editing, repair, and search.”

That second version is a serious research direction.

A very close example already exists: Meta’s **Code World Model / CWM**, a 32B open-weights coding model trained not only on static code, but also on observation-action trajectories from Python interpreters and Docker-like agent environments, followed by RL in verifiable coding, math, and software-engineering environments. Its paper explicitly frames this as code generation with world models and reports strong coding benchmarks such as SWE-bench Verified and LiveCodeBench. ([arXiv](https://arxiv.org/abs/2510.02387))

The **world model** part makes a lot of sense for code. In robotics or games, the world is messy and partially observed. In programming, the “world” is unusually clean: source code, interpreter state, compiler errors, stdout/stderr, stack traces, tests, type checkers, linters, package APIs, file systems, and git diffs. Classic world-model work showed that an agent can learn a model of its environment and even train inside imagined rollouts; Dreamer-style agents later scaled this idea by learning a world model and improving behavior through imagined futures. ([arXiv](https://arxiv.org/abs/1803.10122)) ([arXiv](https://arxiv.org/abs/2301.04104)) For code, the environment is digital and cheap enough that you can alternate between **real execution** and **learned simulation**.

The **diffusion** part is also reasonable, but I would not bet everything on it as the only architecture. Diffusion is attractive for code because programming is not purely left-to-right. Humans sketch, revise, fill holes, move blocks, repair earlier mistakes, and refactor. Diffusion-style language models generate by iterative denoising over a whole sequence, which fits that workflow better than strict autoregression. Apple’s **DiffuCoder** paper argues exactly this: diffusion LLMs have global planning and iterative refinement properties useful for code, trained a 7B diffusion model on 130B code tokens, and proposed a diffusion-native RL method that improved EvalPlus by 4.4%. ([arXiv](https://arxiv.org/abs/2506.20639))

The **pure RL** part is where I would be careful. Pure RL from random initialization is probably infeasible for serious programming, because the action space is huge and rewards are sparse. A random model will almost never stumble into compiling, meaningful, tested programs. But “pure RL after pretraining” is much more realistic. DeepSeek-R1-Zero is a useful analogy: it applied large-scale RL directly to a pretrained base model without SFT as a preliminary step, and the authors report emergent reasoning behaviors; they also note issues like repetition, poor readability, and language mixing, which is why DeepSeek-R1 added cold-start data and more stages. ([GitHub](https://github.com/deepseek-ai/deepseek-r1))

So my verdict is:

**The idea is good. The dream version is pure-RL-from-scratch. The practical version is pretrained code model + execution traces + verifiable RL + world model + real interpreter loop.**

A strong architecture might look like this:

```text
Prompt / Issue / Failing tests
        ↓
Coding policy model
        ↓
Proposes action:
  - write code
  - edit file
  - run test
  - inspect stack trace
  - search repo
  - add fuzz test
        ↓
Digital environment:
  - Python/JS/Rust/etc. interpreter
  - compiler
  - unit tests
  - type checker
  - linter
  - profiler
  - package docs
        ↓
Observation:
  - pass/fail
  - stdout/stderr
  - traceback
  - coverage
  - performance
  - diff
        ↓
Reward + world-model training signal
        ↓
Model improves
```

The reward should not only be “tests pass.” That is too hackable. You would want a bundle of verifiers: compilation, hidden tests, fuzz tests, static analysis, security checks, type checks, coverage, performance constraints, and maybe readability/style scores. Recent RL-for-code work such as REAL uses rewards from static analyzers plus unit tests to train models toward production-quality code, not just syntactically plausible code. ([OpenReview](https://openreview.net/forum?id=3nza35A6I4))

Where diffusion could be especially powerful:

```text
Autoregressive model:
  "Write the next token."

Diffusion coding model:
  "Here is a partial broken program. Iteratively denoise it into a correct program."

World-model coding agent:
  "Predict what will happen if I run this patch, then decide what to edit next."
```

That suggests diffusion may shine less as a chatbot replacement and more as a **patch generator / repair operator / fill-in-middle engine / refactoring engine**. For example, give it a failing function, stack trace, and tests; mask uncertain regions; let it denoise several candidate patches; run them; use RL/verifier feedback to update the policy.

The main hard problems are not philosophical. They are engineering and training dynamics:

1. **Credit assignment:** A test failure may be caused by one wrong line among thousands. The model needs intermediate signals, not just pass/fail.

2. **Reward hacking:** The agent may overfit visible tests, delete tests, mock the wrong thing, hardcode outputs, or exploit the harness unless the environment is locked down.

3. **Long-horizon state:** Real software tasks involve many files, dependencies, build systems, and hidden assumptions. A world model can help, but it can also hallucinate execution unless periodically grounded by a real interpreter.

4. **Diffusion exactness:** Code is brittle. One token can break everything. Diffusion over discrete tokens is promising, but autoregressive models still have massive infrastructure advantages.

5. **Environment scale:** To learn deeply, the model needs millions or billions of useful interactions: generated tasks, real repos, tests, traces, stack states, package APIs, and secure sandboxes.

The smallest credible experiment would be:

```text
Domain: Python only
Tasks: small functions + bug-fix tasks + unit tests
Model: small code model, maybe diffusion or masked-edit model
Environment: pytest + mypy/ruff + hidden fuzz tests
Training:
  1. Pretrain / finetune on code + edits + traces
  2. Let model propose patches
  3. Run tests and static checks
  4. Reward correct, minimal, robust patches
  5. Store successful trajectories
  6. Distill them back into the model
```

Then scale from “single function” to “multi-file repo” to “agentic software engineering.” That is basically the path from code generation toward a real coding-native agent.

My personal take: **your idea is directionally excellent**, but the winning system probably will not be “diffusion + pure RL only.” It will be a hybrid: pretrained code model, diffusion-style editing/search, real execution feedback, learned code world model, RL with verifiable rewards, and constant grounding in actual interpreters. The reason this is exciting is that programming gives AI something rare: an environment where actions are cheap, feedback is objective, and improvement can be measured automatically.

---

**Sources:**

- [[2510.02387] CWM: An Open-Weights LLM for Research on Code Generation with World Models](https://arxiv.org/abs/2510.02387)
- [GitHub - deepseek-ai/DeepSeek-R1 · GitHub](https://github.com/deepseek-ai/deepseek-r1)
- [Training Language Models to Generate Quality Code with Program Analysis Feedback | OpenReview](https://openreview.net/forum?id=3nza35A6I4)

## Prompt:
6/6/2026, 12:32:36 AM

No you don't understand. It's not an LLM

## Response:
6/6/2026, 12:32:37 AM

Ah, yes - then your idea is **much more radical and more interesting** than “another coding LLM.”

You are imagining something closer to:

> **A programming-native RL agent whose internal world is code execution, not language.**  
> It learns by acting inside interpreters, compilers, debuggers, test suites, file systems, profilers, and maybe simulated software environments.

That is a valid idea. I would not call it an LLM. I would call it something like a **code world model**, **program-state model**, **execution-native coding agent**, or **RL program synthesizer**.

The key distinction is this:

```text
LLM coding model:
  Input/output object: text
  Training signal: predict next token
  Later: maybe use tools

Your model:
  Input/output object: program state / execution state / AST / IR / memory / tests
  Training signal: environment reward
  Core loop: act → execute → observe → update
```

That is a very different beast.

The model would not primarily “write code” the way humans write English. It would manipulate **formal objects**: ASTs, bytecode, SSA graphs, control-flow graphs, type constraints, heap states, stack traces, symbolic states, test outcomes, and performance profiles. Text code would just be one possible rendering of the underlying object.

A clean version might look like this:

```text
State:
  - task specification
  - current program graph
  - AST / IR / CFG
  - type information
  - interpreter state
  - test results
  - stack traces
  - coverage
  - profiler data
  - symbolic constraints

Actions:
  - add node to AST
  - replace expression
  - introduce variable
  - change control flow
  - run tests
  - inspect trace
  - ask solver for counterexample
  - specialize function
  - refactor subgraph
  - optimize loop
  - generate new test

Environment:
  - compiler
  - interpreter
  - unit tests
  - fuzzing engine
  - theorem prover / SMT solver
  - profiler
  - package/runtime sandbox

Reward:
  - compiles
  - passes visible tests
  - passes hidden tests
  - satisfies properties
  - avoids undefined behavior
  - lower latency / memory
  - simpler program
  - more general solution
```

In that setup, **diffusion** would also mean something different from diffusion over text. You could use diffusion over **program structures** or **latent execution plans**.

For example:

```text
Start with noisy/incomplete program graph
        ↓
Denoise toward a valid typed AST
        ↓
Run interpreter/tests
        ↓
Use failures as constraints
        ↓
Denoise again with verifier guidance
```

That is interesting because code has global constraints. A program is not just a sequence; it is a structured object where one variable name, type, branch condition, or loop invariant can affect everything else. Diffusion’s “global editable canvas” is naturally attractive here. Recent discrete-diffusion code work is still usually framed around language models, but the useful idea is broader: diffusion exposes intermediate whole-program states that can be constrained or corrected during generation, not only after a full left-to-right sample is produced. ([arXiv](https://arxiv.org/abs/2605.16829?utm_source=chatgpt.com))

The **pure RL** part becomes more plausible if the action space is not raw text. Pure RL over characters or tokens is brutal. But pure RL over a typed program-construction environment is much more realistic.

Think of the difference:

```text
Bad action space:
  choose next character/token

Better action space:
  choose valid grammar production

Even better:
  choose typed AST operation

Even better:
  choose semantic transformation:
    "map over list"
    "add base case"
    "memoize recursive call"
    "strength-reduce multiplication"
    "split function"
    "introduce invariant"
```

This is where your idea stops being a dream. There are already partial precedents. DeepMind’s **AlphaDev** did not behave like a normal coding LLM; it treated assembly-level algorithm discovery as a single-player game and used deep reinforcement learning to discover small sorting routines that beat known human benchmarks. ([Nature](https://www.nature.com/articles/s41586-023-06004-9?utm_source=chatgpt.com)) DreamCoder is another relevant ancestor: not an LLM chatbot, but a program-synthesis system that learns reusable abstractions and a neural search policy while solving programming-like tasks. ([arXiv](https://arxiv.org/abs/2006.08381?utm_source=chatgpt.com))

So the core idea is sound:

> Programming is one of the best domains for pure environment learning because the environment gives exact feedback.

Unlike ordinary language, code can be executed. The model can be wrong, run the program, see exactly how it is wrong, mutate the program, and try again. That makes coding unusually suitable for RL, search, self-play, and world modeling.

The hard part is scale.

For toy DSLs, assembly kernels, theorem-proving subproblems, SQL queries, regex synthesis, small Python functions, compiler optimizations, or algorithm puzzles, your idea is very plausible.

For full software engineering from scratch, pure RL is still extremely hard because the reward is sparse and delayed. A million bad programs may all receive the same reward: “failed.” The model needs a curriculum where early tasks give dense feedback.

A practical path would be:

```text
Stage 1:
  Learn syntax and execution in a tiny DSL.

Stage 2:
  Learn typed AST editing.

Stage 3:
  Learn from interpreter traces:
    input → program → execution path → output/error.

Stage 4:
  Add RL:
    generate/modify programs to satisfy tests.

Stage 5:
  Add adversarial testing:
    model writes code; environment finds counterexamples.

Stage 6:
  Add world model:
    predict what will happen before running the code.

Stage 7:
  Add diffusion planner:
    denoise broken/incomplete program states into valid solutions.

Stage 8:
  Scale to real languages:
    Python subset → full Python → C/Rust/JS → repos.
```

The most important design choice is probably **representation**. If the model’s native representation is still text, it collapses back toward LLM territory. If the native representation is AST/IR/execution state, then it is genuinely programming-native.

I think the best version would look like this:

```text
             ┌─────────────────────────┐
             │     Task / objective     │
             └───────────┬─────────────┘
                         ↓
             ┌─────────────────────────┐
             │ Program-state encoder    │
             │ AST + CFG + types + env  │
             └───────────┬─────────────┘
                         ↓
       ┌─────────────────────────────────────┐
       │ Latent world model                   │
       │ predicts execution/test consequences │
       └───────────┬─────────────────────────┘
                   ↓
       ┌─────────────────────────────────────┐
       │ Diffusion/search policy              │
       │ proposes edits/program trajectories  │
       └───────────┬─────────────────────────┘
                   ↓
       ┌─────────────────────────────────────┐
       │ Real environment                     │
       │ compiler/interpreter/tests/profiler  │
       └───────────┬─────────────────────────┘
                   ↓
       ┌─────────────────────────────────────┐
       │ Reward + counterexamples + traces    │
       └─────────────────────────────────────┘
```

In this framing, language is optional. You could add a language interface later so humans can give tasks in English, but the core intelligence would not be next-token prediction. It would be closer to **model-based RL over formal executable artifacts**.

My honest take: **this is a good idea, not just a dream.** But the winning version probably starts narrow. Do not try to build “general coding intelligence” first. Start with a domain where execution is cheap, correctness is checkable, and the action space is structured.

Good starter domains:

```text
1. Tiny functional DSL
2. Regex synthesis
3. SQL query synthesis
4. Python function synthesis from tests
5. NumPy/tensor program synthesis
6. Compiler peephole optimization
7. Assembly micro-kernel optimization
8. Theorem-prover tactic generation
9. Bug repair in small repos
```

The “pure RL” dream is most realistic in domains like assembly optimization or DSL synthesis, where the environment can score millions of candidates. AlphaDev is basically evidence that this family of approach can work when the game is well-defined. ([Nature](https://www.nature.com/articles/s41586-023-06004-9?utm_source=chatgpt.com))

So I would revise your idea into one sentence:

> **Train a non-LLM, programming-native world model that learns to construct, execute, debug, and optimize structured programs through RL inside formal digital environments, using diffusion as a global program-state refinement mechanism.**

That is not fantasy. That is a serious research program.

## Prompt:
6/6/2026, 12:43:41 AM

What do you mean add RL at stage 4, does that mean stage 1-3 is supervised?

## Response:
6/6/2026, 12:43:42 AM

No — **not necessarily**.

When I said “add RL at stage 4,” I meant:

> **Stage 1–3: learn the programming world.**  
> **Stage 4+: start optimizing behavior using rewards from the environment.**

That does **not** require human-labeled supervised learning.

A better breakdown is:

```text
Stage 1–3:
  Learn representations, syntax, semantics, execution dynamics.

Stage 4:
  Learn to solve tasks by taking actions and receiving rewards.
```

The early stages could be **self-supervised**, **environment-generated**, or **solver-generated**, not supervised in the normal “dataset of labeled answers” sense.

For example:

```text
Generate random valid programs
        ↓
Run them in interpreter
        ↓
Collect traces:
  AST
  input
  execution path
  variable states
  output/error
        ↓
Train model to predict:
  next execution state
  final output
  type errors
  runtime errors
  affected lines
  valid AST repairs
```

That is not human supervision. The interpreter produces the targets.

Technically, some of this is “supervised” in the mathematical sense because the model predicts a known target, like:

```text
(program, input, current_state) → next_state
(program, input) → output
(broken_AST) → repaired_AST
```

But philosophically it is closer to **self-supervised world-model learning from interaction traces**, not classic supervised learning.

The distinction is important:

```text
Supervised coding model:
  "Here is prompt X. Correct answer is code Y."

Self-supervised execution model:
  "Here is a program and state. Predict what happens next."

RL coding agent:
  "Choose actions in the environment to maximize reward."
```

So stages 1–3 might look like this:

### Stage 1: learn the formal substrate

The model learns what valid programs look like.

Not necessarily from labeled solutions. It can learn from generated programs, grammar rules, AST corruption/reconstruction, parser feedback, type-checker feedback, and execution results.

Example objectives:

```text
noisy AST → valid AST
partial AST → completed AST
program → parse tree
program → type constraints
program + input → output
```

This can be self-supervised.

### Stage 2: learn program transformations

The model learns how edits change programs.

For example:

```text
old_AST + edit_action → new_AST
old_program + refactor → equivalent_program
broken_program + compiler_error → likely repair region
```

Again, this can be generated automatically. You can mutate programs, run them, compare behavior, and learn which edits preserve or change semantics.

### Stage 3: learn the world model

This is where the system learns execution dynamics.

```text
program + input + current_state → next_state
program + test_case → pass/fail
program + edit → predicted future test result
```

This is not RL yet because the model is mostly learning to **predict** the environment, not necessarily to **choose actions** that maximize reward.

### Stage 4: add RL

Now the model is allowed to act.

```text
observe task
choose edit
run program
observe failure
choose another edit
run tests
receive reward
update policy
```

Reward could be:

```text
+1.0  hidden tests pass
+0.3  visible tests pass
+0.1  program compiles
+0.1  type checker passes
-0.1  runtime error
-0.2  deletes tests
-0.3  hardcodes visible examples
-0.01 per unnecessary edit
```

This is where it becomes an actual agent.

The reason I would delay “main RL” until stage 4 is that **pure RL from zero is extremely sparse**. If the agent starts totally ignorant, most actions produce garbage:

```text
random AST edit → invalid program
random program → does not compile
random compiled program → wrong behavior
random wrong behavior → no useful reward
```

So the agent learns almost nothing.

But after stages 1–3, the agent already knows things like:

```text
what valid programs look like
how execution works
what errors mean
which edits are legal
which edits are likely useful
```

Then RL has something to improve.

A good analogy is robotics:

```text
World-model phase:
  Learn how objects move when pushed.

RL phase:
  Learn how to push objects to achieve goals.
```

For your coding-native model:

```text
World-model phase:
  Learn how programs execute and fail.

RL phase:
  Learn how to construct/debug/optimize programs to achieve goals.
```

There are basically three possible philosophies:

```text
1. Strict pure RL from birth
   No pretraining, no imitation, only reward.
   Very elegant, very hard.

2. Self-supervised world model + RL
   Learn execution first, then optimize actions.
   Probably the best practical version.

3. Supervised/imitation + self-supervised + RL
   Use solved examples, traces, solvers, maybe humans.
   Most practical, but less philosophically pure.
```

For your idea, I think option 2 is the sweet spot:

> **No LLM. No next-token language modeling. No human-labeled code answers required. Train a program-state world model from interpreter interaction, then train an RL policy over structured program actions.**

So the answer is:

**Stage 1–3 do not have to be supervised. They can be self-supervised from the digital environment. Stage 4 is where the system starts using reward to learn goal-directed programming behavior.**

## Prompt:
6/6/2026, 12:49:43 AM

And the goal is something like, leet code pass rate. Because our model would be new, parameters -wise (no base model LLM), and method-eise (code native world model, not LLM RL on coding task).
So a "toy-ish" "ideal" experiment setup is still acceptable I'd say

## Response:
6/6/2026, 12:49:43 AM

Yes - **a toy-ish ideal experiment is not only acceptable, it is probably the correct way to do this**.

For this kind of idea, you should not start by trying to beat GPT-style models on full LeetCode. That would mix too many problems together:

```text
1. understanding English problem statements
2. inventing algorithms
3. writing syntactically valid code
4. debugging
5. generalizing beyond visible tests
6. using libraries
7. managing long context
```

Your idea is mainly about **2-5**, not necessarily about English understanding. So the cleanest experiment should isolate the programming-native part.

The goal could be:

> **Train a non-LLM, code-native world-model/RL system from scratch, with no pretrained language model, and measure pass rate on LeetCode-like programming tasks.**

That is a good research target.

But I would phrase the first benchmark as **Mini-LeetCode**, not real LeetCode.

Real LeetCode-style benchmarks are useful later. LiveCodeBench, for example, collects programming problems from LeetCode, AtCoder, and CodeForces and evaluates code generation, self-repair, execution, and test-output prediction. ([arXiv](https://arxiv.org/html/2403.07974v2?utm_source=chatgpt.com)) EvalPlus is also relevant because it shows that weak test suites can overestimate coding ability; it augments HumanEval with many generated tests and found that stronger tests can substantially reduce measured pass rates. ([arXiv](https://arxiv.org/abs/2305.01210?utm_source=chatgpt.com)) For your first experiment, you want that same spirit: **hidden, strong, automatically generated tests**, but in a smaller controlled world.

The clean experimental claim could be:

> **A randomly initialized, non-LLM code-native agent can learn to synthesize and repair small programs through interpreter interaction, achieving nontrivial pass@k on held-out LeetCode-like tasks.**

That is a legitimate result even if the tasks are toy-sized.

## The ideal first setup

I would define a tiny programming environment, maybe called something like **MiniPy**, **MiniLeet**, or **CodeGrid**.

Not raw Python at first. Use a constrained typed subset:

```text
Types:
  int
  bool
  list[int]
  string, optional later

Operations:
  arithmetic
  comparisons
  if/else
  for loops
  while loops, maybe later
  list indexing
  list append
  slicing, optional later
  recursion, optional later

Program representation:
  typed AST, not text tokens

Environment:
  parser
  type checker
  interpreter
  unit-test runner
  fuzz tester
  runtime-error detector
  coverage tracker
```

The model does **not** output text.

It acts on structured program states:

```text
Action examples:
  add if statement
  replace expression
  introduce variable
  insert loop
  change comparison operator
  add accumulator
  change return expression
  specialize branch
  run tests
  inspect failing trace
  request counterexample
```

That keeps it genuinely non-LLM.

The model’s observation could be:

```text
current AST
type state
known input-output examples
test failures
execution trace
variable values over time
coverage
runtime errors
```

The target task is:

```text
Given:
  function signature
  input-output examples
  constraints
  hidden tests, inaccessible to agent

Produce:
  a program that passes hidden tests
```

This is LeetCode-like, but without the English front end.

## What “pass rate” should mean

Use something like:

```text
pass@1:
  percentage of tasks solved by the first final submitted program

pass@k:
  percentage solved within k submitted candidate programs

solve@budget:
  percentage solved within N environment interactions

interaction efficiency:
  average test runs / edits / rollouts per solved problem

generalization score:
  visible tests pass vs hidden fuzz tests pass
```

The most important metric is not just pass@k. It is:

```text
hidden pass rate under a fixed interaction budget
```

Because your model is an **interactive agent**, not a one-shot text generator.

For example:

```text
Task budget:
  max 200 AST edits
  max 50 test executions
  max 10 final submissions

Metric:
  hidden pass@10 under 50 executions
```

That would be fairer than comparing it to an LLM’s one-shot answer.

## Training without an LLM

You can train from random initialization.

The early data can come entirely from the environment:

```text
1. Generate random valid programs.
2. Generate random inputs.
3. Run the interpreter.
4. Store execution traces.
5. Corrupt programs.
6. Ask the model to repair or predict behavior.
7. Let the agent attempt tasks.
8. Reward programs that pass tests.
```

So no GitHub. No natural-language code corpus. No pretrained base model.

The training signal could be:

```text
Self-supervised world-model losses:
  predict next interpreter state
  predict final output
  predict runtime error
  predict type error
  predict test pass/fail
  reconstruct corrupted AST
  predict effect of an edit

RL losses:
  maximize hidden/held-out test reward
  minimize unnecessary edits
  avoid runtime errors
  improve pass rate under interaction budget
```

This is where your distinction is strongest.

An LLM-based coding system says:

```text
I learned code mostly as text.
Now I use RL/tools to improve.
```

Your system says:

```text
I learned code as executable state transitions.
Now I use RL/search to construct programs.
```

That is a real methodological difference.

## A good toy benchmark

I would build around small algorithmic families.

For example:

```text
Level 1:
  max of list
  count positives
  sum list
  reverse list
  check sorted
  remove zeros

Level 2:
  two-sum small
  prefix sums
  frequency counting
  palindrome check
  merge sorted lists
  deduplicate list

Level 3:
  sliding window
  binary search
  interval merge
  simple dynamic programming
  graph reachability
  shortest path on tiny graphs
```

But do not train directly on these exact named tasks. Instead, generate many procedural tasks from semantic templates, then evaluate on held-out combinations.

Example:

```text
Training family:
  "filter then aggregate"
  "scan with accumulator"
  "nested loop comparison"
  "frequency map"

Held-out task:
  "return the first index whose prefix sum exceeds k"
```

This tests whether the agent learned reusable programming behavior, not memorized task names.

## The model architecture could be small

You do not need billions of parameters.

A plausible first model:

```text
AST encoder:
  graph neural network or tree transformer over typed AST

Trace encoder:
  small transformer/RNN over execution states

World model:
  predicts next state / test result / error

Policy:
  chooses AST edit actions

Value model:
  predicts probability of eventual solve

Diffusion module, optional:
  denoises broken or incomplete ASTs into valid candidate ASTs
```

The diffusion part should probably operate over **ASTs or latent program graphs**, not source-code tokens.

Something like:

```text
noisy/incomplete typed AST
        ↓
denoising steps
        ↓
valid candidate program
        ↓
real interpreter verification
        ↓
reward/update
```

That would make the diffusion component native to programming structure rather than language.

## Why this is publishable even if toy

Because the claim is not:

> “We beat GPT-5.5 on LeetCode.”

The claim is:

> “A non-language-model agent can acquire programming ability through executable environment interaction.”

That is interesting.

There are precedents showing that narrow code/program-search domains can produce real results. AlphaDev formulated assembly-level algorithm discovery as a single-player game and used deep RL to discover faster small sorting routines. ([Nature](https://www.nature.com/articles/s41586-023-06004-9?utm_source=chatgpt.com)) DreamCoder is another relevant ancestor: it learns reusable program abstractions and a neural search policy rather than behaving like a normal language model. ([arXiv](https://arxiv.org/abs/2006.08381?utm_source=chatgpt.com)) Your proposed experiment would sit in that family, but with a more explicit world-model/interpreter-interaction framing.

## The crucial ablations

To make the experiment convincing, you would need ablations like:

```text
A. Random search over ASTs
B. Enumerative program synthesis
C. Genetic programming baseline
D. Model-free RL agent
E. World model + planning, no diffusion
F. Diffusion repair model, no RL
G. World model + diffusion + RL
```

The result you want is not merely “the agent solves some tasks.”

The result you want is:

```text
World-model training improves sample efficiency.
RL improves pass rate beyond self-supervised prediction.
Diffusion/edit refinement improves repair and search.
The agent generalizes to held-out task compositions.
```

A strong toy result might look like:

```text
MiniLeet-100 hidden pass@10:

Enumerative search:        12%
Genetic programming:       18%
Model-free RL:             21%
World model + planning:    34%
Diffusion repair only:     37%
World model + RL + repair: 52%
```

Those numbers are hypothetical, but that is the kind of result table that would make the idea legible.

## I would not start with raw LeetCode

Raw LeetCode adds an English problem-statement interface. If your model has no LLM, it needs either:

```text
1. a separate language-to-formal-spec front end, or
2. a structured version of the benchmark, or
3. tasks defined only by signatures, examples, and tests.
```

For the first paper or prototype, choose option 3.

Later, you can add a thin translation layer:

```text
English problem
        ↓
formal task spec / examples / constraints
        ↓
non-LLM programming-native agent
        ↓
code solution
```

But the core contribution should not depend on natural-language understanding.

## The cleanest “ideal experiment” in one paragraph

Train a randomly initialized, non-LLM agent in a typed Python-like DSL. The agent observes function signatures, examples, failing tests, AST state, and execution traces. It acts by editing typed ASTs and running tests. First, it self-supervises on interpreter-generated traces to learn a program world model. Then it uses RL to maximize hidden-test pass rate under a fixed interaction budget. Evaluation is on held-out LeetCode-like algorithmic tasks with strong fuzzed hidden tests, using pass@k, solve@budget, and interaction efficiency. Compare against enumerative search, genetic programming, model-free RL, and ablated world-model/diffusion variants.

That is a very reasonable toy-ish ideal setup.

Actually, I think it is better than jumping directly to full LeetCode, because it would answer the real question:

> **Can programming ability emerge from interaction with executable formal environments, without language-model pretraining?**

That question is absolutely worth testing.

---

**Sources:**

- [LiveCodeBench: Holistic and Contamination Free ...](https://arxiv.org/html/2403.07974v2?utm_source=chatgpt.com)
- [Faster sorting algorithms discovered using deep ...](https://www.nature.com/articles/s41586-023-06004-9?utm_source=chatgpt.com)



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)
