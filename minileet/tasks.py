from __future__ import annotations

import random
import zlib
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

from minileet.dsl import BOOL, INT, LIST_INT, Param, Type


Example = tuple[tuple[Any, ...], Any]
InputSampler = Callable[[random.Random], tuple[Any, ...]]
Oracle = Callable[..., Any]


@dataclass(frozen=True)
class Signature:
    name: str
    params: tuple[Param, ...]
    return_type: Type

    @property
    def param_names(self) -> tuple[str, ...]:
        return tuple(param.name for param in self.params)


@dataclass(frozen=True)
class Task:
    name: str
    signature: Signature
    visible_examples: tuple[Example, ...]
    oracle: Oracle
    sampler: InputSampler
    description: str = ""

    def hidden_tests(self, count: int = 64, seed: int = 0) -> tuple[Example, ...]:
        stable_task_seed = zlib.adler32(self.name.encode("utf-8"))
        rng = random.Random(stable_task_seed ^ seed)
        examples: list[Example] = []
        seen: set[str] = set()
        attempts = 0
        while len(examples) < count and attempts < count * 20:
            attempts += 1
            args = self.sampler(rng)
            key = repr(args)
            if key in seen:
                continue
            seen.add(key)
            examples.append((args, self.oracle(*args)))
        return tuple(examples)


@dataclass(frozen=True)
class PredicateSpec:
    name: str
    params: tuple[Param, ...]
    fn: Callable[..., bool]
    sampler: InputSampler


SUITE_NAMES = ("starter", "generated", "train", "eval", "all", "procedural_train", "procedural_eval")


def default_tasks() -> tuple[Task, ...]:
    return (
        Task(
            name="sum_list",
            signature=Signature("solve", (Param("xs", LIST_INT),), INT),
            visible_examples=(
                (((1, 2, 3),), 6),
                (((-2, 5),), 3),
                (((),), 0),
            ),
            oracle=lambda xs: sum(xs),
            sampler=lambda rng: (_rand_list(rng),),
            description="Return the sum of all integers in xs.",
        ),
        Task(
            name="count_positive",
            signature=Signature("solve", (Param("xs", LIST_INT),), INT),
            visible_examples=(
                (((1, -2, 3),), 2),
                (((-2, 0),), 0),
                (((5,),), 1),
            ),
            oracle=lambda xs: sum(1 for x in xs if x > 0),
            sampler=lambda rng: (_rand_list(rng),),
            description="Return the number of values greater than zero.",
        ),
        Task(
            name="contains_target",
            signature=Signature("solve", (Param("xs", LIST_INT), Param("target", INT)), BOOL),
            visible_examples=(
                (((1, 2, 3), 2), True),
                (((1, 2, 3), 7), False),
                (((), 0), False),
            ),
            oracle=lambda xs, target: target in xs,
            sampler=lambda rng: (_rand_list(rng), rng.randint(-5, 5)),
            description="Return whether xs contains target.",
        ),
        Task(
            name="max_or_zero",
            signature=Signature("solve", (Param("xs", LIST_INT),), INT),
            visible_examples=(
                (((1, 4, 2),), 4),
                (((-7, -2),), -2),
                (((),), 0),
            ),
            oracle=lambda xs: max(xs) if xs else 0,
            sampler=lambda rng: (_rand_list(rng),),
            description="Return max(xs), or zero for an empty list.",
        ),
        Task(
            name="all_nonnegative",
            signature=Signature("solve", (Param("xs", LIST_INT),), BOOL),
            visible_examples=(
                (((0, 2, 3),), True),
                (((1, -1, 3),), False),
                (((),), True),
            ),
            oracle=lambda xs: all(x >= 0 for x in xs),
            sampler=lambda rng: (_rand_list(rng),),
            description="Return whether every value in xs is nonnegative.",
        ),
    )


def suite_tasks(suite: str = "starter") -> tuple[Task, ...]:
    if suite == "starter":
        return default_tasks()
    generated = generated_tasks()
    if suite == "generated":
        return generated
    if suite == "train":
        return tuple(task for task in generated if _split_bucket(task.name) != 0)
    if suite == "eval":
        return tuple(task for task in generated if _split_bucket(task.name) == 0)
    if suite == "all":
        return default_tasks() + generated
    if suite == "procedural_train":
        return procedural_tasks(split="train")
    if suite == "procedural_eval":
        return procedural_tasks(split="eval")
    raise ValueError(f"unknown task suite {suite}")


def procedural_tasks(
    split: str,
    count: int = 0,
    seed: int = 0,
) -> tuple[Task, ...]:
    if split not in ("train", "eval"):
        raise ValueError(f"unknown procedural split {split}")

    profiles = _sampler_profiles(seed)
    tasks: list[Task] = []
    for profile_name, sampler in profiles:
        for predicate in _procedural_predicate_specs(sampler):
            tasks.extend(
                [
                    _make_count_task(predicate, profile_name),
                    _make_sum_task(predicate, profile_name),
                    _make_any_task(predicate, profile_name),
                    _make_all_task(predicate, profile_name),
                    _make_first_or_zero_task(predicate, profile_name),
                ]
            )
        tasks.extend(_comparison_tasks(profile_name, sampler))

    filtered = tuple(task for task in tasks if _procedural_split(task.name) == split)
    if count > 0:
        return filtered[:count]
    return filtered


def generated_tasks() -> tuple[Task, ...]:
    tasks: list[Task] = []
    for predicate in _predicate_specs():
        tasks.extend(
            [
                _make_count_task(predicate),
                _make_sum_task(predicate),
                _make_any_task(predicate),
                _make_all_task(predicate),
                _make_first_or_zero_task(predicate),
            ]
        )
    tasks.extend(_comparison_tasks())
    return tuple(tasks)


def task_by_name(name: str, tasks: Iterable[Task] | None = None) -> Task:
    pool = suite_tasks("all") if tasks is None else tuple(tasks)
    for task in pool:
        if task.name == name:
            return task
    raise KeyError(name)


def _predicate_specs() -> tuple[PredicateSpec, ...]:
    xs = Param("xs", LIST_INT)
    k = Param("k", INT)
    return (
        PredicateSpec(
            "positive",
            (xs,),
            lambda x: x > 0,
            lambda rng: (_rand_list(rng),),
        ),
        PredicateSpec(
            "nonnegative",
            (xs,),
            lambda x: x >= 0,
            lambda rng: (_rand_list(rng),),
        ),
        PredicateSpec(
            "negative",
            (xs,),
            lambda x: x < 0,
            lambda rng: (_rand_list(rng),),
        ),
        PredicateSpec(
            "zero",
            (xs,),
            lambda x: x == 0,
            lambda rng: (_rand_list(rng),),
        ),
        PredicateSpec(
            "even",
            (xs,),
            lambda x: x % 2 == 0,
            lambda rng: (_rand_list(rng),),
        ),
        PredicateSpec(
            "odd",
            (xs,),
            lambda x: x % 2 != 0,
            lambda rng: (_rand_list(rng),),
        ),
        PredicateSpec(
            "eq_k",
            (xs, k),
            lambda x, k: x == k,
            lambda rng: (_rand_list(rng), rng.randint(-5, 5)),
        ),
        PredicateSpec(
            "gt_k",
            (xs, k),
            lambda x, k: x > k,
            lambda rng: (_rand_list(rng), rng.randint(-5, 5)),
        ),
        PredicateSpec(
            "lt_k",
            (xs, k),
            lambda x, k: x < k,
            lambda rng: (_rand_list(rng), rng.randint(-5, 5)),
        ),
    )


def _procedural_predicate_specs(sampler: InputSampler) -> tuple[PredicateSpec, ...]:
    xs = Param("xs", LIST_INT)
    k = Param("k", INT)
    specs: list[PredicateSpec] = [
        PredicateSpec("even", (xs,), lambda x: x % 2 == 0, sampler),
        PredicateSpec("odd", (xs,), lambda x: x % 2 != 0, sampler),
    ]
    for const in (-3, -2, -1, 0, 1, 2, 3):
        safe_name = _const_name(const)
        specs.extend(
            [
                PredicateSpec(f"gt_{safe_name}", (xs,), lambda x, c=const: x > c, sampler),
                PredicateSpec(f"ge_{safe_name}", (xs,), lambda x, c=const: x >= c, sampler),
                PredicateSpec(f"lt_{safe_name}", (xs,), lambda x, c=const: x < c, sampler),
                PredicateSpec(f"le_{safe_name}", (xs,), lambda x, c=const: x <= c, sampler),
                PredicateSpec(f"eq_{safe_name}", (xs,), lambda x, c=const: x == c, sampler),
                PredicateSpec(f"ne_{safe_name}", (xs,), lambda x, c=const: x != c, sampler),
            ]
        )
    specs.extend(
        [
            PredicateSpec("eq_k", (xs, k), lambda x, k: x == k, _with_k_sampler(sampler, -6, 6)),
            PredicateSpec("gt_k", (xs, k), lambda x, k: x > k, _with_k_sampler(sampler, -6, 6)),
            PredicateSpec("ge_k", (xs, k), lambda x, k: x >= k, _with_k_sampler(sampler, -6, 6)),
            PredicateSpec("lt_k", (xs, k), lambda x, k: x < k, _with_k_sampler(sampler, -6, 6)),
            PredicateSpec("le_k", (xs, k), lambda x, k: x <= k, _with_k_sampler(sampler, -6, 6)),
            PredicateSpec("ne_k", (xs, k), lambda x, k: x != k, _with_k_sampler(sampler, -6, 6)),
        ]
    )
    return tuple(specs)


def _make_count_task(predicate: PredicateSpec, profile_name: str = "") -> Task:
    def oracle(*args: Any, pred: PredicateSpec = predicate) -> int:
        xs = args[0]
        extra = args[1:]
        return sum(1 for x in xs if pred.fn(x, *extra))

    return _build_generated_task(
        name=_task_name("count", predicate.name, profile_name),
        params=predicate.params,
        return_type=INT,
        oracle=oracle,
        sampler=predicate.sampler,
        description=f"Return how many list elements satisfy {predicate.name}.",
    )


def _make_sum_task(predicate: PredicateSpec, profile_name: str = "") -> Task:
    def oracle(*args: Any, pred: PredicateSpec = predicate) -> int:
        xs = args[0]
        extra = args[1:]
        return sum(x for x in xs if pred.fn(x, *extra))

    return _build_generated_task(
        name=_task_name("sum", predicate.name, profile_name),
        params=predicate.params,
        return_type=INT,
        oracle=oracle,
        sampler=predicate.sampler,
        description=f"Return the sum of list elements satisfying {predicate.name}.",
    )


def _make_any_task(predicate: PredicateSpec, profile_name: str = "") -> Task:
    def oracle(*args: Any, pred: PredicateSpec = predicate) -> bool:
        xs = args[0]
        extra = args[1:]
        return any(pred.fn(x, *extra) for x in xs)

    return _build_generated_task(
        name=_task_name("any", predicate.name, profile_name),
        params=predicate.params,
        return_type=BOOL,
        oracle=oracle,
        sampler=predicate.sampler,
        description=f"Return whether any list element satisfies {predicate.name}.",
    )


def _make_all_task(predicate: PredicateSpec, profile_name: str = "") -> Task:
    def oracle(*args: Any, pred: PredicateSpec = predicate) -> bool:
        xs = args[0]
        extra = args[1:]
        return all(pred.fn(x, *extra) for x in xs)

    return _build_generated_task(
        name=_task_name("all", predicate.name, profile_name),
        params=predicate.params,
        return_type=BOOL,
        oracle=oracle,
        sampler=predicate.sampler,
        description=f"Return whether every list element satisfies {predicate.name}.",
    )


def _make_first_or_zero_task(predicate: PredicateSpec, profile_name: str = "") -> Task:
    def oracle(*args: Any, pred: PredicateSpec = predicate) -> int:
        xs = args[0]
        extra = args[1:]
        for x in xs:
            if pred.fn(x, *extra):
                return x
        return 0

    return _build_generated_task(
        name=_task_name("first_or_zero", predicate.name, profile_name),
        params=predicate.params,
        return_type=INT,
        oracle=oracle,
        sampler=predicate.sampler,
        description=f"Return the first element satisfying {predicate.name}, or zero.",
    )


def _comparison_tasks(profile_name: str = "", sampler: InputSampler | None = None) -> tuple[Task, ...]:
    xs = Param("xs", LIST_INT)
    k = Param("k", INT)

    if sampler is None:
        def input_sampler(rng: random.Random) -> tuple[Any, ...]:
            return (_rand_list(rng), rng.randint(-10, 10))
    else:
        input_sampler = _with_k_sampler(sampler, -10, 10)

    specs: tuple[tuple[str, Type, Oracle], ...] = (
        ("compare_sum_gt_k", BOOL, lambda xs, k: sum(xs) > k),
        ("compare_sum_eq_k", BOOL, lambda xs, k: sum(xs) == k),
        ("compare_len_gt_k", BOOL, lambda xs, k: len(xs) > k),
        ("compare_count_positive_gt_k", BOOL, lambda xs, k: sum(1 for x in xs if x > 0) > k),
        ("compare_count_negative_eq_k", BOOL, lambda xs, k: sum(1 for x in xs if x < 0) == k),
    )
    return tuple(
        _build_generated_task(
            name=_task_name("compare", name.removeprefix("compare_"), profile_name),
            params=(xs, k),
            return_type=return_type,
            oracle=oracle,
            sampler=input_sampler,
            description=f"Generated comparison task {name}.",
        )
        for name, return_type, oracle in specs
    )


def _build_generated_task(
    name: str,
    params: tuple[Param, ...],
    return_type: Type,
    oracle: Oracle,
    sampler: InputSampler,
    description: str,
) -> Task:
    return Task(
        name=name,
        signature=Signature("solve", params, return_type),
        visible_examples=_visible_examples(name, sampler, oracle),
        oracle=oracle,
        sampler=sampler,
        description=description,
    )


def _visible_examples(name: str, sampler: InputSampler, oracle: Oracle) -> tuple[Example, ...]:
    stable_task_seed = zlib.adler32(("visible:" + name).encode("utf-8"))
    rng = random.Random(stable_task_seed)
    examples: list[Example] = []
    seen: set[str] = set()
    for args in _boundary_args(sampler):
        key = repr(args)
        if key not in seen:
            seen.add(key)
            examples.append((args, oracle(*args)))
    attempts = 0
    while len(examples) < 5 and attempts < 100:
        attempts += 1
        args = sampler(rng)
        key = repr(args)
        if key in seen:
            continue
        seen.add(key)
        examples.append((args, oracle(*args)))
    return tuple(examples)


def _task_name(family: str, predicate: str, profile_name: str) -> str:
    if not profile_name:
        return f"{family}_{predicate}"
    return f"{family}_{predicate}_{profile_name}"


def _const_name(value: int) -> str:
    if value < 0:
        return f"neg{abs(value)}"
    return str(value)


def _sampler_profiles(seed: int) -> tuple[tuple[str, InputSampler], ...]:
    del seed
    return (
        ("short", lambda rng: (_rand_list(rng, max_len=4, low=-5, high=5),)),
        ("medium", lambda rng: (_rand_list(rng, max_len=8, low=-9, high=9),)),
        ("long", lambda rng: (_rand_list(rng, max_len=14, low=-9, high=9),)),
        ("wide", lambda rng: (_rand_list(rng, max_len=8, low=-20, high=20),)),
        ("zeros", lambda rng: (_rand_list(rng, max_len=10, low=-2, high=2),)),
    )


def _with_k_sampler(base_sampler: InputSampler, low: int, high: int) -> InputSampler:
    def sample(rng: random.Random) -> tuple[Any, ...]:
        base = base_sampler(rng)
        return base + (rng.randint(low, high),)

    return sample


def _procedural_split(name: str) -> str:
    return "eval" if zlib.adler32(("procedural:" + name).encode("utf-8")) % 5 == 0 else "train"


def _split_bucket(name: str) -> int:
    return zlib.adler32(("split:" + name).encode("utf-8")) % 5


def _boundary_args(sampler: InputSampler) -> tuple[tuple[Any, ...], ...]:
    empty_rng = random.Random(1)
    sample = sampler(empty_rng)
    if len(sample) == 1:
        return (((),), ((0,),), ((-1, 0, 1),))
    if len(sample) == 2:
        return (((), 0), ((0,), 0), ((-2, 0, 3), 0), ((-2, 0, 3), 2))
    return ()


def _rand_list(rng: random.Random, max_len: int = 8, low: int = -9, high: int = 9) -> tuple[int, ...]:
    if rng.random() < 0.15:
        return ()
    if rng.random() < 0.15:
        anchors = tuple(value for value in (-5, -1, 0, 1, 5) if low <= value <= high)
        choices: Sequence[int] = anchors if anchors else (low, 0, high)
        return tuple(rng.choice(choices) for _ in range(rng.randint(1, max_len)))
    return tuple(rng.randint(low, high) for _ in range(rng.randint(0, max_len)))
