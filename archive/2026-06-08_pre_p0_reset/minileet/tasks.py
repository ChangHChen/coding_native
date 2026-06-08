from __future__ import annotations

import random
import zlib
from dataclasses import dataclass
from functools import lru_cache
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
        seen: set[str] = {repr(args) for args, _ in self.visible_examples}
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
class TaskMeta:
    family: str
    predicate: str
    operator: str
    profile: str
    has_k: bool


@dataclass(frozen=True)
class PredicateSpec:
    name: str
    params: tuple[Param, ...]
    fn: Callable[..., bool]
    sampler: InputSampler


SUITE_NAMES = (
    "starter",
    "generated",
    "train",
    "eval",
    "all",
    "procedural_train",
    "procedural_eval",
    "procedural_standard_train",
    "procedural_standard_eval",
    "procedural_hard_train",
    "procedural_hard_eval",
    "procedural_hard_dev_eval",
    "procedural_hard_final_eval",
    "procedural_bridge_train",
    "procedural_first_exact_bridge_train",
    "procedural_strict_bridge_train",
    "procedural_targeted_contrast_train",
    "procedural_scaled_train_50x",
    "procedural_scaled_final_eval_50x",
    "procedural_semantic_hard_train",
    "procedural_semantic_hard_dev_eval",
    "procedural_semantic_hard_final_eval",
    "procedural_semantic_hard_eval",
    "procedural_semantic_clean_train",
    "procedural_semantic_clean_dev_eval",
    "procedural_semantic_clean_final_eval",
)


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
    if suite == "procedural_standard_train":
        return procedural_standard_tasks(split="train")
    if suite == "procedural_standard_eval":
        return procedural_standard_tasks(split="eval")
    if suite == "procedural_hard_train":
        return procedural_hard_tasks(split="train")
    if suite == "procedural_hard_eval":
        return procedural_hard_tasks(split="eval")
    if suite == "procedural_hard_dev_eval":
        return procedural_hard_eval_sample(seed=7, count=100)
    if suite == "procedural_hard_final_eval":
        dev_names = {task.name for task in procedural_hard_eval_sample(seed=7, count=100)}
        return procedural_hard_eval_sample(seed=8, count=100, exclude_names=dev_names)
    if suite == "procedural_bridge_train":
        return procedural_bridge_tasks()
    if suite == "procedural_first_exact_bridge_train":
        return procedural_first_exact_bridge_tasks()
    if suite == "procedural_strict_bridge_train":
        return procedural_strict_bridge_tasks()
    if suite == "procedural_targeted_contrast_train":
        return procedural_targeted_contrast_tasks()
    if suite == "procedural_scaled_train_50x":
        return procedural_scaled_train_tasks(variants=50)
    if suite == "procedural_scaled_final_eval_50x":
        return procedural_scaled_final_eval_tasks(variants=50)
    if suite == "procedural_semantic_hard_train":
        return procedural_semantic_hard_train_tasks(variants=20)
    if suite == "procedural_semantic_hard_dev_eval":
        return procedural_semantic_hard_dev_eval_tasks()
    if suite == "procedural_semantic_hard_final_eval":
        return procedural_semantic_hard_eval_tasks()
    if suite == "procedural_semantic_hard_eval":
        return procedural_semantic_hard_eval_tasks()
    if suite == "procedural_semantic_clean_train":
        return procedural_semantic_clean_train_tasks(variants=20)
    if suite == "procedural_semantic_clean_dev_eval":
        return procedural_semantic_clean_eval_tasks(split="dev")
    if suite == "procedural_semantic_clean_final_eval":
        return procedural_semantic_clean_eval_tasks(split="final")
    raise ValueError(f"unknown task suite {suite}")


def all_procedural_tasks() -> tuple[Task, ...]:
    return procedural_tasks(split="train") + procedural_tasks(split="eval")


def procedural_standard_tasks(split: str, count: int = 0) -> tuple[Task, ...]:
    if split not in ("train", "eval"):
        raise ValueError(f"unknown standard split {split}")
    tasks = all_procedural_tasks()
    filtered = tuple(
        task
        for task in tasks
        if (_standard_split_bucket(task.name) != 0 if split == "train" else _standard_split_bucket(task.name) == 0)
    )
    if count > 0:
        return filtered[:count]
    return filtered


def task_meta(task_name: str) -> TaskMeta:
    profiles = _profile_names()
    parts = task_name.split("_")
    profile = parts[-1] if parts and parts[-1] in profiles else ""
    core_parts = parts[:-1] if profile else parts
    if core_parts[:3] == ["first", "or", "zero"]:
        family = "first_or_zero"
        predicate_parts = core_parts[3:]
    elif core_parts and core_parts[0] == "compare":
        family = "compare"
        predicate_parts = core_parts[1:]
    else:
        family = core_parts[0] if core_parts else "unknown"
        predicate_parts = core_parts[1:]

    predicate = "_".join(predicate_parts)
    operator = _predicate_operator(predicate_parts)
    return TaskMeta(
        family=family,
        predicate=predicate,
        operator=operator,
        profile=profile,
        has_k="k" in predicate_parts,
    )


def procedural_hard_tasks(split: str, count: int = 0) -> tuple[Task, ...]:
    if split not in ("train", "eval"):
        raise ValueError(f"unknown hard split {split}")
    heldout_families = {"first_or_zero"}
    heldout_ops = {"eq", "ne"}
    heldout_profiles = {"wide"}
    tasks = procedural_tasks(split="train") + procedural_tasks(split="eval")
    if split == "eval":
        filtered = tuple(
            task
            for task in tasks
            if _is_hard_eval_meta(task_meta(task.name), heldout_families, heldout_ops, heldout_profiles)
        )
    else:
        filtered = tuple(
            task
            for task in tasks
            if not _is_hard_eval_meta(task_meta(task.name), heldout_families, heldout_ops, heldout_profiles)
        )
    if count > 0:
        return filtered[:count]
    return filtered


def procedural_hard_eval_slice(start: int, stop: int) -> tuple[Task, ...]:
    return procedural_hard_tasks(split="eval")[start:stop]


def procedural_hard_eval_sample(seed: int, count: int, exclude_names: set[str] | None = None) -> tuple[Task, ...]:
    excluded = exclude_names or set()
    tasks = tuple(task for task in procedural_hard_tasks(split="eval") if task.name not in excluded)
    return tuple(random.Random(seed).sample(list(tasks), min(count, len(tasks))))


def procedural_bridge_tasks(count: int = 0) -> tuple[Task, ...]:
    """Bridge train tasks for held-out hard compositions.

    The hard split removes all first_or_zero, all eq/ne operators, and all wide
    profiles from training. This bridge suite restores partial compositions
    without mirroring the whole hard eval distribution:
    - first_or_zero with non-eq/ne predicates on non-wide profiles
    - eq/ne predicates for non-first_or_zero families on non-wide profiles
    """
    tasks = procedural_tasks(split="train") + procedural_tasks(split="eval")
    filtered = tuple(
        task
        for task in tasks
        if _is_bridge_meta(task_meta(task.name))
    )
    if count > 0:
        return filtered[:count]
    return filtered


def procedural_first_exact_bridge_tasks(count: int = 0) -> tuple[Task, ...]:
    tasks: list[Task] = []
    for profile_name, sampler in _first_exact_bridge_profiles():
        for predicate in _procedural_predicate_specs(sampler):
            tasks.append(_make_first_or_zero_task(predicate, profile_name))
    result = tuple(tasks)
    if count > 0:
        return result[:count]
    return result


def procedural_strict_bridge_tasks(count: int = 0) -> tuple[Task, ...]:
    """Bridge held-out compositions without reusing hard-eval task names."""
    tasks: list[Task] = []
    for profile_name, sampler in _first_exact_bridge_profiles():
        for predicate in _procedural_predicate_specs(sampler):
            operator = _predicate_operator(predicate.name.split("_"))
            tasks.append(_make_first_or_zero_task(predicate, profile_name))
            if operator in {"eq", "ne"}:
                tasks.extend(
                    [
                        _make_count_task(predicate, profile_name),
                        _make_sum_task(predicate, profile_name),
                        _make_any_task(predicate, profile_name),
                        _make_all_task(predicate, profile_name),
                    ]
                )
        tasks.extend(
            task
            for task in _comparison_tasks(profile_name, sampler)
            if task_meta(task.name).operator in {"eq", "ne"}
        )
    result = tuple(tasks)
    if count > 0:
        return result[:count]
    return result


def procedural_targeted_contrast_tasks(count: int = 0) -> tuple[Task, ...]:
    """Focused train tasks for boundary and exact-predicate selection errors."""
    tasks: list[Task] = []
    for profile_name, sampler in _targeted_contrast_profiles():
        for predicate in _procedural_predicate_specs(sampler):
            operator = _predicate_operator(predicate.name.split("_"))
            if operator in {"gt", "ge", "lt", "le", "eq", "ne"}:
                tasks.append(_make_first_or_zero_task(predicate, profile_name))
            if operator in {"eq", "ne"}:
                tasks.extend(
                    [
                        _make_sum_task(predicate, profile_name),
                        _make_all_task(predicate, profile_name),
                    ]
                )
    result = tuple(tasks)
    if count > 0:
        return result[:count]
    return result


def procedural_scaled_train_tasks(variants: int, count: int = 0) -> tuple[Task, ...]:
    tasks: list[Task] = []
    for variant in range(variants):
        for profile_name, sampler in _scaled_base_profiles(variant, prefix="tr"):
            for predicate in _procedural_predicate_specs(sampler):
                operator = _predicate_operator(predicate.name.split("_"))
                if operator not in {"eq", "ne"}:
                    tasks.extend(
                        [
                            _make_count_task(predicate, profile_name),
                            _make_sum_task(predicate, profile_name),
                            _make_any_task(predicate, profile_name),
                            _make_all_task(predicate, profile_name),
                        ]
                    )
            tasks.extend(
                task
                for task in _comparison_tasks(profile_name, sampler)
                if task_meta(task.name).operator not in {"eq", "ne"}
            )
        for profile_name, sampler in _scaled_bridge_profiles(variant, prefix="tr"):
            for predicate in _procedural_predicate_specs(sampler):
                operator = _predicate_operator(predicate.name.split("_"))
                tasks.append(_make_first_or_zero_task(predicate, profile_name))
                if operator in {"eq", "ne"}:
                    tasks.extend(
                        [
                            _make_count_task(predicate, profile_name),
                            _make_sum_task(predicate, profile_name),
                            _make_any_task(predicate, profile_name),
                            _make_all_task(predicate, profile_name),
                        ]
                    )
            tasks.extend(
                task
                for task in _comparison_tasks(profile_name, sampler)
                if task_meta(task.name).operator in {"eq", "ne"}
            )
        for profile_name, sampler in _scaled_contrast_profiles(variant, prefix="tr"):
            for predicate in _procedural_predicate_specs(sampler):
                operator = _predicate_operator(predicate.name.split("_"))
                if operator in {"gt", "ge", "lt", "le", "eq", "ne"}:
                    tasks.append(_make_first_or_zero_task(predicate, profile_name))
                if operator in {"eq", "ne"}:
                    tasks.extend(
                        [
                            _make_sum_task(predicate, profile_name),
                            _make_all_task(predicate, profile_name),
                        ]
                    )
    result = tuple(tasks)
    if count > 0:
        return result[:count]
    return result


def procedural_scaled_final_eval_tasks(variants: int, count: int = 0) -> tuple[Task, ...]:
    base_tasks = procedural_hard_final_eval_sample()
    tasks: list[Task] = []
    for variant in range(variants):
        profile_samplers = dict(_scaled_eval_profiles(variant, prefix="ev"))
        profile_names = _scaled_eval_profile_names(variant, prefix="ev")
        for task in base_tasks:
            meta = task_meta(task.name)
            profile_name = profile_names[meta.profile]
            sampler = profile_samplers[profile_name]
            tasks.append(_make_task_from_parts(meta.family, meta.predicate, profile_name, sampler))
    result = tuple(tasks)
    if count > 0:
        return result[:count]
    return result


def procedural_hard_final_eval_sample() -> tuple[Task, ...]:
    dev_names = {task.name for task in procedural_hard_eval_sample(seed=7, count=100)}
    return procedural_hard_eval_sample(seed=8, count=100, exclude_names=dev_names)


def procedural_semantic_hard_eval_tasks(count: int = 0) -> tuple[Task, ...]:
    result = _semantic_hard_tasks(
        one_list=_semantic_list_sampler(max_len=12, low=-8, high=8),
        pair_list=_semantic_pair_sampler(max_len=10, low=-8, high=8),
        profile_name="",
    )
    if count > 0:
        return result[:count]
    return result


def procedural_semantic_hard_train_tasks(variants: int, count: int = 0) -> tuple[Task, ...]:
    tasks: list[Task] = []
    for variant in range(variants):
        shift = (variant % 7) - 3
        width = variant % 5
        profile_name = f"shtr{variant:02d}"
        tasks.extend(
            _semantic_hard_tasks(
                one_list=_semantic_list_sampler(max_len=9 + width, low=-7 + shift, high=7 + shift),
                pair_list=_semantic_pair_sampler(max_len=8 + width, low=-7 + shift, high=7 + shift),
                profile_name=profile_name,
            )
        )
    result = tuple(tasks)
    if count > 0:
        return result[:count]
    return result


def procedural_semantic_hard_dev_eval_tasks(count: int = 0) -> tuple[Task, ...]:
    result = _semantic_hard_tasks(
        one_list=_semantic_list_sampler(max_len=11, low=-9, high=7),
        pair_list=_semantic_pair_sampler(max_len=9, low=-9, high=7),
        profile_name="shdev00",
    )
    if count > 0:
        return result[:count]
    return result


def procedural_semantic_clean_train_tasks(variants: int, count: int = 0) -> tuple[Task, ...]:
    tasks: list[Task] = []
    for variant in range(variants):
        shift = (variant % 7) - 3
        width = variant % 5
        profile_name = f"sctr{variant:02d}"
        tasks.extend(
            _semantic_clean_tasks(
                one_list=_semantic_list_sampler(max_len=9 + width, low=-7 + shift, high=7 + shift),
                pair_list=_semantic_pair_sampler(max_len=8 + width, low=-7 + shift, high=7 + shift),
                profile_name=profile_name,
                split="train",
            )
        )
    result = tuple(tasks)
    if count > 0:
        return result[:count]
    return result


def procedural_semantic_clean_eval_tasks(split: str, count: int = 0) -> tuple[Task, ...]:
    if split == "dev":
        result = _semantic_clean_tasks(
            one_list=_semantic_list_sampler(max_len=11, low=-9, high=7),
            pair_list=_semantic_pair_sampler(max_len=9, low=-9, high=7),
            profile_name="scdev00",
            split="dev",
        )
    elif split == "final":
        result = _semantic_clean_tasks(
            one_list=_semantic_list_sampler(max_len=12, low=-8, high=8),
            pair_list=_semantic_pair_sampler(max_len=10, low=-8, high=8),
            profile_name="scfin00",
            split="final",
        )
    else:
        raise ValueError(f"unknown clean semantic split {split}")
    if count > 0:
        return result[:count]
    return result


def _semantic_hard_tasks(one_list: InputSampler, pair_list: InputSampler, profile_name: str) -> tuple[Task, ...]:
    predicates = _semantic_predicate_specs(one_list)
    tasks: list[Task] = []

    for index in range(10):
        tasks.append(_make_nested_sum_after_task(predicates[index], predicates[index + 10], profile_name))
        tasks.append(_make_nested_count_before_task(predicates[index + 5], predicates[index + 15], profile_name))

    for index in range(20):
        tasks.append(
            _make_twostat_sum_minus_count_task(
                predicates[index],
                predicates[(index + 7) % len(predicates)],
                profile_name,
            )
        )

    for index in range(10):
        tasks.append(_make_first_index_task(predicates[index], profile_name))
        tasks.append(_make_last_index_task(predicates[index + 10], profile_name))

    for index in range(10):
        tasks.append(_make_predicate_max_task(predicates[index], profile_name))
        tasks.append(_make_predicate_min_task(predicates[index + 10], profile_name))

    for index in range(4):
        tasks.extend(_pair_comparison_tasks(pair_list, suffix=str(index), profile_name=profile_name))

    return tuple(tasks)


def _semantic_clean_tasks(
    one_list: InputSampler,
    pair_list: InputSampler,
    profile_name: str,
    split: str,
) -> tuple[Task, ...]:
    predicates = _semantic_predicate_specs(one_list)
    if split == "train":
        base = 0
        pair_suffix_start = 0
        split_id = 0
    elif split == "dev":
        base = 10
        pair_suffix_start = 4
        split_id = 1
    elif split == "final":
        base = 20
        pair_suffix_start = 8
        split_id = 2
    else:
        raise ValueError(f"unknown semantic clean split {split}")

    def pred(offset: int) -> PredicateSpec:
        return predicates[offset % len(predicates)]

    tasks: list[Task] = []
    for index in range(10):
        tasks.append(_make_nested_sum_after_task(pred(base + index), pred(base + index + 12), profile_name))
        tasks.append(_make_nested_count_before_task(pred(base + index + 10), pred(base + index + 22), profile_name))

    for index in range(20):
        tasks.append(
            _make_twostat_sum_minus_count_task(
                pred(index + split_id * 20),
                pred(index + split_id),
                profile_name,
            )
        )

    for index in range(10):
        tasks.append(_make_first_index_task(pred(base + index), profile_name))
        tasks.append(_make_last_index_task(pred(base + index + 10), profile_name))

    for index in range(10):
        tasks.append(_make_predicate_max_task(pred(base + index), profile_name))
        tasks.append(_make_predicate_min_task(pred(base + index + 10), profile_name))

    for index in range(pair_suffix_start, pair_suffix_start + 4):
        tasks.extend(_pair_comparison_tasks(pair_list, suffix=str(index), profile_name=profile_name))

    return tuple(tasks)


def _is_bridge_meta(meta: TaskMeta) -> bool:
    if meta.profile == "wide":
        return False
    if meta.family == "first_or_zero" and meta.operator not in {"eq", "ne"}:
        return True
    if meta.family != "first_or_zero" and meta.operator in {"eq", "ne"}:
        return True
    return False


def _is_hard_eval_meta(
    meta: TaskMeta,
    heldout_families: set[str],
    heldout_ops: set[str],
    heldout_profiles: set[str],
) -> bool:
    return meta.family in heldout_families or meta.operator in heldout_ops or meta.profile in heldout_profiles


def _predicate_operator(predicate_parts: list[str]) -> str:
    if not predicate_parts:
        return "none"
    first = predicate_parts[0]
    if first in {"gt", "ge", "lt", "le", "eq", "ne"}:
        return first
    if first in {"even", "odd"}:
        return "parity"
    if first in {"positive", "nonnegative", "negative", "zero"}:
        return first
    if first == "sum" and len(predicate_parts) > 1:
        return _predicate_operator(predicate_parts[1:])
    if first == "count" and len(predicate_parts) > 1:
        return _predicate_operator(predicate_parts[1:])
    return first


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


def _make_task_from_parts(family: str, predicate_name: str, profile_name: str, sampler: InputSampler) -> Task:
    if family == "compare":
        for task in _comparison_tasks(profile_name, sampler):
            if task_meta(task.name).predicate == predicate_name:
                return task
        raise ValueError(f"unknown compare predicate {predicate_name}")
    predicates = {predicate.name: predicate for predicate in _procedural_predicate_specs(sampler)}
    predicate = predicates[predicate_name]
    if family == "count":
        return _make_count_task(predicate, profile_name)
    if family == "sum":
        return _make_sum_task(predicate, profile_name)
    if family == "any":
        return _make_any_task(predicate, profile_name)
    if family == "all":
        return _make_all_task(predicate, profile_name)
    if family == "first_or_zero":
        return _make_first_or_zero_task(predicate, profile_name)
    raise ValueError(f"unknown task family {family}")


def _make_nested_sum_after_task(
    value_predicate: PredicateSpec,
    gate_predicate: PredicateSpec,
    profile_name: str = "",
) -> Task:
    def oracle(*args: Any, value_pred: PredicateSpec = value_predicate, gate_pred: PredicateSpec = gate_predicate) -> int:
        xs = args[0]
        total = 0
        enabled = False
        for x in xs:
            if gate_pred.fn(x):
                enabled = True
            if enabled and value_pred.fn(x):
                total += x
        return total

    return _build_generated_task(
        name=_semantic_task_name(f"nested_sum_{value_predicate.name}_after_{gate_predicate.name}", profile_name),
        params=value_predicate.params,
        return_type=INT,
        oracle=oracle,
        sampler=value_predicate.sampler,
        description="Return the sum of values satisfying one predicate after another predicate has appeared.",
    )


def _make_nested_count_before_task(
    value_predicate: PredicateSpec,
    stop_predicate: PredicateSpec,
    profile_name: str = "",
) -> Task:
    def oracle(*args: Any, value_pred: PredicateSpec = value_predicate, stop_pred: PredicateSpec = stop_predicate) -> int:
        xs = args[0]
        total = 0
        stopped = False
        for x in xs:
            if stopped:
                continue
            if stop_pred.fn(x):
                stopped = True
            elif value_pred.fn(x):
                total += 1
        return total

    return _build_generated_task(
        name=_semantic_task_name(f"nested_count_{value_predicate.name}_before_{stop_predicate.name}", profile_name),
        params=value_predicate.params,
        return_type=INT,
        oracle=oracle,
        sampler=value_predicate.sampler,
        description="Return how many values satisfy one predicate before a stop predicate first appears.",
    )


def _make_twostat_sum_minus_count_task(
    sum_predicate: PredicateSpec,
    count_predicate: PredicateSpec,
    profile_name: str = "",
) -> Task:
    def oracle(*args: Any, sum_pred: PredicateSpec = sum_predicate, count_pred: PredicateSpec = count_predicate) -> int:
        xs = args[0]
        selected_sum = sum(x for x in xs if sum_pred.fn(x))
        selected_count = sum(1 for x in xs if count_pred.fn(x))
        return selected_sum - selected_count

    return _build_generated_task(
        name=_semantic_task_name(f"twostat_sum_{sum_predicate.name}_minus_count_{count_predicate.name}", profile_name),
        params=sum_predicate.params,
        return_type=INT,
        oracle=oracle,
        sampler=sum_predicate.sampler,
        description="Return a sum over one predicate minus a count over another predicate.",
    )


def _make_first_index_task(predicate: PredicateSpec, profile_name: str = "") -> Task:
    def oracle(*args: Any, pred: PredicateSpec = predicate) -> int:
        xs = args[0]
        for index, x in enumerate(xs):
            if pred.fn(x):
                return index
        return -1

    return _build_generated_task(
        name=_semantic_task_name(f"idx_first_{predicate.name}", profile_name),
        params=predicate.params,
        return_type=INT,
        oracle=oracle,
        sampler=predicate.sampler,
        description="Return the first index whose value satisfies the predicate, or -1.",
    )


def _make_last_index_task(predicate: PredicateSpec, profile_name: str = "") -> Task:
    def oracle(*args: Any, pred: PredicateSpec = predicate) -> int:
        xs = args[0]
        found = -1
        for index, x in enumerate(xs):
            if pred.fn(x):
                found = index
        return found

    return _build_generated_task(
        name=_semantic_task_name(f"idx_last_{predicate.name}", profile_name),
        params=predicate.params,
        return_type=INT,
        oracle=oracle,
        sampler=predicate.sampler,
        description="Return the last index whose value satisfies the predicate, or -1.",
    )


def _make_predicate_max_task(predicate: PredicateSpec, profile_name: str = "") -> Task:
    def oracle(*args: Any, pred: PredicateSpec = predicate) -> int:
        xs = args[0]
        values = [x for x in xs if pred.fn(x)]
        return max(values) if values else 0

    return _build_generated_task(
        name=_semantic_task_name(f"predmax_{predicate.name}", profile_name),
        params=predicate.params,
        return_type=INT,
        oracle=oracle,
        sampler=predicate.sampler,
        description="Return the maximum value satisfying the predicate, or zero.",
    )


def _make_predicate_min_task(predicate: PredicateSpec, profile_name: str = "") -> Task:
    def oracle(*args: Any, pred: PredicateSpec = predicate) -> int:
        xs = args[0]
        values = [x for x in xs if pred.fn(x)]
        return min(values) if values else 0

    return _build_generated_task(
        name=_semantic_task_name(f"predmin_{predicate.name}", profile_name),
        params=predicate.params,
        return_type=INT,
        oracle=oracle,
        sampler=predicate.sampler,
        description="Return the minimum value satisfying the predicate, or zero.",
    )


def _pair_comparison_tasks(
    sampler: InputSampler,
    suffix: str,
    profile_name: str = "",
) -> tuple[Task, ...]:
    xs = Param("xs", LIST_INT)
    ys = Param("ys", LIST_INT)
    params = (xs, ys)
    specs: tuple[tuple[str, Type, Oracle], ...] = (
        ("sum_xs_gt_sum_ys", BOOL, lambda xs, ys: sum(xs) > sum(ys)),
        ("len_xs_eq_len_ys", BOOL, lambda xs, ys: len(xs) == len(ys)),
        (
            "count_pos_xs_gt_count_pos_ys",
            BOOL,
            lambda xs, ys: sum(1 for x in xs if x > 0) > sum(1 for y in ys if y > 0),
        ),
        ("first_xs_gt_first_ys", BOOL, lambda xs, ys: (xs[0] if xs else 0) > (ys[0] if ys else 0)),
        ("sum_absdiff_or_zero", INT, lambda xs, ys: sum(abs(x - y) for x, y in zip(xs, ys))),
    )
    return tuple(
        _build_generated_task(
            name=_semantic_task_name(f"paircmp_{name}_{suffix}", profile_name),
            params=params,
            return_type=return_type,
            oracle=oracle,
            sampler=sampler,
            description="Generated pair/list comparison task.",
        )
        for name, return_type, oracle in specs
    )


def _semantic_task_name(base: str, profile_name: str) -> str:
    if not profile_name:
        return base
    return f"{base}_{profile_name}"


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


def _first_exact_bridge_profiles() -> tuple[tuple[str, InputSampler], ...]:
    return (
        ("bshort", lambda rng: (_rand_list(rng, max_len=5, low=-6, high=6),)),
        ("bmedium", lambda rng: (_rand_list(rng, max_len=9, low=-10, high=10),)),
        ("blong", lambda rng: (_rand_list(rng, max_len=16, low=-10, high=10),)),
        ("bzeros", lambda rng: (_rand_list(rng, max_len=12, low=-3, high=3),)),
    )


def _targeted_contrast_profiles() -> tuple[tuple[str, InputSampler], ...]:
    return (
        ("cshort", lambda rng: (_contrast_list(rng, max_len=5, low=-5, high=5),)),
        ("cmedium", lambda rng: (_contrast_list(rng, max_len=9, low=-7, high=7),)),
        ("clong", lambda rng: (_contrast_list(rng, max_len=15, low=-7, high=7),)),
        ("cwide", lambda rng: (_contrast_list(rng, max_len=9, low=-12, high=12),)),
    )


def _semantic_predicate_specs(sampler: InputSampler) -> tuple[PredicateSpec, ...]:
    xs = Param("xs", LIST_INT)
    specs: list[PredicateSpec] = [
        PredicateSpec("even", (xs,), lambda x: x % 2 == 0, sampler),
        PredicateSpec("odd", (xs,), lambda x: x % 2 != 0, sampler),
    ]
    for const in (-2, -1, 0, 1, 2):
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
    return tuple(specs)


def _semantic_list_sampler(max_len: int, low: int, high: int) -> InputSampler:
    def sample(rng: random.Random) -> tuple[Any, ...]:
        return (_contrast_list(rng, max_len=max_len, low=low, high=high),)

    return sample


def _semantic_pair_sampler(max_len: int, low: int, high: int) -> InputSampler:
    def sample(rng: random.Random) -> tuple[Any, ...]:
        return (
            _contrast_list(rng, max_len=max_len, low=low, high=high),
            _contrast_list(rng, max_len=max_len, low=low, high=high),
        )

    return sample


def _scaled_base_profiles(variant: int, prefix: str) -> tuple[tuple[str, InputSampler], ...]:
    shift = (variant % 7) - 3
    width = variant % 4
    return (
        (f"{prefix}{variant:02d}s", lambda rng, shift=shift, width=width: (_rand_list(rng, max_len=4 + width, low=-5 + shift, high=5 + shift),)),
        (f"{prefix}{variant:02d}m", lambda rng, shift=shift, width=width: (_rand_list(rng, max_len=8 + width, low=-9 + shift, high=9 + shift),)),
        (f"{prefix}{variant:02d}l", lambda rng, shift=shift, width=width: (_rand_list(rng, max_len=14 + width, low=-9 + shift, high=9 + shift),)),
        (f"{prefix}{variant:02d}z", lambda rng, shift=shift, width=width: (_rand_list(rng, max_len=10 + width, low=-2 + shift, high=2 + shift),)),
    )


def _scaled_bridge_profiles(variant: int, prefix: str) -> tuple[tuple[str, InputSampler], ...]:
    shift = (variant % 7) - 3
    width = variant % 4
    return (
        (f"{prefix}{variant:02d}bs", lambda rng, shift=shift, width=width: (_rand_list(rng, max_len=5 + width, low=-6 + shift, high=6 + shift),)),
        (f"{prefix}{variant:02d}bm", lambda rng, shift=shift, width=width: (_rand_list(rng, max_len=9 + width, low=-10 + shift, high=10 + shift),)),
        (f"{prefix}{variant:02d}bl", lambda rng, shift=shift, width=width: (_rand_list(rng, max_len=16 + width, low=-10 + shift, high=10 + shift),)),
        (f"{prefix}{variant:02d}bz", lambda rng, shift=shift, width=width: (_rand_list(rng, max_len=12 + width, low=-3 + shift, high=3 + shift),)),
    )


def _scaled_contrast_profiles(variant: int, prefix: str) -> tuple[tuple[str, InputSampler], ...]:
    shift = (variant % 7) - 3
    width = variant % 4
    return (
        (f"{prefix}{variant:02d}cs", lambda rng, shift=shift, width=width: (_contrast_list(rng, max_len=5 + width, low=-5 + shift, high=5 + shift),)),
        (f"{prefix}{variant:02d}cm", lambda rng, shift=shift, width=width: (_contrast_list(rng, max_len=9 + width, low=-7 + shift, high=7 + shift),)),
        (f"{prefix}{variant:02d}cl", lambda rng, shift=shift, width=width: (_contrast_list(rng, max_len=15 + width, low=-7 + shift, high=7 + shift),)),
        (f"{prefix}{variant:02d}cw", lambda rng, shift=shift, width=width: (_contrast_list(rng, max_len=9 + width, low=-12 + shift, high=12 + shift),)),
    )


def _scaled_eval_profiles(variant: int, prefix: str) -> tuple[tuple[str, InputSampler], ...]:
    names = _scaled_eval_profile_names(variant, prefix)
    shift = (variant % 7) - 3
    width = variant % 4
    return (
        (names["short"], lambda rng, shift=shift, width=width: (_rand_list(rng, max_len=4 + width, low=-5 + shift, high=5 + shift),)),
        (names["medium"], lambda rng, shift=shift, width=width: (_rand_list(rng, max_len=8 + width, low=-9 + shift, high=9 + shift),)),
        (names["long"], lambda rng, shift=shift, width=width: (_rand_list(rng, max_len=14 + width, low=-9 + shift, high=9 + shift),)),
        (names["wide"], lambda rng, shift=shift, width=width: (_rand_list(rng, max_len=8 + width, low=-20 + shift, high=20 + shift),)),
        (names["zeros"], lambda rng, shift=shift, width=width: (_rand_list(rng, max_len=10 + width, low=-2 + shift, high=2 + shift),)),
    )


def _scaled_eval_profile_names(variant: int, prefix: str) -> dict[str, str]:
    return {
        "short": f"{prefix}{variant:02d}s",
        "medium": f"{prefix}{variant:02d}m",
        "long": f"{prefix}{variant:02d}l",
        "wide": f"{prefix}{variant:02d}w",
        "zeros": f"{prefix}{variant:02d}z",
    }


@lru_cache(maxsize=1)
def _profile_names() -> frozenset[str]:
    scaled_names: set[str] = set()
    for variant in range(50):
        scaled_names.update(name for name, _ in _scaled_base_profiles(variant, prefix="tr"))
        scaled_names.update(name for name, _ in _scaled_bridge_profiles(variant, prefix="tr"))
        scaled_names.update(name for name, _ in _scaled_contrast_profiles(variant, prefix="tr"))
        scaled_names.update(name for name, _ in _scaled_eval_profiles(variant, prefix="ev"))
    semantic_names = (
        {f"shtr{variant:02d}" for variant in range(50)}
        | {f"sctr{variant:02d}" for variant in range(50)}
        | {"shdev00", "scdev00", "scfin00"}
    )
    return frozenset(
        {name for name, _ in _sampler_profiles(0)}
        | {name for name, _ in _first_exact_bridge_profiles()}
        | {name for name, _ in _targeted_contrast_profiles()}
        | scaled_names
        | semantic_names
    )


def _contrast_list(rng: random.Random, max_len: int, low: int, high: int) -> tuple[int, ...]:
    if rng.random() < 0.08:
        return ()
    anchors = tuple(value for value in range(-4, 5) if low <= value <= high)
    if rng.random() < 0.65 and anchors:
        pivot_choices = tuple(value for value in range(-3, 4) if low < value < high)
        if not pivot_choices:
            return tuple(rng.choice(anchors) for _ in range(rng.randint(1, max_len)))
        pivot = rng.choice(pivot_choices)
        patterns = (
            (pivot, pivot + 1, pivot - 1),
            (pivot + 1, pivot, pivot - 1),
            (pivot - 1, pivot, pivot + 1),
            (pivot, pivot, pivot + 1, pivot - 1),
            (pivot + 2, pivot - 2, pivot, pivot + 1),
        )
        values = [value for value in rng.choice(patterns) if low <= value <= high]
        target_len = rng.randint(1, max_len)
        while len(values) < target_len:
            values.append(rng.choice(anchors))
        return tuple(values[:max_len])
    return tuple(rng.randint(low, high) for _ in range(rng.randint(1, max_len)))


def _with_k_sampler(base_sampler: InputSampler, low: int, high: int) -> InputSampler:
    def sample(rng: random.Random) -> tuple[Any, ...]:
        base = base_sampler(rng)
        return base + (rng.randint(low, high),)

    return sample


def _procedural_split(name: str) -> str:
    return "eval" if zlib.adler32(("procedural:" + name).encode("utf-8")) % 5 == 0 else "train"


def _standard_split_bucket(name: str) -> int:
    return zlib.adler32(("standard:" + name).encode("utf-8")) % 5


def _split_bucket(name: str) -> int:
    return zlib.adler32(("split:" + name).encode("utf-8")) % 5


def _boundary_args(sampler: InputSampler) -> tuple[tuple[Any, ...], ...]:
    empty_rng = random.Random(1)
    sample = sampler(empty_rng)
    if len(sample) == 1:
        return (((),), ((0,),), ((-1, 0, 1),))
    if len(sample) == 2:
        if isinstance(sample[1], tuple):
            return (((), ()), ((0,), ()), ((), (0,)), ((-2, 0, 3), (1, -1)))
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
