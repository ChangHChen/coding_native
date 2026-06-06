from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|==|!=|>=|<=|//|[%+*<>\-]|\d+")


@dataclass(frozen=True)
class FeatureSpec:
    token_vocab: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        return {"token_vocab": list(self.token_vocab)}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "FeatureSpec":
        return cls(tuple(data["token_vocab"]))


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def fit_feature_spec(rows: Iterable[dict[str, Any]], max_tokens: int = 256) -> FeatureSpec:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(_token_features(row))
    vocab = tuple(token for token, _ in counts.most_common(max_tokens))
    return FeatureSpec(vocab)


def feature_names(spec: FeatureSpec) -> tuple[str, ...]:
    return _numeric_feature_names() + tuple(f"tok:{token}" for token in spec.token_vocab)


def extract_features(row: dict[str, Any], spec: FeatureSpec) -> list[float]:
    numeric = _numeric_features(row)
    token_counts = Counter(_token_features(row))
    token_values = [float(token_counts[token]) for token in spec.token_vocab]
    return numeric + token_values


def target_hidden_pass_rate(row: dict[str, Any]) -> float:
    return float(row["labels"]["hidden_pass_rate"])


def row_task_name(row: dict[str, Any]) -> str:
    return str(row["task"]["name"])


def row_visible_all_pass(row: dict[str, Any]) -> bool:
    return float(row["labels"]["visible_pass_rate"]) == 1.0


def row_solved(row: dict[str, Any]) -> bool:
    return bool(row["labels"]["solved"])


def _numeric_feature_names() -> tuple[str, ...]:
    return (
        "candidate_index_log",
        "return_is_int",
        "return_is_bool",
        "param_count",
        "int_param_count",
        "list_param_count",
        "program_body_size",
        "program_line_count",
        "visible_pass_rate",
        "visible_passes",
        "visible_total",
        "visible_all_pass",
        "visible_ok_rate",
        "visible_error_rate",
        "visible_trace_avg",
        "visible_trace_max",
        "visible_return_events",
        "visible_finish_events",
        "visible_assign_events",
        "visible_loop_events",
        "visible_branch_events",
        "visible_error_events",
        "int_abs_error_avg",
        "int_abs_error_max",
        "bool_wrong_count",
    )


def _numeric_features(row: dict[str, Any]) -> list[float]:
    signature = row["task"]["signature"]
    params = signature["params"]
    return_type = signature["return_type"]
    visible = row["visible"]
    traces = [case["run"] for case in visible]
    trace_lens = [float(run["trace_len"]) for run in traces]
    event_counts = Counter(
        event["kind"]
        for case in visible
        for event in case["run"].get("trace", ())
    )
    int_errors: list[float] = []
    bool_wrong = 0
    ok_count = 0
    error_count = 0
    for case in visible:
        run = case["run"]
        ok_count += int(bool(run["ok"]))
        error_count += int(run["error"] is not None)
        value = run.get("value")
        expected = case["expected"]
        if isinstance(value, int) and isinstance(expected, int) and not isinstance(value, bool):
            int_errors.append(abs(float(value - expected)))
        if isinstance(value, bool) and isinstance(expected, bool) and value != expected:
            bool_wrong += 1

    return [
        math.log1p(float(row["candidate_index"])),
        float(return_type == "int"),
        float(return_type == "bool"),
        float(len(params)),
        float(sum(param["type"] == "int" for param in params)),
        float(sum(param["type"] == "list[int]" for param in params)),
        float(row["program"]["body_size"]),
        float(row["program"]["rendered"].count("\n") + 1),
        float(row["labels"]["visible_pass_rate"]),
        float(row["labels"]["visible_passes"]),
        float(row["labels"]["visible_total"]),
        float(row["labels"]["visible_pass_rate"] == 1.0),
        ok_count / len(visible) if visible else 0.0,
        error_count / len(visible) if visible else 0.0,
        sum(trace_lens) / len(trace_lens) if trace_lens else 0.0,
        max(trace_lens) if trace_lens else 0.0,
        float(event_counts["return"]),
        float(event_counts["finish"]),
        float(event_counts["assign"]),
        float(event_counts["loop"]),
        float(event_counts["branch"]),
        float(event_counts["error"]),
        sum(int_errors) / len(int_errors) if int_errors else 0.0,
        max(int_errors) if int_errors else 0.0,
        float(bool_wrong),
    ]


def _token_features(row: dict[str, Any]) -> list[str]:
    tokens: list[str] = []
    task_name = row["task"]["name"]
    return_type = row["task"]["signature"]["return_type"]
    tokens.extend(f"task:{part}" for part in task_name.split("_"))
    tokens.append(f"return:{return_type}")
    for param in row["task"]["signature"]["params"]:
        tokens.append(f"param:{param['type']}")
    program = row["program"]["rendered"]
    tokens.extend(f"prog:{token}" for token in TOKEN_RE.findall(program))
    for case in row["visible"]:
        tokens.extend(f"trace:{event['kind']}" for event in case["run"].get("trace", ()))
    return tokens

