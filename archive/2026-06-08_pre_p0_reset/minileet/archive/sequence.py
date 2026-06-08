from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

from minileet.features import TOKEN_RE


PAD = "<pad>"
UNK = "<unk>"
CLS = "<cls>"
SEP = "<sep>"


@dataclass(frozen=True)
class TokenSpec:
    vocab: tuple[str, ...]

    @property
    def stoi(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self.vocab)}

    def to_json(self) -> dict[str, Any]:
        return {"vocab": list(self.vocab)}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "TokenSpec":
        return cls(tuple(data["vocab"]))


def fit_token_spec(rows: Iterable[dict[str, Any]], max_vocab: int = 4096) -> TokenSpec:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(row_to_tokens(row))
    base = [PAD, UNK, CLS, SEP]
    vocab = base + [token for token, _ in counts.most_common(max_vocab - len(base)) if token not in base]
    return TokenSpec(tuple(vocab))


def encode_row(row: dict[str, Any], spec: TokenSpec, max_len: int) -> tuple[list[int], list[int]]:
    stoi = spec.stoi
    tokens = row_to_tokens(row)
    return encode_tokens(tokens, spec, max_len)


def encode_tokens(tokens: list[str], spec: TokenSpec, max_len: int) -> tuple[list[int], list[int]]:
    stoi = spec.stoi
    ids = [stoi[CLS]]
    ids.extend(stoi.get(token, stoi[UNK]) for token in tokens[: max_len - 1])
    if len(ids) < max_len:
        ids.extend([stoi[PAD]] * (max_len - len(ids)))
    mask = [int(token_id != stoi[PAD]) for token_id in ids]
    return ids, mask


def fit_structured_token_spec(rows: Iterable[dict[str, Any]], max_vocab: int = 4096) -> TokenSpec:
    counts: Counter[str] = Counter()
    for row in rows:
        structured = row_to_structured_tokens(row)
        counts.update(structured["task"])
        counts.update(structured["program"])
        for test_tokens in structured["tests"]:
            counts.update(test_tokens)
    base = [PAD, UNK, CLS, SEP]
    vocab = base + [token for token, _ in counts.most_common(max_vocab - len(base)) if token not in base]
    return TokenSpec(tuple(vocab))


def row_to_structured_tokens(row: dict[str, Any]) -> dict[str, Any]:
    task = row["task"]
    signature = task["signature"]
    labels = row["labels"]

    task_tokens = ["section:task"]
    task_tokens.extend(f"task:{part}" for part in task["name"].split("_"))
    task_tokens.append(f"return:{signature['return_type']}")
    for param in signature["params"]:
        task_tokens.append(f"param_name:{param['name']}")
        task_tokens.append(f"param_type:{param['type']}")

    program_tokens = ["section:program"]
    program_tokens.extend(f"prog:{token}" for token in TOKEN_RE.findall(row["program"]["rendered"]))
    program_tokens.append(f"program_body_size:{_small_bucket(row['program']['body_size'])}")
    program_tokens.append(f"candidate_bucket:{_small_bucket(row['candidate_index'])}")
    program_tokens.append(f"visible_passes:{labels['visible_passes']}")
    program_tokens.append(f"visible_total:{labels['visible_total']}")
    program_tokens.append(f"visible_all_pass:{labels['visible_pass_rate'] == 1.0}")

    test_tokens = []
    for case_index, case in enumerate(row["visible"]):
        tokens = ["section:test", f"case:{case_index}", f"case_passed:{case['passed']}"]
        tokens.extend(_value_tokens("arg", case["args"]))
        tokens.extend(_value_tokens("expected", case["expected"]))
        run = case["run"]
        tokens.append(f"run_ok:{run['ok']}")
        if run["error"] is not None:
            tokens.append("run_error:" + str(run["error"]).split(":", 1)[0])
        tokens.extend(_value_tokens("value", run["value"]))
        tokens.append(f"trace_len:{_small_bucket(run['trace_len'])}")
        for event in run.get("trace", ()):
            tokens.append("trace:" + event["kind"])
            for key, value in sorted(event.get("data", {}).items()):
                tokens.extend(_value_tokens(f"trace_{key}", value, max_items=3))
        test_tokens.append(tokens)

    return {"task": task_tokens, "program": program_tokens, "tests": test_tokens}


def row_to_tokens(row: dict[str, Any]) -> list[str]:
    tokens: list[str] = []
    task = row["task"]
    signature = task["signature"]
    labels = row["labels"]

    tokens.append("section:task")
    tokens.extend(f"task:{part}" for part in task["name"].split("_"))
    tokens.append(f"return:{signature['return_type']}")
    for param in signature["params"]:
        tokens.append(f"param_name:{param['name']}")
        tokens.append(f"param_type:{param['type']}")

    tokens.append(SEP)
    tokens.append("section:program")
    tokens.extend(f"prog:{token}" for token in TOKEN_RE.findall(row["program"]["rendered"]))
    tokens.append(f"program_body_size:{_small_bucket(row['program']['body_size'])}")
    tokens.append(f"candidate_bucket:{_small_bucket(row['candidate_index'])}")

    tokens.append(SEP)
    tokens.append("section:visible_summary")
    tokens.append(f"visible_passes:{labels['visible_passes']}")
    tokens.append(f"visible_total:{labels['visible_total']}")
    tokens.append(f"visible_all_pass:{labels['visible_pass_rate'] == 1.0}")

    for case_index, case in enumerate(row["visible"]):
        tokens.append(SEP)
        tokens.append(f"case:{case_index}")
        tokens.append(f"case_passed:{case['passed']}")
        tokens.extend(_value_tokens("arg", case["args"]))
        tokens.extend(_value_tokens("expected", case["expected"]))
        run = case["run"]
        tokens.append(f"run_ok:{run['ok']}")
        if run["error"] is not None:
            tokens.append("run_error:" + str(run["error"]).split(":", 1)[0])
        tokens.extend(_value_tokens("value", run["value"]))
        tokens.append(f"trace_len:{_small_bucket(run['trace_len'])}")
        for event in run.get("trace", ()):
            tokens.append("trace:" + event["kind"])
            for key, value in sorted(event.get("data", {}).items()):
                tokens.extend(_value_tokens(f"trace_{key}", value, max_items=3))

    return tokens


def _value_tokens(prefix: str, value: Any, max_items: int = 8) -> list[str]:
    if isinstance(value, bool):
        return [f"{prefix}:bool:{value}"]
    if isinstance(value, int):
        return [f"{prefix}:int:{_int_bucket(value)}"]
    if isinstance(value, list):
        tokens = [f"{prefix}:list_len:{_small_bucket(len(value))}"]
        for item in value[:max_items]:
            tokens.extend(_value_tokens(prefix + "_item", item, max_items=max_items))
        return tokens
    if value is None:
        return [f"{prefix}:none"]
    return [f"{prefix}:other"]


def _int_bucket(value: int) -> str:
    if -10 <= value <= 10:
        return str(value)
    if value < -10:
        return "lt_neg10"
    return "gt_10"


def _small_bucket(value: int) -> str:
    if value <= 0:
        return "0"
    if value <= 1:
        return "1"
    if value <= 2:
        return "2"
    if value <= 4:
        return "3_4"
    if value <= 8:
        return "5_8"
    if value <= 16:
        return "9_16"
    if value <= 32:
        return "17_32"
    if value <= 64:
        return "33_64"
    if value <= 128:
        return "65_128"
    return "gt_128"
