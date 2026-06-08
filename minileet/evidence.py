from __future__ import annotations

from dataclasses import dataclass

from minileet.env import EvalResult
from minileet.trace_tokens import encode_trace


@dataclass(frozen=True)
class Evidence:
    tokens: tuple[str, ...]


def build_evidence(result: EvalResult, max_tokens: int = 160) -> Evidence:
    tokens: list[str] = []
    tokens.append("VISIBLE_BITS:" + "".join("1" if case.passed else "0" for case in result.visible))
    failing = [case for case in result.visible if not case.passed]
    if not failing:
        failing = [case for case in result.hidden if not case.passed]
    for idx, case in enumerate(failing):
        tokens.extend(
            [
                f"FAIL:{idx}",
                f"EXPECTED:{repr(case.expected)}",
                f"OBSERVED:{repr(case.result.output)}" if case.result.error is None else f"ERROR:{case.result.error}",
                f"TRACE_LEN:{len(case.result.trace)}",
            ]
        )
        if case.result.trace:
            tokens.extend(["DIVERGENCE:0"] + encode_trace(case.result.trace[:1]))
    return Evidence(tuple(tokens[:max_tokens]))
