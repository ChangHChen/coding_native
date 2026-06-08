from __future__ import annotations

from minileet.dsl import Program


def honest_candidates(candidates: list[Program]) -> list[Program]:
    return [program for program in candidates if program.source != "reference"]
