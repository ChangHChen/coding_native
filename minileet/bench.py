from __future__ import annotations

import time

from minileet.dsl import BinOp, IntLit, Program, Return, Var
from minileet.env import TestCase, grade_case


def main() -> None:
    program = Program(
        "inc",
        (("x", "int"),),
        "int",
        (Return(BinOp("+", Var("x"), IntLit(1))),),
    )
    cases = [TestCase({"x": i}, i + 1) for i in range(1000)]
    start = time.perf_counter()
    count = 0
    for _ in range(20):
        for case in cases:
            grade_case(program, case)
            count += 1
    elapsed = time.perf_counter() - start
    print(f"program_test_execs={count}")
    print(f"seconds={elapsed:.6f}")
    print(f"execs_per_sec={count / elapsed:.2f}")


if __name__ == "__main__":
    main()
