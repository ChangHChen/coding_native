import unittest

from minileet.dsl import Assign, BinOp, BoolLit, ForEach, If, IntLit, Program, Return, Var
from minileet.env import Task, TestCase, evaluate, validate_task
from minileet.interpreter import value_type
from minileet.typecheck import TypeErrorDSL, typecheck


class P0SoundnessTests(unittest.TestCase):
    def test_bool_is_not_int(self):
        self.assertEqual(value_type(True), "bool")
        self.assertNotEqual(value_type(True), "int")

    def test_bool_return_fails_int_program(self):
        program = Program("bad", (("x", "int"),), "int", (Return(BoolLit(True)),))
        with self.assertRaises(TypeErrorDSL):
            typecheck(program)

    def test_invalid_operator_rejected(self):
        program = Program("bad", (("x", "int"),), "int", (Return(BinOp("div", Var("x"), IntLit(1))),))
        with self.assertRaises(TypeErrorDSL):
            typecheck(program)

    def test_foreach_empty_does_not_define_body_variable(self):
        program = Program(
            "bad",
            (("xs", "list[int]"),),
            "int",
            (ForEach("x", Var("xs"), (Assign("y", Var("x")),)), Return(Var("y"))),
        )
        with self.assertRaises(TypeErrorDSL):
            typecheck(program)

    def test_if_intersection_merge(self):
        program = Program(
            "ok",
            (("flag", "bool"),),
            "int",
            (If(Var("flag"), (Assign("x", IntLit(1)),), (Assign("x", IntLit(2)),)), Return(Var("x"))),
        )
        typecheck(program)

    def test_visible_hidden_overlap_rejected(self):
        task = Task(
            "bad",
            (("x", "int"),),
            "int",
            (TestCase({"x": 1}, 2),),
            (TestCase({"x": 1}, 2), TestCase({"x": 2}, 3)),
        )
        with self.assertRaises(ValueError):
            validate_task(task)

    def test_evaluate_strictly(self):
        task = Task(
            "inc",
            (("x", "int"),),
            "int",
            (TestCase({"x": 1}, 2),),
            (TestCase({"x": 2}, 3), TestCase({"x": 3}, 4)),
        )
        program = Program("inc", (("x", "int"),), "int", (Return(BinOp("+", Var("x"), IntLit(1))),))
        result = evaluate(task, program)
        self.assertTrue(result.solved)


if __name__ == "__main__":
    unittest.main()
