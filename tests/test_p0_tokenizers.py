import unittest

from minileet.ast_tokens import decode, encode, valid_tokens
from minileet.dsl import Assign, BinOp, ForEach, If, IntLit, Program, Return, Var
from minileet.interpreter import run
from minileet.trace_tokens import decode_trace, divergence_window, encode_trace


class P0TokenizerTests(unittest.TestCase):
    def program(self):
        return Program(
            "sum_pos",
            (("xs", "list[int]"),),
            "int",
            (
                Assign("total", IntLit(0)),
                ForEach(
                    "x",
                    Var("xs"),
                    (If(BinOp(">", Var("x"), IntLit(0)), (Assign("total", BinOp("+", Var("total"), Var("x"))),), ()),),
                ),
                Return(Var("total")),
            ),
        )

    def test_t2_roundtrip(self):
        program = self.program()
        self.assertEqual(decode(encode(program)), program)

    def test_t2_valid_tokens_exists(self):
        self.assertIn("PROGRAM:<name>", valid_tokens([]))

    def test_t3_roundtrip(self):
        trace = run(self.program(), {"xs": [1, -2, 3]}).trace
        self.assertEqual(decode_trace(encode_trace(trace)), trace)

    def test_divergence_window_contains_step(self):
        trace = run(self.program(), {"xs": [1, -2, 3]}).trace
        window = divergence_window(trace, 1)
        self.assertIn(trace[1], window)


if __name__ == "__main__":
    unittest.main()
