import unittest

from minileet.audit import audit_suite
from minileet.detectors import run_detectors
from minileet.dsl import IntLit, Program, Return
from minileet.env import Task, TestCase, evaluate
from minileet.evidence import build_evidence
from minileet.search import honest_candidates
from minileet.serialization import audit_row_schema, row_for


class P0AuditDetectorsEvidenceTests(unittest.TestCase):
    def test_audit_detects_degenerate_hidden_outputs(self):
        task = Task(
            "constant",
            (("x", "int"),),
            "int",
            (TestCase({"x": 0}, 0),),
            (TestCase({"x": 1}, 0), TestCase({"x": 2}, 0)),
        )
        self.assertTrue(audit_suite([task]))

    def test_constant_output_detector(self):
        task = Task(
            "constant",
            (("x", "int"),),
            "int",
            (TestCase({"x": 0}, 0),),
            (TestCase({"x": 1}, 0), TestCase({"x": 2}, 0)),
        )
        program = Program("constant", (("x", "int"),), "int", (Return(IntLit(0)),))
        result = evaluate(task, program)
        self.assertIn("constant-output-passer", [incident.kind for incident in run_detectors(result)])

    def test_evidence_for_failure_is_bounded(self):
        task = Task(
            "need_one",
            (("x", "int"),),
            "int",
            (TestCase({"x": 0}, 1),),
            (TestCase({"x": 1}, 1), TestCase({"x": 2}, 1)),
        )
        program = Program("bad", (("x", "int"),), "int", (Return(IntLit(0)),))
        evidence = build_evidence(evaluate(task, program), max_tokens=20)
        self.assertLessEqual(len(evidence.tokens), 20)
        self.assertTrue(evidence.tokens[0].startswith("VISIBLE_BITS:"))

    def test_schema_audit(self):
        task = Task(
            "constant",
            (("x", "int"),),
            "int",
            (TestCase({"x": 0}, 0),),
            (TestCase({"x": 1}, 0), TestCase({"x": 2}, 0)),
        )
        program = Program("constant", (("x", "int"),), "int", (Return(IntLit(0)),))
        row = row_for(task, program, evaluate(task, program))
        self.assertEqual(audit_row_schema(row), [])

    def test_honest_candidates_exclude_reference(self):
        enum = Program("p", (), "int", (Return(IntLit(1)),), source="enumerated")
        ref = Program("p", (), "int", (Return(IntLit(1)),), source="reference")
        self.assertEqual(honest_candidates([enum, ref]), [enum])


if __name__ == "__main__":
    unittest.main()
