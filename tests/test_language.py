import io
import subprocess
import sys
import unittest

from purrhaps import Interpreter, Answer, CommandOracle
from purrhaps.oracle import ConfabulationError, HedgeOverflowError, OracleError
from purrhaps.runtime import UncertaintyException, TypeError, NameError
from purrhaps.syntax import Parser, SyntaxError, PurrhapsError


class ScriptedOracle:
    def __init__(self, *answers):
        self.answers, self.requests = iter(answers), []

    def ask(self, request):
        self.requests.append(request)
        value = next(self.answers)
        return value if isinstance(value, Answer) else Answer(value, .9, "A test oracle answered.")


class LanguageTests(unittest.TestCase):
    def run_source(self, source, *answers):
        oracle, output = ScriptedOracle(*answers), io.StringIO()
        runtime = Interpreter(oracle, output=output)
        runtime.run(source)
        return output.getvalue(), oracle, runtime

    def test_each_read_and_contagion(self):
        out, oracle, _ = self.run_source('maybe n: Int = "number"\nval m = n + 1\nprint(m, m)', 3, 8)
        self.assertEqual(out, "4 9\n")
        self.assertEqual(len(oracle.requests), 2)

    def test_perhaps_per_reading_scope(self):
        out, oracle, _ = self.run_source('''perhaps n = 0
print(n, n)
fun f() { print(n, n) }
f()
f()
print(n)
''', 1, 2, 3)
        self.assertEqual(out, "1 1\n2 2\n3 3\n1\n")
        self.assertEqual(len(oracle.requests), 3)

    def test_collapse_is_a_snapshot_and_keeps_provenance(self):
        out, oracle, _ = self.run_source('maybe n = 0\nval x = n!!\nprint(x, x, x.confidence, x.because)',
                                          Answer(42, .8, "A defensible guess.", [1]))
        self.assertEqual(out, "42 42 0.8 A defensible guess.\n")
        self.assertEqual(len(oracle.requests), 1)

    def test_low_confidence_collapse(self):
        with self.assertRaises(UncertaintyException):
            self.run_source('maybe n = 0\nprint(n!!)', Answer(42, .79, "A guess."))

    def test_low_confidence_propagates(self):
        with self.assertRaises(UncertaintyException):
            self.run_source('maybe n = 0\nval m = n + 1\nprint(m!!)', Answer(42, .3, "A guess."))

    def test_nested_maybe_cannot_launder_confidence(self):
        with self.assertRaises(UncertaintyException):
            self.run_source('maybe n = 0\nmaybe m = n\nprint(m!!)', Answer(42, .2, "A guess."), 42)

    def test_equality_rejects_maybes_without_calls(self):
        for op in ('==', '!='):
            with self.assertRaises(TypeError):
                self.run_source(f'maybe n = 0\nprint(n {op} 0)')

    def test_equivalence_is_nontransitive(self):
        out, oracle, _ = self.run_source('print("a" ~= "b", "b" ~= "c", "a" ~= "c")', True, True, False)
        self.assertEqual(out, "true true false\n")
        self.assertTrue(all(r['kind'] == 'equivalence' for r in oracle.requests))

    def test_branch_collapses_and_checks_boolean(self):
        out, _, _ = self.run_source('if ("cat" ~= "kitty") { print("yes") } else { print("no") }', True)
        self.assertEqual(out, "yes\n")
        with self.assertRaises(UncertaintyException):
            self.run_source('if ("cat" ~= "kitty") print("yes")', Answer(True, .5, "A guess."))
        with self.assertRaises(TypeError):
            self.run_source('if (1) print("yes")')

    def test_temperature_restored_on_failure(self):
        runtime = Interpreter(ScriptedOracle(Answer(1, .1, "A guess.")))
        with self.assertRaises(UncertaintyException):
            runtime.run('maybe n = 0\nwith temperature(0.0) { print(n!!) }')
        self.assertEqual(runtime.temperature, .7)
        self.assertEqual(runtime.oracle.requests[0]['temperature'], 0)

    def test_temperature_nested(self):
        _, oracle, _ = self.run_source('''maybe n = 0
with temperature(0.0) {
 print(n)
 with temperature(1.0) { print(n) }
 print(n)
}
print(n)''', 1, 2, 3, 4)
        self.assertEqual([r['temperature'] for r in oracle.requests], [0, 1, 0, .7])

    def test_ownership_moves_on_cached_reads(self):
        out, _, _ = self.run_source('''perhaps n = 0
print(n.owner)
print(n)
fun f() { print(n) }
f()
print(n.owner)
print(n)
print(n.owner)''', 1, 2)
        self.assertEqual(out, "null\n1\n2\nf#1\n1\nmodule\n")

    def test_metadata_reads_are_independent(self):
        out, _, _ = self.run_source('maybe n = 0\nprint(n.confidence, n.because)',
                                    Answer(1, .3, "First."), Answer(2, .9, "Second."))
        self.assertEqual(out, "0.3 Second.\n")

    def test_function_uncertainty_and_lexical_scope(self):
        out, _, _ = self.run_source('''val offset = 2
fun add(n: Int): Int { return n + offset }
maybe x: Int = 0
val y = add(x)
print(y, y)''', 3, 4)
        self.assertEqual(out, "5 6\n")

    def test_function_checks_and_return(self):
        with self.assertRaises(TypeError):
            self.run_source('fun f(x: Int) { return x }\nf("cat")')
        with self.assertRaises(TypeError):
            self.run_source('fun f(x: Int) { return x }\nf()')
        with self.assertRaises(PurrhapsError):
            self.run_source('return 1')
        with self.assertRaises(NameError):
            self.run_source('fun f(x, x) { return x }')

    def test_type_bool_is_not_int(self):
        with self.assertRaises(TypeError):
            self.run_source('maybe n: Int = 0\nprint(n)', True)
        with self.assertRaises(TypeError):
            self.run_source('val n: Int = true')

    def test_precedence_comments_strings(self):
        out, _, _ = self.run_source('print(2 + 3 * 4, (2 + 3) * 4, -2, !false); print("cat // ☀️") // comment')
        self.assertEqual(out, "14 20 -2 true\ncat // ☀️\n")

    def test_invalid_programs_have_language_errors(self):
        for source in ['val x =', 'if (true) {', 'print("bad\\q")', '@', 'val x = 1 val y = 2', 'print(', 'maybe = 1']:
            with self.subTest(source=source), self.assertRaises(SyntaxError):
                Parser(source).program()

    def test_confabulation(self):
        for answer in [Answer(1, .9, 'See line 999.'), Answer(1, .9, 'Evidence.', [0]), Answer(1, .9, 'Evidence.', [3])]:
            with self.assertRaises(ConfabulationError):
                self.run_source('maybe x = 0\nprint(x)', answer)

    def test_hedge_overflow(self):
        for answer in [Answer(1, .9, 'Maybe, perhaps, possibly, however.'),
                       Answer(1, .9, 'Evidence.', caveats=['a', 'b', 'c', 'd'])]:
            with self.assertRaises(HedgeOverflowError):
                self.run_source('maybe x = 0\nprint(x)', answer)

    def test_invalid_answers(self):
        for answer in [Answer(1, float('nan'), 'x'), Answer(1, 2, 'x'), Answer(1, True, 'x'),
                       Answer(1, .9, ''), Answer(1, .9, 'x', ['1']), Answer(float('inf'), .9, 'x')]:
            with self.assertRaises(OracleError):
                self.run_source('maybe x = 0\nprint(x)', answer)

    def test_budget(self):
        runtime = Interpreter(ScriptedOracle(1), output=io.StringIO(), max_calls=1)
        with self.assertRaises(OracleError):
            runtime.run('maybe x = 0\nprint(x, x)')
        self.assertEqual(runtime.calls, 1)

    def test_hi_never_calls_oracle(self):
        out, oracle, _ = self.run_source('print(hi(), hi().confidence)')
        self.assertEqual(out, 'hi! 🐈‍⬛ 1.0\n')
        self.assertEqual(oracle.requests, [])

    def test_demo_zero_temperature(self):
        out = io.StringIO()
        Interpreter(output=out).run('maybe n: Int = "the answer to everything"\nwith temperature(0) { print(n!!, n!!) }')
        self.assertEqual(out.getvalue(), '42 42\n')

    def test_cli_examples(self):
        for name in ('hello', 'ecology'):
            result = subprocess.run([sys.executable, '-m', 'purrhaps', f'examples/{name}.prr', '--stats'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('calls=', result.stderr)
        result = subprocess.run([sys.executable, '-m', 'purrhaps', 'examples/uncertainty.prr'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn('UncertaintyException: line 4:', result.stderr)
        self.assertNotIn('Traceback', result.stderr)


class CommandTests(unittest.TestCase):
    def test_protocol_roundtrip(self):
        code = 'import sys,json; r=json.load(sys.stdin); print(json.dumps({"value":r["input"],"confidence":0.9,"because":"Echo."}))'
        oracle = CommandOracle([sys.executable, '-c', code])
        output = io.StringIO()
        Interpreter(oracle, output=output).run('maybe n = 7\nprint(n!!)')
        self.assertEqual(output.getvalue(), '7\n')

    def test_bad_output_and_redacted_errors(self):
        for code in ['print("not json")', 'import sys; print("private child output", file=sys.stderr); sys.exit(3)']:
            with self.assertRaises(OracleError) as caught:
                CommandOracle([sys.executable, '-c', code]).ask({})
            self.assertNotIn('private child output', str(caught.exception))

    def test_timeout(self):
        with self.assertRaises(OracleError):
            CommandOracle([sys.executable, '-c', 'import time; time.sleep(1)'], timeout=.01).ask({})


if __name__ == '__main__':
    unittest.main()
