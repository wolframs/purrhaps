"""Compile generated Rust and compare its observable behavior with Python."""
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from purrhaps import Interpreter, CommandOracle
from purrhaps.syntax import PurrhapsError
from purrhaps.transpile import transpile


class EmitterTests(unittest.TestCase):
    def test_refuses_existing_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / 'precious.txt'
            marker.write_text('keep me')
            with self.assertRaises(PurrhapsError):
                transpile('print(hi())', directory)
            self.assertEqual(marker.read_text(), 'keep me')

    def test_invalid_source_leaves_no_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'invalid'
            with self.assertRaises(PurrhapsError):
                transpile('val n =', output)
            self.assertFalse(output.exists())

    def test_integer_range_diagnostic(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(PurrhapsError, 'signed 64-bit'):
                transpile('print(9223372036854775808)', Path(directory) / 'overflow')


@unittest.skipUnless(shutil.which('cargo'), 'Rust integration requires Cargo')
class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='purrhaps-native-tests-')
        cls.root = Path(cls.temp.name)
        cls.count = 0
        cls.env = dict(os.environ, CARGO_TARGET_DIR=str(cls.root / 'target'))
        cls.oracle = cls.root / 'oracle.py'
        cls.oracle.write_text('''import json, sys
r = json.load(sys.stdin)
x = r['input']
if x == 'bad-citation':
    print(json.dumps(dict(value=1, confidence=.9, because='See line 9999.')))
elif x == 'hedges':
    print(json.dumps(dict(value=1, confidence=.9, because='Perhaps maybe possibly however.')))
elif x == 'low':
    print(json.dumps(dict(value=1, confidence=.3, because='A guess.')))
elif x == 'invalid':
    print('not JSON')
elif x == 'leak':
    print('private error output', file=sys.stderr)
    sys.exit(3)
elif x == 'slow':
    import time
    time.sleep(2)
else:
    if r['kind'] == 'equivalence':
        x = x in [['a', 'b'], ['b', 'c']]
    print(json.dumps(dict(value=x, confidence=.9, because='Scripted evidence.', citations=[r['line']])))
''')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def compile(self, source):
        type(self).count += 1
        project = self.root / f'project-{self.count}'
        transpile(source, project)
        built = subprocess.run(['cargo', 'build', '--offline', '--quiet', '--manifest-path', str(project / 'Cargo.toml')],
                               env=self.env, capture_output=True, text=True, timeout=90)
        self.assertEqual(built.returncode, 0, built.stderr)
        return self.root / 'target/debug/purrhaps-program'

    def native(self, source, *args, external=False, no_path=False):
        binary = self.compile(source)
        if external:
            args = ('--oracle', sys.executable, '--oracle-arg', str(self.oracle), *args)
        return subprocess.run([str(binary), *args], capture_output=True, text=True,
                              timeout=10, env={'PATH': ''} if no_path else None)

    def compare(self, source, external=False):
        output = io.StringIO()
        oracle = CommandOracle([sys.executable, str(self.oracle)]) if external else None
        runtime = Interpreter(oracle, output=output)
        failure = None
        try:
            runtime.run(source)
        except PurrhapsError as exc:
            failure = type(exc).__name__
        result = self.native(source, '--stats', external=external)
        self.assertEqual(result.stdout, output.getvalue())
        self.assertIn(f'calls={runtime.calls}', result.stderr)
        if failure:
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn(failure + ':', result.stderr)
        else:
            self.assertEqual(result.returncode, 0, result.stderr)
        return result

    def test_core_semantics_differential(self):
        programs = [
            'print(hi(), 2 + 3 * 4, (2 + 3) * 4, -2, !false, -7 % 3, 7 % -3, 5 / 2)',
            'print("Unicode ☀️ \\"quoted\\" #", true && false, false || true, 2 < 3, "a" + "b")',
            'with temperature(0) { maybe n: Int = "the answer to everything"; val x = n + 1; print(x, x); val y = n!!; print(y, y.because, y.confidence) }',
            'perhaps n = 7\nprint(n, n)\nfun visit() { print(n, n) }\nvisit()\nprint(n.owner)\nvisit()\nprint(n, n.owner)',
            'fun greet(user: Human) { maybe mood = user.lastMessage; if (mood ~= "sunny") print("☀️") }\ngreet(Human("cheerful"))',
            'val offset = 2\nfun f(n: Int): Int { return n + offset }\nmaybe x: Int = 3\nval y = f(x)\nprint(y, y)',
            'fun f(): Int { with temperature(0) { if (true) { return 7 } }; return 9 }\nprint(f())',
            'maybe n = 7\nprint(n.confidence, n.because)\nprint(n.owner)',
            'maybe n = 7\nprint(n == 7)',
            'maybe n: Int = true\nprint(n)',
            'maybe me = this\nprint(me!!) ',
            'if (1) print("no")',
            'fun f(x: Int) { return x }\nf("cat")',
            'val x = 1\nval x = 2',
            'return 1',
            'print(1 / 0)',
            'print(-9223372036854775808, 9223372036854775807)',
        ]
        for source in programs:
            with self.subTest(source=source):
                self.compare(source)

    def test_subprocess_oracle_differential(self):
        for source in [
            'maybe n = 7\nprint(n + 1, n!!) ',
            'print("a" ~= "b", "b" ~= "c", "a" ~= "c")',
            'maybe x = "low"\nval y = x + 1\nprint(y!!)',
            'maybe x = "bad-citation"\nprint(x)',
            'maybe x = "hedges"\nprint(x)',
            'maybe x = "invalid"\nprint(x)',
            'maybe x = "leak"\nprint(x)',
        ]:
            with self.subTest(source=source):
                result = self.compare(source, external=True)
                self.assertNotIn('private error output', result.stderr)

    def test_runs_without_python_or_path(self):
        result = self.native('print(hi())', no_path=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, 'hi! 🐈‍⬛\n')

    def test_cli_emission(self):
        project = self.root / 'cli-output'
        result = subprocess.run([sys.executable, '-m', 'purrhaps', '-', '--emit-rust', str(project)],
                                input='print(hi())', capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('fn main()', (project / 'src/main.rs').read_text())
        self.assertTrue((project / 'src/runtime.rs').exists())

    def test_call_budget(self):
        result = self.native('maybe n = 1\nprint(n, n)', '--max-calls', '1', '--stats')
        self.assertEqual(result.returncode, 1)
        self.assertIn('budget exhausted', result.stderr)
        self.assertIn('calls=1', result.stderr)

    def test_timeout(self):
        result = self.native('maybe x = "slow"\nprint(x)', '--timeout', '.05', external=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn('timed out', result.stderr)

    def test_overflow_is_diagnosed(self):
        result = self.native('print(9223372036854775807 + 1)')
        self.assertEqual(result.returncode, 1)
        self.assertIn('integer overflow', result.stderr)
        self.assertNotIn('panicked', result.stderr)

    def test_recursive_recipe_has_diagnostic(self):
        result = self.native('maybe x = x\nprint(x)')
        self.assertEqual(result.returncode, 1)
        self.assertIn('recursion limit', result.stderr)

    def test_examples_compile_and_run(self):
        for path in Path('examples').glob('*.prr'):
            with self.subTest(path=path):
                result = self.native(path.read_text())
                expected = 1 if path.stem == 'uncertainty' else 0
                self.assertEqual(result.returncode, expected, result.stderr)


if __name__ == '__main__':
    unittest.main()
