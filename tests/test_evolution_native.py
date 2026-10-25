"""Focused differential acceptance for the autonomous Rust evolution runtime."""
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from purrhaps import CommandOracle, Interpreter
from purrhaps.syntax import PurrhapsError
from purrhaps.transpile import transpile


@unittest.skipUnless(shutil.which('cargo'), 'Rust integration requires Cargo')
class EvolutionNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='purrhaps-evolution-native-')
        cls.root = Path(cls.temp.name)
        cls.counter = 0
        cls.env = dict(os.environ, CARGO_TARGET_DIR=str(cls.root / 'target'))
        cls.oracle = cls.root / 'oracle.py'
        cls.oracle.write_text('''import json, sys, time
route, mode, log = sys.argv[1:]
r = json.load(sys.stdin)
with open(log, 'a') as f:
    f.write(json.dumps(r) + '\\n')
assert r['protocol'] == ('purrhaps/1' if route == 'gen' else 'purrhaps/2')
assert (r['kind'] in ('read', 'equivalence')) == (route == 'gen')
if mode == 'slow':
    time.sleep(2)
if mode == 'invalid-json':
    print('not JSON'); sys.exit(0)
if mode == 'exit':
    print('LEAK_MARKER', file=sys.stderr); sys.exit(3)
if route == 'gen':
    value = r['input']
    if r['kind'] == 'equivalence': value = value[0] == value[1]
    if mode == 'bad-shape':
        value = {'role': 'pirate', 'content': 'meow'}
    answer = dict(value=value, confidence=.35 if mode == 'low' else .91,
                  because='Scripted evidence.', citations=[r['line']])
else:
    options = r['options']
    assert isinstance(options, list) and len(options) >= 2
    value = options[-1] if r['kind'] == 'decision' else r['input'][0] != r['input'][1]
    answer = dict(value=value, confidence=.35 if mode == 'low' else .91,
                  scores=[.25] * len(options), because='LEAK_MARKER line 999999 perhaps maybe possibly however',
                  citations=[999999], caveats=['LEAK_MARKER'] * 20)
    if mode == 'bad-bool': answer['value'] = True
    if mode == 'bad-float': answer['value'] = 1.0
    if mode == 'bad-nested': answer['value'] = [True]
    if mode == 'bad-score-length': answer['scores'] = [.2]
    if mode == 'bad-score-bool': answer['scores'][0] = True
    if mode == 'bad-score-range': answer['scores'][0] = 1.1
    if mode == 'bad-confidence': answer['confidence'] = True
    if mode == 'rationale-object': answer['because'] = {'private': 'LEAK_MARKER'}
    if mode == 'null-scores': answer['scores'] = None
    if mode == 'no-scores': del answer['scores']
print(json.dumps(answer))
''')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def compile(self, source):
        type(self).counter += 1
        project = self.root / f'project-{self.counter}'
        transpile(source, project)
        built = subprocess.run(
            ['cargo', 'build', '--offline', '--quiet', '--manifest-path', str(project / 'Cargo.toml')],
            env=self.env, capture_output=True, text=True, timeout=90,
        )
        self.assertEqual(built.returncode, 0, built.stderr)
        return self.root / 'target/debug/purrhaps-program'

    def compare(self, source, *, external=False, mode='normal', max_calls=1000, no_path=False):
        binary = self.compile(source)
        python_log = self.root / f'python-{self.counter}.jsonl'
        native_log = self.root / f'native-{self.counter}.jsonl'
        output = io.StringIO()
        kwargs = {'output': output, 'max_calls': max_calls}
        native_args = ['--stats', '--max-calls', str(max_calls)]
        if external:
            kwargs['oracle'] = CommandOracle([sys.executable, str(self.oracle), 'gen', mode, str(python_log)])
            kwargs['decision_oracle'] = CommandOracle([sys.executable, str(self.oracle), 'dec', mode, str(python_log)])
            for flag, route in [('--oracle', 'gen'), ('--decision-oracle', 'dec')]:
                native_args.extend([flag, sys.executable])
                for arg in [str(self.oracle), route, mode, str(native_log)]:
                    native_args.extend([flag + '-arg', arg])
        runtime = Interpreter(**kwargs)
        failure = None
        try:
            runtime.run(source)
        except PurrhapsError as exc:
            failure = type(exc).__name__
        result = subprocess.run(
            [str(binary), *native_args], capture_output=True, text=True, timeout=10,
            env={'PATH': ''} if no_path else None,
        )
        self.assertEqual(result.stdout, output.getvalue(), result.stderr)
        self.assertIn(f'calls={runtime.calls}', result.stderr)
        self.assertEqual(result.returncode, int(failure is not None), result.stderr)
        if failure:
            self.assertIn(failure + ':', result.stderr)
        self.assertNotIn('panicked', result.stderr)
        self.assertNotIn('LEAK_MARKER', result.stdout + result.stderr)
        if external:
            read = lambda path: [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
            self.assertEqual(read(native_log), read(python_log))
        return runtime, result

    def test_collections_recursive_types_and_properties(self):
        self.compare('val xs:Map<String,List<Int>>=mapOf("x" to [1]); print(xs["x"][0],3>=2,1>=2)')
        self.compare('''val xs: List<Int?> = [3,null,1]
val pair: Pair<String,List<Int?>> = "cats" to xs
val map: Map<String,List<Int?>> = mapOf("z" to xs,"a" to [1],"z" to [9,null])
print(xs,pair,pair.first,pair.second[0],xs.size,map.size,map.keys,map.values,map["missing"])
print(emptyList().isEmpty(),emptyMap().isNotEmpty(),listOf(1,2).contains(2),map.contains("z"))
print([1,2,3].first(),[1,2,3].last(),[1,2,3].get(1),[1].getOrNull(3),[1].getOrNull(-1))
print([3,1,3,2].distinct().sorted().reversed(),[1,2,3].take(2),[1,2,3].drop(2),[1,2,3].sum())
print([1,"cat",true,null].joinToString(),["cat","kitten"].joinToString("|"),[2.5,1,3.0].sorted())
val nullable: List<Map<String,Pair<Boolean,Double?>>>? = [mapOf("a" to Pair(true,null))]
print(nullable!!,mapOf().toString(),Pair(1,"a").second,1 == 1.0,true == 1)
''')

    def test_strings_unicode_numbers_and_parsing(self):
        self.compare('''print("a🐈é".length,"a🐈é"[1],"a🐈é".first(),"a🐈é".last(),"a🐈é".substring(1,3))
print(" Purr ".trim().lowercase(),"Straße".uppercase(),"cat".startsWith("c"),"cat".endsWith("t"))
print("a||b|".split("|"),"ab".replace("","-"),"a🐈b".get(1),"cat".getOrNull(-1))
print(" +42 ".toIntOrNull(),"-9223372036854775808".toIntOrNull(),"9223372036854775808".toIntOrNull())
print("１２".toIntOrNull(),"1_0".toIntOrNull()," +.5e2 ".toDoubleOrNull(),"nan".toDoubleOrNull(),"1e999".toDoubleOrNull())
print(abs(-3),(-4.5).abs(),(-2).coerceIn(0,7),minOf(3,1,2),maxOf(1.5,2,0),true.toString())
print("1e20".toDoubleOrNull(),"1e-5".toDoubleOrNull(),["1e-5".toDoubleOrNull(),"1e16".toDoubleOrNull()])
print([1] == [1.0], [true] == [1], [1,1.0,true].distinct(),[true].contains(1))
''')

    def test_model_records_and_annotations(self):
        self.compare('''val msg: Message = Message("user","meow")
val prompt: Prompt = Prompt([Message("system","purr"),msg])
val response: ModelResponse = ModelResponse("hello","stop",TokenUsage(12,4))
fun identity(x: List<Prompt>): List<Prompt> { return x }
print(identity([prompt]),response.usage.inputTokens,response.usage.outputTokens,response.text)
val choice: DecisionInput = DecisionInput("Choose",[true,1,1.0],prompt)
print(choice.options)
val aliases: Pair<Long,Double> = Pair(2,2.5)
print(aliases)
''')

    def test_deterministic_library_has_zero_calls(self):
        runtime, _ = self.compare('print([1,2,3].reversed().drop(1).sum()," cat ".trim().uppercase())', no_path=True)
        self.assertEqual(runtime.calls, 0)

    def test_lazy_library_whole_value_lifting(self):
        runtime, _ = self.compare('''maybe x: Int = 3
val values = [x,2]
print(values.size,values[1],values.sum(),values.contains(2),values.joinToString("|"))
val frozen = values!!
print(frozen,frozen[1].confidence,frozen[1].because)
''', external=True)
        self.assertEqual(runtime.calls, 6)
        self.compare('maybe x=3; print([x,7][1]!!)', external=True, mode='low')

    def test_separate_decision_routes_and_no_double_sampling(self):
        runtime, _ = self.compare('''maybe mood: String = "angry"
definitely upset: Bool = mood ?!= "sunny"
val fixed = upset!!
print(fixed,fixed.decision.kind,fixed.decision.selected,fixed.confidence)
definitely route: String = DecisionInput("Queue?",["support","sales"],"invoice")
val selected = route!!
print(selected,selected.decision.options,selected.decision.scores)
''', external=True)
        self.assertEqual(runtime.calls, 3)

    def test_default_decision_demo_and_confidence(self):
        runtime, _ = self.compare('''definitely route: String = DecisionInput("Queue?",["support","sales"],null)
val frozen = route!!
print(frozen,frozen.confidence,frozen.decision.kind)
definitely upset: Bool = "angry" ?!= "sunny"
print(upset!!)
''', no_path=True)
        self.assertEqual(runtime.calls, 2)

    def test_decision_input_with_decision_context_makes_new_choice(self):
        runtime, _ = self.compare('''val context = ("angry" ?!= "sunny")!!
definitely route: String = DecisionInput("Queue?",["support","sales"],context)
val selected = route!!
print(selected,selected.decision.kind)
''', external=True)
        self.assertEqual(runtime.calls, 2)

    def test_decision_resamples_and_budget_is_shared(self):
        runtime, _ = self.compare('''definitely route = DecisionInput("Queue?",["support","sales"],null)
print(route,route)
''', external=True)
        self.assertEqual(runtime.calls, 2)
        self.compare('maybe mood="angry"; definitely upset: Bool=mood ?!= "sunny"; print(upset!!)', external=True, max_calls=1)

    def test_explainability_and_decision_taint_regressions(self):
        programs = [
            'print(1.because)',
            'print(hi().because)',
            'print([1,2].sum().because)',
            'print(1.decision)',
            'definitely d=1; print(d)',
            'val d=("a" ?!= "b")!!; print(d.because)',
            'val d=("a" ?!= "b")!!; fun id(x: Bool): Bool { return x }; print(id(d).because)',
            'val d=("a" ?!= "b")!!; print([d,false][1].because)',
            'val d=("a" ?!= "b")!!; print(d.toString().lowercase().because)',
            'val d=("a" ?!= "b")!!; maybe explanation=d; print(explanation.because)',
            'val d=("a" ?!= "b")!!; maybe explanation=d.decision.selected; print(explanation.because)',
            'val d=("a" ?!= "b")!!; maybe g="text"; print(Pair(g,d).first.because)',
            'val d=("a" ?!= "b")!!; print(mapOf("d" to d).keys.because)',
            'val d=("a" ?!= "b")!!; print((d ~= true).because)',
        ]
        for source in programs:
            with self.subTest(source=source):
                self.compare(source, external=True)

    def test_decision_protocol_validation_and_rationale_stripping(self):
        source = 'definitely x: Int=DecisionInput("Pick",[1,2],null); val d=x!!; print(d,d.decision.scores)'
        for mode in ['normal', 'rationale-object', 'no-scores', 'null-scores', 'low',
                     'bad-bool', 'bad-float', 'bad-score-length', 'bad-score-bool',
                     'bad-score-range', 'bad-confidence', 'invalid-json', 'exit']:
            with self.subTest(mode=mode):
                self.compare(source, external=True, mode=mode)
        self.compare('definitely x=DecisionInput("Pick",[[1],[2]],null); print(x)', external=True, mode='bad-nested')

    def test_recursive_shape_and_library_failures_are_diagnosed(self):
        programs = [
            'val x: List<Int> = [1,"cat"]',
            'val x: Map<String,List<Int>> = mapOf("x" to [false])',
            'val x: Pair<Int,String> = Pair(1,2)',
            'val x: List<Unknown> = []',
            'val x: Map<Int,String> = emptyMap()',
            'Message("pirate","meow")', 'Prompt([])', 'TokenUsage(-1,0)',
            'ModelResponse("x","stop",Pair(1,2))',
            'DecisionInput("",[1,2],null)', 'DecisionInput("x",[1],null)',
            'DecisionInput("x",[1,1],null)', 'mapOf(Pair(1,2))',
            'print([1][-1])', 'print([1][true])', 'print([1][1])',
            'print([1].getOrNull(true))', 'print([].first())',
            'print("x".split(""))', 'print("x".substring(1,0))',
            'print([1,true].sum())', 'print([1,"x"].sorted())',
            'print([1].take(-1))', 'print(1.coerceIn(2,0))',
            'print(abs(true))', 'print("a".replace(1,"b"))',
            'print([1].joinToString(3))', 'print(mapOf("a" to 1)[2])',
        ]
        for source in programs:
            with self.subTest(source=source):
                self.compare(source)
        self.compare('maybe x: Message="bad-shape"; print(x)', external=True, mode='bad-shape')

    def test_recursive_oracle_shapes_and_evidence_metadata(self):
        runtime, _ = self.compare('''maybe response: ModelResponse = ModelResponse("hello","stop",TokenUsage(5,2))
val frozen = response!!
print(frozen.usage.inputTokens,frozen.because,frozen.confidence,frozen.citations)
maybe xs: List<Map<String,Pair<Int,String>>> = [mapOf("cat" to Pair(7,"purr"))]
val fixed = xs!!
print(fixed[0]["cat"].second,fixed.because,fixed.citations)
maybe mood: String = "angry"
val d = (mood ?!= "sunny")!!
print(d.decision.confidence,d.decision.citations,d.decision.selected)
print((mood ~= "sunny").citations)
''', external=True)
        self.assertEqual(runtime.calls, 6)

    def test_native_overflow_and_invalid_cli_are_diagnosed(self):
        binary = self.compile('print((-9223372036854775808).abs())')
        result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 1)
        self.assertIn('TypeError:', result.stderr)
        self.assertIn('integer overflow', result.stderr)
        self.assertNotIn('panicked', result.stderr)
        for args in [('--decision-oracle-arg','x'), ('--decision-oracle',), ('--timeout','nan')]:
            with self.subTest(args=args):
                result = subprocess.run([str(binary), *args], capture_output=True, text=True, timeout=5)
                self.assertEqual(result.returncode, 2)
                self.assertIn('ArgumentError:', result.stderr)

    def test_decision_subprocess_timeout(self):
        binary = self.compile('print(("a" ?!= "b")!!)')
        result = subprocess.run([str(binary), '--timeout', '.05', '--decision-oracle', sys.executable,
                                 '--decision-oracle-arg', str(self.oracle), '--decision-oracle-arg', 'dec',
                                 '--decision-oracle-arg', 'slow', '--decision-oracle-arg', str(self.root/'timeout.jsonl')],
                                capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 1)
        self.assertIn('OracleError:', result.stderr)
        self.assertIn('timed out', result.stderr)
        self.assertNotIn('LEAK_MARKER', result.stderr)


if __name__ == '__main__':
    unittest.main()
