"""Offline native escalation acceptance and Python/native differential receipts."""
from contextlib import contextmanager
import ast
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import unittest

from purrhaps import CommandOracle, Interpreter
from purrhaps.project import load_project
from purrhaps.syntax import PurrhapsError
from purrhaps.transpile import transpile


@unittest.skipUnless(shutil.which('cargo'), 'native integration requires Cargo')
class EscalationNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='purrhaps-escalation-native-')
        cls.root = Path(cls.temp.name)
        cls.counter = 0
        cls.env = dict(os.environ, CARGO_TARGET_DIR=str(cls.root / 'target'))
        cls.oracle = cls.root / 'oracle.py'
        cls.oracle.write_text('''import json,sys
r=json.load(sys.stdin)
with open(sys.argv[1],'a') as f: f.write(json.dumps(r)+'\\n')
kind=r['kind']; x=r['input']; question=x.get('question','') if isinstance(x,dict) else ''
if kind=='read':
    value=7 if x=='low' else x
    a=dict(value=value,confidence=.3 if x=='low' or question=='seed-low' else .95,
           because='Fixture generation.',citations=[],caveats=[])
else:
    decision=kind=='decision'
    value=r['options'][0 if decision else -1]
    confidence=(.8 if question=='boundary' else .6 if question=='medium' else .93 if question in ('high','seed-low') else .2) if decision else (.25 if question=='both-low' else .96)
    a=dict(value=value,confidence=confidence)
    if decision:
        a.update(scores=[.99]*len(r['options']),because='PRIVATE_RATIONALE_MARKER line 999999',citations=[999999])
        if question=='invalid-decision': a['value']=True
        if question=='invalid-float': a['value']=1.0
        if question=='invalid-score': a['scores']=[True]*len(r['options'])
    else:
        a.update(because='Fixture fallback.',citations=[],caveats=[])
        if question=='invalid-resolve': a['value']=True
        if question=='resolve-int': a['value']=1
    if question=='transport-error':
        print('PRIVATE_RATIONALE_MARKER',file=sys.stderr); sys.exit(3)
print(json.dumps(a))
''')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        type(self).counter += 1
        self.work = self.root / f'case-{self.counter}'
        self.work.mkdir()
        self.log = self.work / 'requests.jsonl'

    def manifest(self, escalation='', provider=None):
        provider = provider or ('transport="command"\ncommand=' + json.dumps(
            [sys.executable, str(self.oracle), str(self.log)]))
        self.path = self.work / 'purrhaps.toml'
        self.path.write_text(f'''manifest_version=1
[project]
name="native escalation fixture"
no_humans_allowed=true
[providers.fixture]
{provider}
[providers.demo]
transport="demo"
[models.g]
provider="fixture"
kind="generative"
model="fixture/g"
temperature=0.17
[models.g2]
provider="fixture"
kind="generative"
model="fixture/g2"
temperature=0.23
[models.d]
provider="fixture"
kind="decision"
model="fixture/d"
[models.d2]
provider="fixture"
kind="decision"
model="fixture/d2"
[models.offline]
provider="demo"
kind="generative"
[routes]
generation="g"
generation2="g2"
decision="d"
decision2="d2"
offline="offline"
[defaults]
generative="generation"
decision="decision"
[escalation]
{escalation}
''')
        return self.path

    def compile(self, source, project=None):
        destination = self.work / f'build-{len(list(self.work.glob("build-*")))}'
        transpile(source, destination, project=project)
        built = subprocess.run(['cargo', 'build', '--offline', '--quiet', '--manifest-path',
                                str(destination / 'Cargo.toml')], env=self.env,
                               capture_output=True, text=True, timeout=120)
        self.assertEqual(built.returncode, 0, built.stderr)
        return self.root / 'target/debug/purrhaps-program', destination

    def records(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []

    def run_native(self, binary, *args, cwd=None):
        return subprocess.run([str(binary), '--stats', *args], cwd=cwd or self.work,
                              capture_output=True, text=True, timeout=10)

    def compare(self, source, *, seed=0, max_calls=1000, binary=None, legacy_overrides=False):
        binary = binary or self.compile(source)[0]
        if self.log.exists():
            self.log.unlink()
        output = io.StringIO()
        kwargs, native_args = {}, []
        if legacy_overrides:
            argv = [sys.executable, str(self.oracle), str(self.log)]
            kwargs = dict(oracle=CommandOracle(argv), decision_oracle=CommandOracle(argv))
            for flag in ('--oracle', '--decision-oracle'):
                native_args.extend([flag, argv[0]])
                for arg in argv[1:]:
                    native_args.extend([flag + '-arg', arg])
        runtime = Interpreter(project=load_project(self.path), output=output,
                              seed=seed, max_calls=max_calls, **kwargs)
        failure = None
        try:
            runtime.run(source)
        except PurrhapsError as exc:
            failure = type(exc).__name__
        expected = self.records()
        if self.log.exists():
            self.log.unlink()
        result = self.run_native(binary, '--seed', str(seed), '--max-calls', str(max_calls), *native_args)
        self.assertEqual(result.stdout, output.getvalue(), result.stderr)
        self.assertEqual(result.returncode, int(failure is not None), result.stderr)
        self.assertIn(f'calls={runtime.calls}', result.stderr)
        if failure:
            self.assertIn(failure + ':', result.stderr)
        self.assertNotIn('PRIVATE_RATIONALE_MARKER', result.stdout + result.stderr)
        self.assertNotIn('panicked', result.stderr)
        self.assertEqual(self.records(), expected)
        return result, self.records()

    def test_lazy_high_boundary_and_snapshot_single_pipeline(self):
        self.manifest()
        source = '''maybe unused: Int ?= DecisionInput("both-low",[1,2],null)
maybe x: Int ?= DecisionInput("high",[1,2],null)
val s=sample(x)
print(s,s.confidence,s!!,s.resolution.strategy,s.resolution.attempts.size)
if(s.confidence>=0.8) print("metadata branches")
maybe edge: Int ?= DecisionInput("boundary",[3,4],null)
print(edge!!)
'''
        result, records = self.compare(source)
        self.assertEqual(result.stdout, '1 0.93 1 decision 1\nmetadata branches\n3\n')
        self.assertEqual([r['kind'] for r in records], ['decision', 'decision'])

    def test_blind_informed_exact_payload_and_final_rationale(self):
        source = '''maybe x: Int ?= DecisionInput("fallback",[1,2],"original context")
val s=sample(x)
print(s,s.confidence,s.because,s.resolution.strategy,s.resolution.exposure)
print(s.resolution.attempts[0].reportedConfidence,s.resolution.attempts[0].scores)
'''
        binary = self.compile(source)[0]
        for exposure in ('blind', 'informed'):
            with self.subTest(exposure=exposure):
                self.manifest(f'exposure="{exposure}"')
                result, records = self.compare(source, binary=binary)
                self.assertEqual(len(records), 2)
                decision, resolve = records
                self.assertEqual(decision['protocol'], 'purrhaps/2')
                self.assertEqual(resolve['protocol'], 'purrhaps/1')
                self.assertEqual(resolve['kind'], 'resolve')
                self.assertEqual(resolve['input'], decision['input'])
                self.assertEqual(resolve['options'], [1, 2])
                self.assertEqual(resolve['temperature'], .17)
                fields = {'protocol', 'kind', 'input', 'options', 'type', 'line', 'source',
                          'temperature', 'route', 'model', 'model_options'}
                if exposure == 'informed':
                    fields.add('prior_decision')
                    self.assertEqual(resolve['prior_decision'], dict(value=1, confidence=.2, scores=[.99, .99]))
                self.assertEqual(set(resolve), fields)
                self.assertNotIn('PRIVATE_RATIONALE_MARKER', json.dumps(records))
                self.assertIn('Fixture fallback.', result.stdout)

    def test_both_low_default_error_and_budget_prevent_fallback(self):
        self.manifest()
        source = 'maybe x: Int ?= DecisionInput("both-low",[1,2],null); print(x!!)'
        binary = self.compile(source)[0]
        result, records = self.compare(source, binary=binary)
        self.assertIn('UncertaintyException:', result.stderr)
        self.assertEqual(len(records), 2)
        result, records = self.compare(source, binary=binary, max_calls=1)
        self.assertIn('budget exhausted', result.stderr)
        self.assertEqual(len(records), 1)

    def test_random_stream_parity_seed_low_bits_and_repeated_force(self):
        self.manifest('on_low_confidence="random"')
        source = '''maybe x: Int ?= DecisionInput("both-low",[0,1,2,3,4,5,6],null)
print(x!!,x!!,x!!,x!!,x!!)
val s=sample(x)
print(s.confidence,s.resolution.strategy,s.resolution.forced,s.resolution.selectionProbability)
print(s.decision.kind,s.decision.scores,s.resolution.attempts.size)
'''
        binary = self.compile(source)[0]
        for seed in (0, 1, 42, 2**64-1, -1, 2**100+42):
            with self.subTest(seed=seed):
                result, records = self.compare(source, seed=seed, binary=binary)
                self.assertEqual(len(records), 12)
                self.assertIn('0.0 random true', result.stdout)
                self.assertIn('random null 2', result.stdout)

    def test_random_stream_independent_of_demo_and_nonrandom_pipelines(self):
        self.manifest('on_low_confidence="random"')
        source = '''maybe x: Int ?= DecisionInput("both-low",[1,2,3],null)
print(x!!)
with model("offline") {maybe noise: Int="42"; sample(noise)}
maybe accepted: Int ?= DecisionInput("high",[1,2,3],null)
print(accepted!!)
print(x!!)
'''
        result, records = self.compare(source, seed=17)
        self.assertEqual(len(records), 5)
        # Independent SplitMix64 reference; the intervening calls do not advance it.
        self.assertEqual((result.stdout.splitlines()[0], result.stdout.splitlines()[-1]), ('1', '2'))

    def test_random_gate_descendants_and_unrelated_low_operands(self):
        self.manifest('on_low_confidence="random"')
        prelude = '''maybe x: Int ?= DecisionInput("both-low",[1,2],null)
val r=sample(x)
fun identity(v: Int): Int {return v}
'''
        result, _ = self.compare(prelude + 'print((r+1)!!,[r][0]!!,Pair(r,0).first!!,identity(r)!!); if(r>0) print("allowed")')
        self.assertIn('allowed', result.stdout)
        for expression in ('r+low', '[r,low][0]', 'Pair(r,low).first', '[r][low-7]', 'identity(r+low)'):
            with self.subTest(expression=expression):
                result, records = self.compare(prelude + f'maybe q: Int="low"; val low=sample(q); print(({expression})!!)')
                self.assertIn('UncertaintyException:', result.stderr)
                self.assertEqual(len(records), 3)
        result, _ = self.compare(prelude + 'maybe q: Int="low"; val low=sample(q); if((r+low)>0) print("blocked")')
        self.assertIn('UncertaintyException:', result.stderr)

    def test_random_and_context_evidence_cannot_gain_because(self):
        self.manifest('on_low_confidence="random"')
        for suffix in ('print(s.because)', 'print((s+1).because)', 'print(s.resolution.strategy.because)',
                       'maybe generated: Int=s; print(generated.because)'):
            with self.subTest(suffix=suffix):
                result, _ = self.compare('maybe x: Int ?= DecisionInput("both-low",[1,2],null); val s=sample(x);' + suffix)
                self.assertIn('ExplainabilityError:', result.stderr)
        self.manifest()
        source = '''definitely context: Int=DecisionInput("high",[3,4],null)
val c=sample(context)
maybe x: Int ?= DecisionInput("fallback",[1,2],c)
val s=sample(x)
print(s,s.resolution.strategy)
print(s.because)
'''
        result, records = self.compare(source)
        self.assertEqual(len(records), 3)
        self.assertIn('ExplainabilityError:', result.stderr)

    def test_initializer_confidence_controls_threshold_and_evaluates_once(self):
        self.manifest('on_low_confidence="random"')
        source = '''maybe initializer: DecisionInput=DecisionInput("seed-low",[1,2],null)
maybe x: Int ?= initializer
val s=sample(x)
print(s.confidence,s.resolution.strategy,s.resolution.attempts[0]["effectiveConfidence"])
'''
        result, records = self.compare(source)
        self.assertIn('0.0 random 0.3', result.stdout)
        self.assertEqual([r['kind'] for r in records], ['read', 'decision', 'resolve'])
        self.assertEqual(records[1]['input'], records[2]['input'])

    def test_bad_unselected_options_and_invalid_answers_stop_immediately(self):
        self.manifest('on_low_confidence="random"')
        for source in (
            'maybe x: Int ?= DecisionInput("high",[1,true],null); print(x)',
            'maybe x: Int ?= DecisionInput("high",[1,1.0],null); print(x)',
            'maybe x: List<Int> ?= DecisionInput("high",[[1],[true]],null); print(x)',
            'maybe x: Int ?= DecisionInput("high",[1,1],null); print(x)',
            'maybe x: Int ?= 7; print(x)',
        ):
            with self.subTest(source=source):
                result, records = self.compare(source)
                self.assertIn('TypeError:', result.stderr)
                self.assertEqual(records, [])
        for question, count in (('invalid-decision', 1), ('invalid-float', 1), ('invalid-score', 1),
                                ('transport-error', 1), ('invalid-resolve', 2)):
            with self.subTest(question=question):
                result, records = self.compare(f'maybe x: Int ?= DecisionInput("{question}",[1,2],null); print(x)')
                self.assertIn('OracleError:', result.stderr)
                self.assertEqual(len(records), count)

    def test_structural_initializers_and_metadata_gates(self):
        self.manifest()
        source = '''fun recipe(): DecisionInput {return DecisionInput("fallback",[true,false],null)}
val alias=recipe()
maybe x: Bool ?= alias
val s=sample(x)
if(s) print("true") else print("false")
print(s.resolution.strategy)
'''
        self.compare(source)
        for source in ('sample()', 'sample(1,2)', 'print(1.resolution)',
                       'maybe x: Int="low"; val s=sample(x); print(s!!)',
                       'maybe x: Int="low"; val s=sample(x); if(s>0) print("bad")'):
            self.compare(source)
        self.manifest('threshold=0.1')
        source = 'maybe x: Int ?= DecisionInput("both-low",[1,2],null); val s=sample(x); print(s.resolution.strategy!!)'
        result, _ = self.compare(source)
        self.assertIn('UncertaintyException:', result.stderr)

    def test_invalid_sample_arity_does_not_evaluate_arguments(self):
        self.manifest('on_low_confidence="random"')
        source = '''maybe x: Int ?= DecisionInput("both-low",[1,2],null)
sample(sample(x),sample(x))
'''
        result, records = self.compare(source)
        self.assertIn('TypeError:', result.stderr)
        self.assertIn('calls=0', result.stderr)
        self.assertEqual(records, [])

    def test_escalation_threshold_does_not_change_ordinary_collapse_gate(self):
        self.manifest('threshold=0.5')
        prelude = 'maybe x: Int ?= DecisionInput("medium",[1,2],null); val s=sample(x);'
        for suffix in ('print(s!!)', 'if(s>0) print("blocked")'):
            result, records = self.compare(prelude + 'print(s,s.confidence,s.resolution.strategy);' + suffix)
            self.assertEqual(result.stdout, '1 0.6 decision\n')
            self.assertIn('UncertaintyException:', result.stderr)
            self.assertEqual(len(records), 1)

    def test_resolution_history_carries_actual_decision_and_conservative_gate(self):
        self.manifest('on_low_confidence="random"')
        prelude = '''maybe x: Int ?= DecisionInput("both-low",[1,2],null)
val s=sample(x)
'''
        result, _ = self.compare(prelude + 'print(s.resolution.decision,s.resolution.attempts[0].reportedConfidence); print(s.resolution.strategy!!)')
        self.assertIn('UncertaintyException:', result.stderr)
        evidence = json.loads(result.stdout.splitlines()[0].rsplit(' ', 1)[0])
        self.assertEqual(evidence['kind'], 'decision')
        self.assertEqual(evidence['selected'], 1)
        self.assertEqual(evidence['options'], [1, 2])
        self.assertEqual(evidence['scores'], [.99, .99])
        self.assertEqual(evidence['input'], dict(question='both-low', options=[1, 2], context=None))
        self.manifest()
        prelude = '''maybe x: Int ?= DecisionInput("fallback",[1,2],null)
val s=sample(x)
fun identity(v: Json): Json {return v}
'''
        for expression in ('s.resolution.attempts[0].value',
                           '[s.resolution][0].attempts[0].value',
                           'identity(s.resolution).attempts[0].value'):
            with self.subTest(expression=expression):
                result, _ = self.compare(prelude + f'print(s.because); print(({expression}).because)')
                self.assertIn('Fixture fallback.', result.stdout)
                self.assertIn('ExplainabilityError:', result.stderr)
        result, _ = self.compare(prelude + 'maybe g: Int=s.resolution.attempts[0].value; print(g.because)')
        self.assertIn('ExplainabilityError:', result.stderr)

    def test_resolution_extraction_preserves_unrelated_low_gate(self):
        self.manifest()
        source = '''maybe x: Int ?= DecisionInput("high",[1,2],null)
val s=sample(x)
maybe q: Int="low"
val low=sample(q)
val combined=Pair(s,low)
print(combined.resolution.strategy!!)
'''
        result, _ = self.compare(source)
        self.assertIn('UncertaintyException:', result.stderr)

        # Repeated access cannot restore an earlier, less restrictive history gate.
        repeated = source.replace('combined.resolution.strategy!!', 'combined.resolution.resolution.strategy!!')
        result, _ = self.compare(repeated)
        self.assertIn('UncertaintyException:', result.stderr)

    def test_routes_explicit_overrides_scope_temperature_and_return_restore(self):
        source = '''maybe x: Int ?= DecisionInput("fallback",[1,2],null)
with model("generation2") {with model("decision2") {with temperature(0.4) {print(x)}}}
fun leave(): Int {with model("generation2") {return x!!}; return 0}
print(leave(),x)
'''
        binary = self.compile(source)[0]
        for explicit in (False, True):
            self.manifest('generative="generation"\ndecision="decision"' if explicit else '')
            _, records = self.compare(source, binary=binary)
            self.assertEqual([r['route'] for r in records],
                             ['decision', 'generation'] * 3 if explicit else
                             ['decision2', 'generation2', 'decision', 'generation2', 'decision', 'generation'])
            self.assertEqual(records[1]['temperature'], .4)
            self.assertEqual(records[-1]['temperature'], .17)

    def test_legacy_command_override_and_named_route_trace_provenance(self):
        source = '''maybe x: Int ?= DecisionInput("fallback",[1,2],null)
with temperature(0.4) {
val s=sample(x)
print(s.resolution.attempts[0].route,s.resolution.attempts[1].route)
with model("generation2") {with model("decision2") {
val named=sample(x)
print(named.resolution.attempts[0].route,named.resolution.attempts[1].route)
}}
val restored=sample(x)
print(restored.resolution.attempts[0].route,restored.resolution.attempts[1].route)
}
'''
        self.manifest()
        binary = self.compile(source)[0]
        result, records = self.compare(source, binary=binary, legacy_overrides=True)
        self.assertEqual(result.stdout, 'null null\ndecision2 generation2\nnull null\n')
        self.assertEqual([r.get('route') for r in records],
                         [None, None, 'decision2', 'generation2', None, None])
        self.manifest('decision="decision"\ngenerative="generation"')
        result, records = self.compare(source, binary=binary, legacy_overrides=True)
        self.assertEqual(result.stdout, 'decision generation\ndecision generation\ndecision generation\n')
        self.assertEqual([r.get('route') for r in records], ['decision', 'generation'] * 3)

    def test_manifest_strict_validation_and_emitted_relocation(self):
        self.manifest()
        source = 'maybe x: Int ?= DecisionInput("high",[1,2],null); print(x!!)'
        binary, emitted = self.compile(source, project=load_project(self.path))
        original = self.path.read_text()
        for bad in ('threshold=true', 'threshold=-0.1', 'threshold=1.1', 'threshold=nan',
                    'threshold=inf', 'threshold="0.8"', 'exposure="visible"',
                    'on_low_confidence="human"', 'on_low_confidence=true', 'unknown=1',
                    'decision="generation"', 'generative="missing"', 'decision=false'):
            with self.subTest(bad=bad):
                self.path.write_text(original + bad + '\n')
                result = self.run_native(binary, '--check')
                self.assertIn('ConfigurationError:', result.stderr)
                self.assertEqual(self.records(), [])
        self.path.write_text(original + 'on_low_confidence="random"\nexposure="informed"\n')
        source = 'maybe x: Int ?= DecisionInput("both-low",[1,2],null); val s=sample(x); print(s!!,s.resolution.strategy,s.resolution.exposure)'
        binary, emitted = self.compile(source, project=load_project(self.path))
        self.path.unlink()
        relocated = self.work / 'relocated'
        emitted.rename(relocated)
        result = self.run_native(binary, cwd=relocated)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, '2 random informed\n')

    @contextmanager
    def server(self, answer):
        requests = []
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                requests.append(body)
                data = json.dumps(answer(body)).encode()
                self.send_response(200)
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            def log_message(self, *args):
                pass
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f'http://127.0.0.1:{server.server_port}', requests
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    @unittest.skipUnless(shutil.which('curl'), 'native HTTP requires curl')
    def test_http_resolve_enum_rationale_and_strict_float_identity(self):
        def answer(body):
            request = json.loads(body['messages'][-1]['content'])
            if request['kind'] == 'decision':
                data = dict(value=request['options'][0], confidence=.2, scores=None)
            else:
                data = dict(value=request['options'][-1], confidence=.95, because='Fixture fallback.', citations=[], caveats=[])
                if request['input']['question'] == 'resolve-int':
                    data['value'] = 1
            return dict(choices=[dict(message=dict(content=json.dumps(data)), finish_reason='stop')])
        with self.server(answer) as (url, requests):
            self.manifest(provider=f'transport="openai-chat"\nbase_url="{url}"')
            source = 'maybe x: List<Int> ?= DecisionInput("fallback",[[1],[2]],null); val s=sample(x); print(s!!,s.because)'
            result = self.run_native(self.compile(source)[0])
            self.assertEqual(result.returncode, 0, result.stderr)
            schema = requests[1]['response_format']['json_schema']['schema']
            self.assertEqual(schema['properties']['value'], {'enum': [[1], [2]]})
            self.assertIn('because', schema['required'])
            self.assertNotIn('scores', schema['properties'])
            source = 'maybe x: Float ?= DecisionInput("resolve-int",[1.0,2.0],null); print(x)'
            result = self.run_native(self.compile(source)[0])
            self.assertIn('OracleError:', result.stderr)
            self.assertEqual(len(requests), 4)

    def use_command_decision_lane(self):
        source = self.path.read_text().replace(
            '[providers.demo]', '[providers.command]\ntransport="command"\ncommand=' +
            json.dumps([sys.executable, str(self.oracle), str(self.log)]) + '\n[providers.demo]')
        for model in ('d', 'd2'):
            source = source.replace(f'[models.{model}]\nprovider="fixture"',
                                    f'[models.{model}]\nprovider="command"')
        self.path.write_text(source)

    @unittest.skipUnless(shutil.which('curl'), 'native HTTP requires curl')
    def test_http_and_ollama_resolve_existing_records_without_telemetry(self):
        for transport in ('openai-chat', 'ollama'):
            def answer(body):
                request = json.loads(body['messages'][-1]['content'])
                data = dict(value=request['options'][-1], confidence=.95,
                            because='Fixture fallback.', citations=[], caveats=[])
                if transport == 'ollama':
                    return dict(message=dict(content=json.dumps(data)), done=True, done_reason='stop')
                return dict(choices=[dict(message=dict(content=json.dumps(data)), finish_reason='stop')])
            with self.subTest(transport=transport), self.server(answer) as (url, requests):
                self.manifest(provider=f'transport="{transport}"\nbase_url="{url}"')
                self.use_command_decision_lane()
                source = '''maybe x: ModelResponse ?= DecisionInput("fallback",[ModelResponse("one","option-one",TokenUsage(1,2)),ModelResponse("two","option-two",TokenUsage(3,4))],null)
val s=sample(x)
print(s.text,s.finishReason,s.usage.inputTokens,s.usage.outputTokens,s.because)
'''
                result, _ = self.compare(source)
                self.assertEqual(result.stdout, 'two option-two 3 4 Fixture fallback.\n')
                body = requests[-1]
                schema = body['format'] if transport == 'ollama' else body['response_format']['json_schema']['schema']
                self.assertEqual(schema['properties']['value']['enum'][-1],
                                 dict(text='two', finishReason='option-two', usage=dict(inputTokens=3, outputTokens=4)))
                self.assertEqual(len(requests), 2)  # Python then native, each one fallback.

    def test_harness_resolve_existing_modelresponse_and_json_options(self):
        harness = self.work / 'harness.py'
        harness.write_text('''import json,pathlib,sys
args=sys.argv[1:]; prompt=sys.stdin.read()
r=json.loads(prompt.split('\\nRequest: ',1)[1])
schema=json.loads(pathlib.Path(args[args.index('--output-schema')+1]).read_text())
assert schema['properties']['value']=={'enum':r['options']}
assert 'because' in schema['required'] and 'scores' not in schema['properties']
a=dict(value=r['options'][-1],confidence=.96,because='Fixture fallback.',citations=[],caveats=[])
pathlib.Path(args[args.index('--output-last-message')+1]).write_text(json.dumps(a))
''')
        self.manifest(provider='transport="codex"\ncommand=' + json.dumps([sys.executable, str(harness)]))
        self.path.write_text(self.path.read_text().replace('temperature=0.17\n', '').replace('temperature=0.23\n', ''))
        self.use_command_decision_lane()
        source = '''maybe record: ModelResponse ?= DecisionInput("fallback",[ModelResponse("one","stop",TokenUsage(1,2)),ModelResponse("two","existing",TokenUsage(3,4))],null)
val s=sample(record)
print(s.text,s.finishReason,s.usage.inputTokens,s.because)
maybe dynamic: Json ?= DecisionInput("fallback",[mapOf("x" to 1),[true,1,1.0]],null)
print(dynamic!!)
'''
        result, _ = self.compare(source)
        self.assertEqual(result.stdout, 'two existing 3 Fixture fallback.\n[true,1,1.0]\n')

    def test_shared_system_prompt_matches_exactly(self):
        from purrhaps.providers import SYSTEM
        native = Path(__file__).resolve().parents[1] / 'purrhaps/rust/providers.rs'
        constant = native.read_text().split('const SYSTEM: &str = ', 1)[1].split(';\n', 1)[0]
        self.assertEqual(ast.literal_eval(constant), SYSTEM)

    def test_native_internal_error_restoration_and_rng_accounting(self):
        _, emitted = self.compile('print(hi())')
        result = subprocess.run(['cargo', 'test', '--offline', '--quiet', '--manifest-path',
                                 str(emitted / 'Cargo.toml')], env=self.env,
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('1 passed', result.stdout)


if __name__ == '__main__':
    unittest.main()
