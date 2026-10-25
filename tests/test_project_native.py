"""Native manifests/routing verified against local HTTP and fake harnesses only."""
from contextlib import contextmanager
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import tomllib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import unittest

from purrhaps.project import load_project, ConfigurationError
from purrhaps import Interpreter
from purrhaps.transpile import transpile


@unittest.skipUnless(shutil.which('cargo'), 'native integration requires Cargo')
class ProjectNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='purrhaps-project-native-')
        cls.root = Path(cls.temp.name)
        cls.counter = 0
        cls.build_env = dict(os.environ, CARGO_TARGET_DIR=str(cls.root / 'target'))
        cls.oracle = cls.root / 'oracle.py'
        cls.oracle.write_text('''import json, sys, os, time
r=json.load(sys.stdin)
with open(sys.argv[1], 'a') as f: f.write(json.dumps(dict(r, cwd=os.getcwd()))+'\\n')
if r['input']=='slow': time.sleep(2)
if r['input']=='exit':
    print('PRIVATE_FIXTURE_MARKER',file=sys.stderr); sys.exit(9)
if r['input']=='invalid': print('PRIVATE_FIXTURE_MARKER'); sys.exit(0)
value=r['options'][-1] if 'options' in r else r['input']
if value=='route': value=r.get('route','override')
a=dict(value=value,confidence=.95)
if 'options' in r: a['scores']=[.5]*len(r['options'])
else: a.update(because='Scripted evidence.',citations=[r['line']],caveats=[])
print(json.dumps(a))
''')
        cls.harness = cls.root / 'harness.py'
        cls.harness.write_text('''import json, sys, os, pathlib, time
log, mode = sys.argv[1:3]; args=sys.argv[3:]; prompt=sys.stdin.read()
with open(log,'a') as f: f.write(json.dumps(dict(args=args,cwd=os.getcwd(),prompt=prompt))+'\\n')
if mode=='slow': time.sleep(2)
if mode=='exit': print('PRIVATE_FIXTURE_MARKER',file=sys.stderr); sys.exit(7)
a=dict(value='harness',confidence=.95,because='Scripted evidence.',citations=[1],caveats=[])
if 'exec' in args:
    schema=pathlib.Path(args[args.index('--output-schema')+1]); assert json.loads(schema.read_text())['type']=='object'
    output=pathlib.Path(args[args.index('--output-last-message')+1]); output.write_text(json.dumps(a))
    print('PRIVATE_FIXTURE_MARKER (ignored logs)')
else:
    e=dict(type='result',subtype='success',is_error=False,structured_output=a)
    if mode=='refuse': e.update(subtype='error_during_execution',is_error=True,result='PRIVATE_FIXTURE_MARKER')
    print(json.dumps(e))
''')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        type(self).counter += 1
        self.work = self.root / f'case-{self.counter}'
        self.work.mkdir()
        self.log = self.work / 'requests.jsonl'

    def manifest(self, provider=None, *, options='', temperature=None, runtime='', extra=''):
        provider = provider or f'transport="command"\ncommand={json.dumps([sys.executable,str(self.oracle),str(self.log)])}'
        source = f'''manifest_version=1
[project]
name="native fixture"
entry="main.prr"
{runtime}
[providers.main]
{provider}
[providers.offline]
transport="demo"
[models.write]
provider="main"
kind="generative"
model="fixture/model"
{f'temperature={temperature}' if temperature is not None else ''}
{options}
[models.careful]
provider="main"
kind="generative"
model="fixture/careful"
[models.decide]
provider="main"
kind="decision"
model="fixture/decide"
[models.demo_decide]
provider="offline"
kind="decision"
[routes]
writing="write"
careful="careful"
triage="decide"
offline_decision="demo_decide"
[defaults]
generative="writing"
decision="triage"
{extra}
'''
        path = self.work / 'purrhaps.toml'
        path.write_text(source)
        return path

    def compile(self, source='print(hi())', project=None):
        # Each binary has a separate emitted source tree, sharing only Cargo cache.
        destination = self.work / f'build-{len(list(self.work.glob("build-*")))}'
        transpile(source, destination, project=project)
        result = subprocess.run(['cargo','build','--offline','--quiet','--manifest-path',str(destination/'Cargo.toml')],
                                env=self.build_env,capture_output=True,text=True,timeout=120)
        self.assertEqual(result.returncode,0,result.stderr)
        return self.root/'target/debug/purrhaps-program', destination

    def run_native(self, binary, *args, cwd=None, env=None):
        return subprocess.run([str(binary),*args],cwd=cwd or self.work,env=env,capture_output=True,text=True,timeout=10)

    def records(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []

    def test_command_routes_lazy_cache_lane_isolation_and_return_restore(self):
        self.manifest(temperature=.15)
        source = '''perhaps x: String="route"
with model("careful") {
    print(x,x)
    with model("writing") { print(x) }
    print(x)
    with model("offline_decision") { definitely d=DecisionInput("pick",[1,2],null); print(d!!) }
    definitely d=DecisionInput("pick",[1,2],null); print(d!!)
}
print(x)
fun leave(): String { with model("careful") { return x!! }; return "no" }
print(leave(),x)
'''
        binary,_=self.compile(source)
        result=self.run_native(binary,'--stats')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(result.stdout,'careful careful\nwriting\ncareful\n1\n2\nwriting\ncareful writing\n')
        records=self.records()
        self.assertEqual([r['route'] for r in records],['careful','writing','triage','writing','careful'])
        self.assertIn('calls=6',result.stderr)
        self.assertEqual(records[0]['model'],'fixture/careful')
        self.assertEqual(records[0]['cwd'],str(self.work))
        self.assertEqual(records[1]['temperature'],.15)

    def test_declared_in_block_forced_outside_uses_active_route(self):
        self.manifest()
        binary,_=self.compile('fun deferred() { with model("careful") { maybe x: String="route"; return x } }; print(deferred())')
        result=self.run_native(binary)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(result.stdout,'writing\n')
        self.assertEqual(self.records()[0]['route'],'writing')

    def test_temperature_profile_explicit_scope_and_options_request_parity(self):
        self.manifest(temperature=.1, options='options={reasoning_effort="max", max_tokens=17}')
        source='maybe x: String="route"; print(x); with temperature(0.3) { print(x); with model("careful") { print(x) } }; print(x)'
        binary,_=self.compile(source)
        native=self.run_native(binary,'--stats')
        native_records=self.records(); self.log.unlink()
        output=io.StringIO(); runtime=Interpreter(project=load_project(self.work), output=output)
        runtime.run(source)
        self.assertEqual(native.returncode,0,native.stderr)
        self.assertEqual(native.stdout,output.getvalue())
        self.assertEqual(native_records,self.records())
        self.assertEqual([r['temperature'] for r in native_records],[.1,.3,.3,.1])
        self.assertEqual(native_records[0]['model_options'],{'reasoning_effort':'max','max_tokens':17})

    def test_cli_routes_explicit_command_override_and_named_blocks(self):
        self.manifest()
        binary,_=self.compile('maybe x: String="route"; print(x); with model("careful") { print(x) }; print(x)')
        result=self.run_native(binary,'--model','careful')
        self.assertEqual(result.stdout,'careful\ncareful\ncareful\n',result.stderr)
        result=self.run_native(binary,'--oracle',sys.executable,'--oracle-arg',str(self.oracle),'--oracle-arg',str(self.log))
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(result.stdout,'override\ncareful\noverride\n')
        self.assertNotIn('route',self.records()[-1])
        self.assertNotIn('model_options',self.records()[-1])
        for args in [('--model','careful','--oracle','false'),('--decision-model','triage','--decision-oracle','false'),('--no-project','--project',str(self.work))]:
            result=self.run_native(binary,*args)
            self.assertEqual(result.returncode,2,result.stderr)
            self.assertIn('ArgumentError',result.stderr)

    def test_no_project_and_missing_wrong_kind_routes_never_fall_back(self):
        self.manifest()
        binary,_=self.compile('with model("careful") { print(1) }')
        for args in [('--no-project',),('--model','missing'),('--decision-model','writing'),('--model','triage')]:
            result=self.run_native(binary,*args)
            self.assertEqual(result.returncode,1,result.stderr)
            self.assertIn('ConfigurationError',result.stderr)
        self.assertEqual(self.records(),[])
        binary,_=self.compile('with model(print("must not evaluate")) {print(1)}')
        result=self.run_native(binary,'--no-project')
        self.assertIn('ConfigurationError',result.stderr)
        self.assertEqual(result.stdout,'')
        binary,_=self.compile('maybe name="careful"; with model(name) {print(1)}')
        result=self.run_native(binary)
        self.assertIn('TypeError',result.stderr)
        self.assertEqual(self.records(),[])

    def test_manifest_budget_shared_across_lanes_and_cli_precedence(self):
        self.manifest(runtime='[runtime]\nmax_calls=1\ntimeout=1')
        binary,_=self.compile('maybe x: String="route"; print(x); definitely d=DecisionInput("pick",[1,2],null); print(d!!)')
        result=self.run_native(binary,'--stats')
        self.assertEqual(result.returncode,1,result.stderr)
        self.assertIn('calls=1',result.stderr)
        self.assertIn('budget exhausted',result.stderr)
        result=self.run_native(binary,'--max-calls','2','--stats')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('calls=2',result.stderr)

    def test_timeout_precedence_and_hidden_child_failures(self):
        self.manifest(runtime='[runtime]\ntimeout=0.03')
        binary,_=self.compile('maybe x="slow"; print(x)')
        result=self.run_native(binary)
        self.assertEqual(result.returncode,1,result.stderr)
        self.assertIn('timed out',result.stderr)
        result=self.run_native(binary,'--timeout','.05')
        self.assertIn('timed out',result.stderr)
        for value in ['exit','invalid']:
            binary,_=self.compile(f'maybe x="{value}"; print(x)')
            result=self.run_native(binary)
            self.assertEqual(result.returncode,1,result.stderr)
            self.assertNotIn('PRIVATE_FIXTURE_MARKER',result.stdout+result.stderr)

    def test_discovery_closest_manifest_check_and_emission_relocation(self):
        path=self.manifest()
        path.write_text(path.read_text().replace('[project]', '[project]\nno_humans_allowed=true'))
        project=load_project(path)
        path.write_text('# PRIVATE_FIXTURE_MARKER in a later source comment\n'+path.read_text())
        binary,emitted=self.compile('maybe x: String="route"; print(x)',project)
        self.assertEqual(tomllib.loads((emitted/'purrhaps.toml').read_text()),project.data)
        self.assertNotIn('PRIVATE_FIXTURE_MARKER',(emitted/'purrhaps.toml').read_text())
        self.assertIn('mod project', (emitted/'src/runtime.rs').read_text())
        self.assertTrue((emitted/'src/providers.rs').is_file())
        nested=emitted/'nested'; nested.mkdir()
        result=self.run_native(binary,cwd=nested)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(self.records()[-1]['cwd'],str(emitted))
        result=self.run_native(binary,'--project',str(path),'--check',env=dict(os.environ,PATH=''))
        self.assertEqual(result.returncode,0,result.stderr)
        path.write_text(path.read_text().replace('no_humans_allowed=true', 'no_humans_allowed=false'))
        result=self.run_native(binary,'--project',str(path),'--check',env=dict(os.environ,PATH=''))
        self.assertEqual(result.returncode,0,result.stderr)
        result=self.run_native(binary,'--no-project',env={'PATH':''})
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(result.stdout,'route\n')

    def test_manifest_validation_fails_without_leaking_values_or_launching(self):
        path=self.manifest(); valid=path.read_text()
        binary,_=self.compile('print(hi())')
        corruptions=[
            *(valid.replace('[project]', '[project]\nno_humans_allowed='+value)
              for value in ('0', '1', '"true"', '"PRIVATE_FIXTURE_MARKER"', '[]', '{}')),
            valid.replace('manifest_version=1','manifest_version=2'),
            valid.replace('manifest_version=1','manifest_version=true'),
            valid.replace('name="native fixture"','name=""'),
            valid.replace('entry="main.prr"','entry="/PRIVATE_FIXTURE_MARKER"'),
            valid.replace('transport="command"','transport="PRIVATE_FIXTURE_MARKER"'),
            valid.replace('transport="command"','transport="demo"'),
            valid.replace('provider="main"','provider="PRIVATE_FIXTURE_MARKER"'),
            valid.replace('writing="write"','writing="PRIVATE_FIXTURE_MARKER"'),
            valid.replace('generative="writing"','generative="triage"'),
            valid.replace('model="fixture/model"','temperature=nan'),
            valid.replace('model="fixture/model"','timeout=0'),
            valid.replace('model="fixture/model"','options={token="PRIVATE_FIXTURE_MARKER"}'),
            valid.replace('model="fixture/model"','options={nested={x-api-key="PRIVATE_FIXTURE_MARKER"}}'),
            valid.replace('model="fixture/model"','options={nested={Authorization="PRIVATE_FIXTURE_MARKER"}}'),
            valid.replace('model="fixture/model"','options={date=2026-10-25}'),
            valid.replace('model="fixture/model"','options={n=inf}'),
            valid.replace('name="native fixture"','unknown="PRIVATE_FIXTURE_MARKER"'),
            valid+'\n[PRIVATE_FIXTURE_MARKER]\nx=1',
            'PRIVATE_FIXTURE_MARKER = "unterminated',
        ]
        for source in corruptions:
            path.write_text(source)
            with self.subTest(source=source[:80]):
                result=self.run_native(binary,'--check')
                self.assertEqual(result.returncode,1,result.stderr)
                self.assertIn('ConfigurationError',result.stderr)
                self.assertNotIn('PRIVATE_FIXTURE_MARKER',result.stdout+result.stderr)
                with self.assertRaises(ConfigurationError): load_project(path)
        self.assertEqual(self.records(),[])

    @contextmanager
    def server(self, responder):
        captured=[]
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                captured.append((self.path,body,dict(self.headers)))
                status,data=responder(body)
                self.send_response(status); self.send_header('Content-Type','application/json'); self.end_headers()
                self.wfile.write(json.dumps(data).encode())
            def log_message(self,*args): pass
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
        try: yield f'http://127.0.0.1:{server.server_port}',captured
        finally: server.shutdown(); server.server_close(); thread.join()

    def openai(self, body, value='http', **overrides):
        request=json.loads(body['messages'][-1]['content'])
        if request['kind'] in ('decision','difference'):
            answer=dict(value=request['options'][-1],confidence=.94,scores=None)
        else:
            answer=dict(value=value,confidence=.94,because='Observed fixture.',citations=[request['line']],caveats=[])
        result=dict(choices=[dict(message=dict(content=json.dumps(answer)),finish_reason='stop')],usage=dict(prompt_tokens=13,completion_tokens=5))
        result.update(overrides)
        return result

    @unittest.skipUnless(shutil.which('curl'),'native HTTP requires curl')
    def test_openai_body_options_temperature_and_typed_decisions(self):
        with self.server(lambda body:(200,self.openai(body))) as (url,requests):
            self.manifest(f'transport="openai-chat"\nbase_url="{url}/v1"',options='options={reasoning_effort="max", max_completion_tokens=29, provider={order=["fixture"]}}')
            binary,_=self.compile('maybe x: String="input"; print(x); with temperature(0.2) {print(x)}; definitely d: Int=DecisionInput("pick",[1,2],null); print(d!!)')
            result=self.run_native(binary,'--stats')
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(result.stdout,'http\nhttp\n2\n')
            self.assertEqual(len(requests),3)
            self.assertTrue(all(r[0]=='/v1/chat/completions' for r in requests))
            self.assertNotIn('temperature',requests[0][1])
            self.assertEqual(requests[1][1]['temperature'],.2)
            self.assertEqual(requests[0][1]['reasoning_effort'],'max')
            self.assertEqual(requests[0][1]['provider'],{'order':['fixture']})
            self.assertEqual(requests[2][1]['response_format']['json_schema']['schema']['properties']['value'],{'enum':[1,2]})

    @unittest.skipUnless(shutil.which('curl'),'native HTTP requires curl')
    def test_openai_modelresponse_real_telemetry_and_recursive_float_normalization(self):
        values=iter([{'text':'observed'},[{'first':1,'second':[2]}],None])
        with self.server(lambda body:(200,self.openai(body,next(values)))) as (url,requests):
            self.manifest(f'transport="openai-chat"\nbase_url="{url}"')
            source='maybe x: ModelResponse="input"; val r=x!!; print(r.text,r.finishReason,r.usage.inputTokens,r.usage.outputTokens); maybe xs: List<Pair<Float,List<Float>>>=null; print(xs!!); maybe nil: ModelResponse?=null; print(nil!!)'
            binary,_=self.compile(source)
            result=self.run_native(binary)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('observed stop 13 5',result.stdout)
            self.assertIn('1.0',result.stdout)
            self.assertTrue(result.stdout.endswith('null\n'))
            schema=requests[0][1]['response_format']['json_schema']['schema']['properties']['value']
            self.assertEqual(schema['required'],['text'])

    @unittest.skipUnless(shutil.which('curl'),'native HTTP requires curl')
    def test_http_finish_refusal_redirect_and_missing_usage_fail_without_retry(self):
        mode=['length']
        def responder(body):
            envelope=self.openai(body,{'text':'hello'})
            if mode[0]=='status': return 503,{'private':'PRIVATE_FIXTURE_MARKER'}
            if mode[0]=='redirect': return 302,{'private':'PRIVATE_FIXTURE_MARKER'}
            if mode[0]=='missing-usage': del envelope['usage']
            elif mode[0]=='refusal': envelope['choices'][0]['message']['refusal']='PRIVATE_FIXTURE_MARKER'
            elif mode[0]=='tool': envelope['choices'][0]['message']['tool_calls']=[{'private':'PRIVATE_FIXTURE_MARKER'}]
            else: envelope['choices'][0]['finish_reason']=mode[0]
            return 200,envelope
        with self.server(responder) as (url,requests):
            self.manifest(f'transport="openai-chat"\nbase_url="{url}"')
            binary,_=self.compile('maybe x: ModelResponse="input"; print(x!!)')
            for m in ['length','refusal','tool','missing-usage','status','redirect']:
                mode[0]=m; before=len(requests)
                result=self.run_native(binary)
                self.assertEqual(result.returncode,1,result.stderr)
                self.assertIn('Error:',result.stderr)
                self.assertNotIn('PRIVATE_FIXTURE_MARKER',result.stdout+result.stderr)
                self.assertEqual(len(requests),before+1)

    @unittest.skipUnless(shutil.which('curl'),'native HTTP requires curl')
    def test_http_missing_env_only_at_force_and_curlrc_disabled(self):
        with self.server(lambda body:(200,self.openai(body))) as (url,requests):
            self.manifest(f'transport="openai-chat"\nbase_url="{url}"\napi_key_env="PURRHAPS_NATIVE_MISSING_FIXTURE"')
            binary,_=self.compile('maybe x: String="input"; print(x)')
            env=dict(os.environ); env.pop('PURRHAPS_NATIVE_MISSING_FIXTURE',None)
            result=self.run_native(binary,'--check',env=dict(env,PATH=''))
            self.assertEqual(result.returncode,0,result.stderr)
            result=self.run_native(binary,env=env)
            self.assertIn('PURRHAPS_NATIVE_MISSING_FIXTURE',result.stderr)
            self.assertEqual(requests,[])
            # A synthetic fixture value, never an account credential.
            env['PURRHAPS_NATIVE_MISSING_FIXTURE']='synthetic-fixture-value'
            curlhome=self.work/'fake-curl-home'; curlhome.mkdir()
            (curlhome/'.curlrc').write_text('invalid-fixture-option\n')
            env['CURL_HOME']=str(curlhome)
            result=self.run_native(binary,env=env)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(len(requests),1)
            self.assertEqual(requests[0][2]['Authorization'],'Bearer synthetic-fixture-value')

    def test_strict_schema_preflight_and_harness_temperature_rejected_before_spawn(self):
        provider=f'transport="codex"\ncommand={json.dumps([sys.executable,str(self.harness),str(self.log),"normal"])}'
        # Harness provider cannot have a decision profile; retain an explicit offline lane.
        path=self.manifest(provider)
        path.write_text(path.read_text().replace('[models.decide]\nprovider="main"','[models.decide]\nprovider="offline"'))
        for source,needle in [
            ('maybe x="data"; print(x)','strict output'),
            ('maybe x: Map<String,Int>=mapOf("x" to 1); print(x)','strict output'),
            ('maybe x: List=[1]; print(x)','strict output'),
            ('maybe x: ModelResponse="data"; print(x)','telemetry'),
            ('maybe x: ModelResponse?=null; print(x)','telemetry'),
            ('with temperature(0.1) { maybe x: String="data"; print(x) }','temperature'),
        ]:
            with self.subTest(source=source):
                binary,_=self.compile(source); result=self.run_native(binary)
                self.assertEqual(result.returncode,1,result.stderr)
                self.assertIn(needle,result.stderr)
        self.assertEqual(self.records(),[])

    def test_harness_exact_flags_options_isolation_cleanup_and_claude_failures(self):
        for transport,options in [('codex','options={reasoning_effort="max",service_tier="priority"}'),('claude','options={effort="max"}')]:
            with self.subTest(transport=transport):
                path=self.manifest(f'transport="{transport}"\ncommand={json.dumps([sys.executable,str(self.harness),str(self.log),"normal"])}',options=options)
                path.write_text(path.read_text().replace('[models.decide]\nprovider="main"','[models.decide]\nprovider="offline"'))
                binary,_=self.compile('maybe x: String="data"; print(x!!)')
                result=self.run_native(binary)
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertEqual(result.stdout,'harness\n')
                self.assertNotIn('PRIVATE_FIXTURE_MARKER',result.stdout+result.stderr)
                record=self.records()[-1]; args=record['args']; cwd=Path(record['cwd'])
                self.assertNotEqual(cwd,self.work)
                self.assertFalse(cwd.exists())
                self.assertIn('Request:',record['prompt'])
                self.assertIn('--model',args)
                if transport=='codex':
                    self.assertIn('model_reasoning_effort="max"',args)
                    self.assertIn('service_tier="priority"',args)
                    self.assertEqual(args[:3],['-a','never','exec'])
                    for flag in ['--ignore-user-config','--ephemeral','--skip-git-repo-check','--sandbox','read-only']:
                        self.assertIn(flag,args)
                else:
                    for flag in ['--strict-mcp-config','--no-session-persistence','--disable-slash-commands','--safe-mode']:
                        self.assertIn(flag,args)
                    self.assertEqual(args[args.index('--effort')+1],'max')
                    self.assertEqual(args[args.index('--tools')+1],'')
                    self.assertEqual(args[args.index('--setting-sources')+1],'')
                    self.assertNotIn('--bare',args)
                    path.write_text(path.read_text().replace('"normal"','"refuse"'))
                    result=self.run_native(binary)
                    self.assertEqual(result.returncode,1,result.stderr)
                    self.assertNotIn('PRIVATE_FIXTURE_MARKER',result.stdout+result.stderr)

    @unittest.skipUnless(shutil.which('curl'),'native HTTP requires curl')
    def test_json_object_explicitly_allows_dynamic_json_and_no_fallback(self):
        with self.server(lambda body:(200,self.openai(body,{'dynamic':[1,True]}))) as (url,requests):
            path=self.manifest(f'transport="openai-chat"\nbase_url="{url}"')
            binary,_=self.compile('maybe x: Json="data"; print(x)')
            result=self.run_native(binary)
            self.assertIn('strict output',result.stderr)
            self.assertEqual(requests,[])
            path.write_text(path.read_text().replace(f'base_url="{url}"',f'base_url="{url}"\nresponse_format="json_object"'))
            result=self.run_native(binary)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(len(requests),1)
            self.assertEqual(requests[0][1]['response_format'],{'type':'json_object'})

    @unittest.skipUnless(shutil.which('curl'),'native HTTP requires curl')
    def test_ollama_generation_and_specialist_decision(self):
        def responder(body):
            if 'questions' in body:
                return 200,{'answers':{'purrhaps':{'type':'choice','choice':'option_1','confidence':.9,'probabilities':{'option_0':.2,'option_1':.8}}}}
            request=json.loads(body['messages'][-1]['content'])
            answer=dict(value='ollama',confidence=.9,because='Observed fixture.',citations=[request['line']],caveats=[])
            return 200,dict(message=dict(content=json.dumps(answer)),done=True,done_reason='stop')
        with self.server(responder) as (url,requests):
            path=self.manifest(f'transport="ollama"\nbase_url="{url}"',temperature=.25,options='options={num_predict=23}')
            source=path.read_text().replace('[providers.offline]','[providers.specialist]\ntransport="ollama-decision"\nbase_url="'+url+'"\n[providers.offline]').replace('[models.decide]\nprovider="main"','[models.decide]\nprovider="specialist"')
            path.write_text(source)
            binary,_=self.compile('maybe x: String="data"; print(x); definitely d: Int=DecisionInput("pick",[1,2],null); print(d!!,d.decision.scores)')
            result=self.run_native(binary)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertTrue(result.stdout.startswith('ollama\n2 '),result.stdout)
            self.assertEqual(requests[0][0],'/api/chat')
            self.assertEqual(requests[0][1]['options'],{'temperature':.25,'num_predict':23})
            self.assertEqual(requests[1][0],'/v1/systemone')
            self.assertEqual(requests[1][1]['questions']['purrhaps']['criteria'],{'option_0':'1','option_1':'2'})
            self.assertNotIn('options',requests[1][1])


    @unittest.skipUnless(shutil.which('curl'),'native HTTP requires curl')
    def test_http_requires_complete_envelope_even_in_json_object_mode(self):
        def responder(body):
            envelope=self.openai(body)
            answer=json.loads(envelope['choices'][0]['message']['content'])
            for key in ['citations','caveats','scores']: answer.pop(key,None)
            envelope['choices'][0]['message']['content']=json.dumps(answer)
            return 200,envelope
        with self.server(responder) as (url,requests):
            self.manifest(f'transport="openai-chat"\nbase_url="{url}"\nresponse_format="json_object"')
            binary,_=self.compile('maybe x: String="input"; print(x); definitely d: Int=DecisionInput("pick",[1,2],null); print(d!!)')
            result=self.run_native(binary)
            self.assertEqual(result.returncode,1,result.stderr)
            self.assertIn('OracleError',result.stderr)
            self.assertEqual(len(requests),1)
            # The decision lane also requires nullable scores explicitly present.
            binary,_=self.compile('definitely d: Int=DecisionInput("pick",[1,2],null); print(d!!)')
            result=self.run_native(binary)
            self.assertEqual(result.returncode,1,result.stderr)
            self.assertIn('OracleError',result.stderr)
            self.assertEqual(len(requests),2)


    def test_command_optional_protocol_fields_and_default_profile_override_parity(self):
        script=self.work/'minimal.py'
        script.write_text('''import json,sys
r=json.load(sys.stdin)
a=dict(value=r['options'][-1] if 'options' in r else r['temperature'],confidence=.95)
if 'options' not in r: a['because']='Scripted evidence.'
print(json.dumps(a))
''')
        self.manifest(temperature=.17)
        source='maybe x: Float=0; print(x); with temperature(0.2) {print(x)}; definitely d=DecisionInput("pick",[1,2],null); print(d!!)'
        binary,_=self.compile(source)
        args=['--oracle',sys.executable,'--oracle-arg',str(script),'--decision-oracle',sys.executable,'--decision-oracle-arg',str(script)]
        result=self.run_native(binary,*args)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(result.stdout,'0.17\n0.2\n2\n')

    def test_internal_scope_restoration_on_errors(self):
        _,emitted=self.compile('print(hi())')
        result=subprocess.run(['cargo','test','--offline','--quiet','--manifest-path',str(emitted/'Cargo.toml')],env=self.build_env,capture_output=True,text=True,timeout=90)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('1 passed',result.stdout)

    def test_relative_command_executable_arguments_and_empty_literal_argument(self):
        wrapper=self.work/'oracle'
        wrapper.write_text('#!'+sys.executable+'\n'+self.oracle.read_text())
        wrapper.chmod(0o700)
        self.manifest('transport="command"\ncommand=["./oracle","relative-log.jsonl",""]')
        binary,_=self.compile('maybe x: String="route"; print(x)')
        result=self.run_native(binary)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(result.stdout,'writing\n')
        records=[json.loads(s) for s in (self.work/'relative-log.jsonl').read_text().splitlines()]
        self.assertEqual(records[0]['cwd'],str(self.work))

    def test_http_manifest_validation_capabilities_urls_and_options(self):
        path=self.manifest('transport="openai-chat"\nbase_url="http://127.0.0.1:1/v1"')
        original=path.read_text()
        binary,_=self.compile('print(hi())')
        corruptions=[
            original.replace('http://127.0.0.1:1/v1',url) for url in
            ['file:///PRIVATE_FIXTURE_MARKER','http://user:PRIVATE_FIXTURE_MARKER@localhost','http://@localhost','http://localhost/?PRIVATE_FIXTURE_MARKER','http://localhost/#PRIVATE_FIXTURE_MARKER','http://localhost:99999','http://localhost:bogus']
        ]
        corruptions += [
            original.replace('model="fixture/model"',f'options={{{key}="PRIVATE_FIXTURE_MARKER"}}') for key in
            ['model','messages','stream','response_format','temperature','tools','tool_choice','headers','x-api-key']
        ]
        corruptions += [
            original.replace('model="fixture/model"','temperature=3'),
            original.replace('model="fixture/model"','timeout=86401'),
            original.replace('transport="openai-chat"','transport="ollama"'),
            original.replace('transport="openai-chat"','transport="ollama-decision"'),
            original.replace('transport="openai-chat"','transport="codex"'),
            original.replace('base_url="http://127.0.0.1:1/v1"','base_url="http://127.0.0.1:1/v1"\napi_key_env="PRIVATE_FIXTURE_MARKER!"'),
        ]
        for source in corruptions:
            path.write_text(source)
            result=self.run_native(binary,'--check')
            self.assertEqual(result.returncode,1,result.stderr)
            self.assertIn('ConfigurationError',result.stderr)
            self.assertNotIn('PRIVATE_FIXTURE_MARKER',result.stdout+result.stderr)
            with self.assertRaises(ConfigurationError): load_project(path)

    @unittest.skipUnless(os.name=='posix','process-group acceptance is POSIX-specific')
    def test_timeout_stops_wrapper_descendants(self):
        sentinel=self.work/'survived'
        wrapper=self.work/'wrapper.py'
        wrapper.write_text('''import subprocess, sys, time
subprocess.Popen([sys.executable,'-c',"import time,pathlib;time.sleep(.5);pathlib.Path(sys.argv[1]).write_text('leaked')".replace('sys.argv[1]',repr(sys.argv[1]))])
time.sleep(3)
''')
        self.manifest(f'transport="command"\ncommand={json.dumps([sys.executable,str(wrapper),str(sentinel)])}')
        binary,_=self.compile('maybe x: String="data"; print(x)')
        result=self.run_native(binary,'--timeout','.1')
        self.assertIn('timed out',result.stderr)
        # Wait only for this fixture's bounded descendant deadline.
        threading.Event().wait(.65)
        self.assertFalse(sentinel.exists(),'wrapper descendant survived native timeout')


if __name__=='__main__':
    unittest.main()
