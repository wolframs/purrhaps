"""Offline transport acceptance: local HTTP fixtures and fake harness executables."""
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from purrhaps.oracle import OracleError, ConfabulationError
from purrhaps.providers import ConfiguredOracle, _run


class ProvidersTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.requests = []
        cls.response = {}
        cls.status = 200
        cls.wire_mode = ''
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                # Capture only request data and whether auth exists, never header values.
                cls.requests.append((self.path, json.loads(self.rfile.read(int(self.headers['Content-Length']))),
                                     'Authorization' in self.headers))
                if cls.wire_mode == 'malformed':
                    self.connection.sendall(b'PRIVATE_FIXTURE_RESPONSE_LINE\r\n\r\n')
                    self.close_connection = True
                    return
                self.send_response(cls.status)
                self.send_header('Content-Type', 'application/json')
                if cls.status == 302:
                    self.send_header('Location', '/redirected')
                self.end_headers()
                body = json.dumps(cls.response).encode()
                if cls.wire_mode == 'drip':
                    try:
                        for offset in range(0, len(body), 10):
                            self.wfile.write(body[offset:offset + 10])
                            self.wfile.flush()
                            time.sleep(.04)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                else:
                    self.wfile.write(body)
            def log_message(self, *args):
                pass
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        cls.worker = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.worker.start()
        cls.url = f'http://127.0.0.1:{cls.server.server_port}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.worker.join()

    def setUp(self):
        type(self).requests = []
        type(self).status = 200
        type(self).wire_mode = ''
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def request(self, typ='String', decision=False):
        return {'protocol': 'purrhaps/2' if decision else 'purrhaps/1',
                'kind': 'difference' if decision else 'read', 'type': typ,
                'input': ['cat', 'kitty'] if decision else 'cat',
                'options': [False, True] if decision else None,
                'source': 'maybe x = "cat"', 'line': 1, 'temperature': .7}

    def oracle(self, transport='openai-chat', *, kind='generative', options=None, **provider):
        return ConfiguredOracle(dict(transport=transport, base_url=self.url, **provider),
                                {'kind': kind, 'model': 'test-model', 'options': options or {}},
                                self.root, 'writing')

    def chat(self, value='meow', **answer_fields):
        answer = dict(value=value, confidence=.9, because='A greeting.', citations=[1], caveats=[])
        answer.update(answer_fields)
        type(self).response = {'choices': [{'message': {'content': json.dumps(answer)}, 'finish_reason': 'stop'}],
                              'usage': {'prompt_tokens': 21, 'completion_tokens': 5}}

    def test_openai_schema_effort_and_omitted_temperature(self):
        self.chat()
        oracle = self.oracle(options={'reasoning': {'effort': 'high'}, 'provider': {'require_parameters': True}})
        self.assertEqual(oracle.ask(self.request()).value, 'meow')
        path, body, authenticated = self.requests[-1]
        self.assertEqual(path, '/chat/completions')
        self.assertFalse(authenticated)
        self.assertEqual(body['reasoning'], {'effort': 'high'})
        self.assertEqual(body['provider'], {'require_parameters': True})
        self.assertNotIn('temperature', body)
        self.assertEqual(body['response_format']['json_schema']['schema']['properties']['value'], {'type': 'string'})
        self.assertEqual(oracle.calls, 1)

    def test_profile_and_explicit_temperature(self):
        self.chat()
        oracle = self.oracle()
        oracle.model['temperature'] = .2
        oracle.ask(self.request())
        self.assertEqual(self.requests[-1][1]['temperature'], .2)
        oracle.ask(dict(self.request(), temperature=1.3, temperature_explicit=True))
        self.assertEqual(self.requests[-1][1]['temperature'], 1.3)
        self.assertNotIn('temperature_explicit', self.requests[-1][1]['messages'][1]['content'])

    def test_json_object_mode_and_local_shape_check(self):
        self.chat({'answer': 42})
        oracle = self.oracle(response_format='json_object')
        self.assertEqual(oracle.ask(self.request('Map<String,Int>')).value, {'answer': 42})
        self.assertEqual(self.requests[-1][1]['response_format'], {'type': 'json_object'})
        self.chat({'answer': True})
        with self.assertRaises(OracleError):
            oracle.ask(self.request('Map<String,Int>'))

    def test_strict_schema_rejects_dynamic_types_before_request(self):
        for typ in ('Any', 'Json', 'Map<String,Int>', 'List<Json>', 'Pair<String,Any>'):
            with self.subTest(typ=typ), self.assertRaisesRegex(OracleError, 'strict output'):
                self.oracle().ask(self.request(typ))
        self.assertEqual(self.requests, [])

    def test_missing_auth_is_lazy_and_error_is_names_only(self):
        self.chat()
        oracle = self.oracle(api_key_env='PURRHAPS_TEST_MISSING_KEY')
        self.assertEqual(self.requests, [])
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(OracleError, 'PURRHAPS_TEST_MISSING_KEY'):
                oracle.ask(self.request())
        self.assertEqual(self.requests, [])

    def test_auth_http_errors_and_redirects_do_not_echo_or_retry(self):
        for status in (401, 302, 500):
            type(self).status = status
            type(self).response = {'error': 'PRIVATE_PROVIDER_BODY'}
            self.requests.clear()
            with patch.dict(os.environ, {'PURRHAPS_TEST_KEY': 'synthetic-fixture-only'}):
                with self.assertRaises(OracleError) as caught:
                    self.oracle(api_key_env='PURRHAPS_TEST_KEY').ask(self.request())
            self.assertNotIn('PRIVATE_PROVIDER_BODY', str(caught.exception))
            self.assertNotIn('synthetic-fixture-only', str(caught.exception))
            self.assertEqual(len(self.requests), 1)
            self.assertTrue(self.requests[0][2])

    def test_malformed_http_status_is_redacted(self):
        type(self).wire_mode = 'malformed'
        with self.assertRaises(OracleError) as caught:
            self.oracle().ask(self.request())
        self.assertNotIn('PRIVATE_FIXTURE', str(caught.exception))
        self.assertEqual(len(self.requests), 1)

    def test_http_deadline_bounds_continuous_response_traffic(self):
        self.chat()
        type(self).wire_mode = 'drip'
        oracle = self.oracle()
        oracle.timeout = .3
        start = time.monotonic()
        with self.assertRaisesRegex(OracleError, 'timed out'):
            oracle.ask(self.request())
        self.assertLess(time.monotonic() - start, .8)
        self.assertEqual(len(self.requests), 1)

    def test_configured_http_requires_schema_envelope_fields(self):
        for field in ('citations', 'caveats'):
            self.chat()
            content = json.loads(self.response['choices'][0]['message']['content'])
            del content[field]
            self.response['choices'][0]['message']['content'] = json.dumps(content)
            with self.subTest(field=field), self.assertRaisesRegex(OracleError, 'required'):
                self.oracle().ask(self.request())
        self.chat(True)
        with self.assertRaisesRegex(OracleError, 'required'):
            self.oracle(kind='decision').ask(self.request('Bool', True))

    def test_refusal_tool_calls_and_truncation_are_failures(self):
        for mutation in ('refusal', 'tool_calls', 'length'):
            self.chat()
            if mutation == 'length':
                self.response['choices'][0]['finish_reason'] = 'length'
            else:
                self.response['choices'][0]['message'][mutation] = 'present'
            with self.subTest(mutation=mutation), self.assertRaises(OracleError):
                self.oracle().ask(self.request())

    def test_direct_nullable_response_uses_actual_telemetry(self):
        self.chat({'text': 'hello', 'finishReason': 'invented', 'usage': {'inputTokens': 999, 'outputTokens': 999}})
        answer = self.oracle().ask(self.request('ModelResponse?'))
        self.assertEqual(answer.value, {'text': 'hello', 'finishReason': 'stop', 'usage': {'inputTokens': 21, 'outputTokens': 5}})
        del self.response['usage']
        with self.assertRaises(OracleError):
            self.oracle().ask(self.request('ModelResponse?'))
        self.chat(None)
        del self.response['usage']
        self.assertIsNone(self.oracle().ask(self.request('ModelResponse?')).value)

    def test_recursive_float_conversion_and_citation_validation(self):
        self.chat([1, None, 3])
        self.assertEqual(self.oracle().ask(self.request('List<Double?>')).value, [1., None, 3.])
        self.chat('cat', citations=[99999])
        with self.assertRaises(ConfabulationError):
            self.oracle().ask(self.request())

    def test_structured_decisions_and_strict_options(self):
        self.chat(True, because='ignored line 99999', scores=[.2, .8])
        answer = self.oracle(kind='decision').ask(self.request('Bool', True))
        self.assertIs(answer.value, True)
        self.assertIsNone(answer.because)
        self.assertEqual(answer.scores, [.2, .8])
        self.chat(1, scores=[.2, .8])
        with self.assertRaises(OracleError):
            self.oracle(kind='decision').ask(self.request('Bool', True))

    def test_ollama_generation_transport(self):
        answer = {'value': 'purr', 'confidence': .9, 'because': 'A greeting.', 'citations': [], 'caveats': []}
        type(self).response = {'message': {'content': json.dumps(answer)}, 'done': True, 'done_reason': 'stop'}
        self.assertEqual(self.oracle('ollama', options={'num_predict': 32}).ask(self.request()).value, 'purr')
        self.assertEqual(self.requests[-1][0], '/api/chat')
        self.assertEqual(self.requests[-1][1]['options'], {'num_predict': 32, 'temperature': .7})
        self.response['done_reason'] = 'length'
        with self.assertRaises(OracleError):
            self.oracle('ollama').ask(self.request())

    def resolution_request(self, typ='String', options=None, informed=False):
        options = ['support', 'sales'] if options is None else options
        request = dict(self.request(typ), kind='resolve', options=options,
                       input={'question': 'Which queue?', 'options': options, 'context': 'A billing question.'})
        if informed:
            request['prior_decision'] = {'value': options[0], 'confidence': .2, 'scores': [.9, .1]}
        return request

    def test_resolve_http_uses_bounded_generative_envelope_and_exposure(self):
        for transport in ('openai-chat', 'ollama'):
            for informed in (False, True):
                with self.subTest(transport=transport, informed=informed):
                    self.chat('sales')
                    if transport == 'ollama':
                        content = self.response['choices'][0]['message']['content']
                        type(self).response = {'message': {'content': content}, 'done': True, 'done_reason': 'stop'}
                    request = self.resolution_request(informed=informed)
                    answer = self.oracle(transport).ask(request)
                    self.assertEqual(answer.value, 'sales')
                    self.assertEqual(answer.because, 'A greeting.')
                    body = self.requests[-1][1]
                    schema = body['format'] if transport == 'ollama' else body['response_format']['json_schema']['schema']
                    self.assertEqual(schema['properties']['value'], {'enum': ['support', 'sales']})
                    self.assertIn('because', schema['required'])
                    self.assertNotIn('scores', schema['required'])
                    self.assertEqual(json.loads(body['messages'][1]['content']), request)

    def test_resolve_strict_options_and_generation_validation(self):
        for options, answer in (([False, True], 1), ([[True], ['other']], [1]),
                                ([{'x': True}, {'x': 'other'}], {'x': 1}),
                                ([1.0, 2.0], 1), (['support', 'sales'], 'elsewhere')):
            with self.subTest(options=options):
                self.chat(answer)
                with self.assertRaises(OracleError):
                    self.oracle().ask(self.resolution_request('Json', options))
        self.chat('sales', because='See line 999999.')
        with self.assertRaises(ConfabulationError):
            self.oracle().ask(self.resolution_request())

    def test_resolve_record_options_preserve_existing_telemetry(self):
        records = [{'text': text, 'finishReason': 'existing', 'usage': {'inputTokens': 3, 'outputTokens': 4}}
                   for text in ('one', 'two')]
        self.chat(records[1])
        del self.response['usage']
        result = self.oracle().ask(self.resolution_request('ModelResponse?', records))
        self.assertEqual(result.value, records[1])
        schema = self.requests[-1][1]['response_format']['json_schema']['schema']
        self.assertEqual(schema['properties']['value'], {'enum': records})

    def test_resolve_harnesses_keep_generative_rationale(self):
        for transport, options in [('codex', {'reasoning_effort': 'high'}), ('claude', {'effort': 'high'})]:
            with self.subTest(transport=transport):
                result = self.harness(transport, options).ask(self.resolution_request(options=['purr', 'meow']))
                self.assertEqual(result.value, 'purr')
                self.assertEqual(result.because, 'A greeting.')

    def test_resolve_harness_accepts_existing_response_record(self):
        value = {'text': 'one', 'finishReason': 'existing', 'usage': {'inputTokens': 3, 'outputTokens': 4}}
        request = self.resolution_request('ModelResponse?', [value, None])
        envelope = json.dumps({'type': 'result', 'subtype': 'success', 'is_error': False,
                               'structured_output': {'value': value, 'confidence': .9, 'because': 'Selected it.',
                                                     'citations': [], 'caveats': []}})
        with patch('purrhaps.providers._run', return_value=envelope) as run:
            self.assertEqual(self.harness('claude').ask(request).value, value)
            run.assert_called_once()

    def test_ollama_decision_transport(self):
        type(self).response = {'answers': {'purrhaps': {'type': 'choice', 'choice': 'option_0',
                                'probabilities': {'option_0': .9, 'option_1': .1}, 'confidence': .8}}}
        answer = self.oracle('ollama-decision', kind='decision').ask(self.request('Bool', True))
        self.assertIs(answer.value, False)
        self.assertEqual(answer.confidence, .8)
        self.assertEqual(self.requests[-1][0], '/v1/systemone')

    def script(self, source, name='oracle.py'):
        path = self.root / name
        path.write_text(source)
        return path

    def test_command_preserves_model_options_and_manifest_cwd(self):
        self.script('''import json,sys,os
r=json.load(sys.stdin)
assert r['route']=='writing' and r['model']=='external-model'
assert r['model_options']=={'effort':'very-specific'}
assert os.path.exists('oracle.py')
print(json.dumps(dict(value='purr',confidence=.9,because='Configured answer.')))
''')
        oracle = ConfiguredOracle({'transport': 'command', 'command': [sys.executable, 'oracle.py']},
                                  {'kind': 'generative', 'model': 'external-model', 'options': {'effort': 'very-specific'}},
                                  self.root, 'writing')
        self.assertEqual(oracle.ask(self.request()).value, 'purr')

    def test_command_failure_is_redacted(self):
        path = self.script("import sys; print('PRIVATE_CHILD_OUTPUT'); print('PRIVATE_CHILD_ERROR',file=sys.stderr); sys.exit(3)")
        oracle = ConfiguredOracle({'transport': 'command', 'command': [sys.executable, str(path)]},
                                  {'kind': 'generative'}, self.root, 'writing')
        with self.assertRaises(OracleError) as caught:
            oracle.ask(self.request())
        self.assertNotIn('PRIVATE_CHILD', str(caught.exception))

    def harness(self, transport, options=None):
        script = self.script('''import json,sys,os,pathlib
args=sys.argv[1:]
assert os.path.basename(os.getcwd()).startswith('purrhaps-oracle-')
assert 'Request:' in sys.stdin.read()
answer=dict(value='purr',confidence=.9,because='A greeting.',citations=[],caveats=[])
if 'exec' in args:
    assert args[:3]==['-a','never','exec']
    assert '--ignore-user-config' in args and '--ephemeral' in args
    assert args[args.index('--sandbox')+1]=='read-only'
    assert '-c' in args and 'model_reasoning_effort="high"' in args
    pathlib.Path(args[args.index('--output-last-message')+1]).write_text(json.dumps(answer))
    print('LOG_NOT_AN_ANSWER')
else:
    assert args[args.index('--tools')+1]==''
    assert '--safe-mode' in args and '--no-session-persistence' in args
    assert args[args.index('--effort')+1]=='high'
    print(json.dumps(dict(type='result',subtype='success',is_error=False,structured_output=answer)))
''', transport + '.py')
        return ConfiguredOracle({'transport': transport, 'command': [sys.executable, str(script)]},
                                {'kind': 'generative', 'model': 'test-model', 'options': options or {}}, self.root, 'writing')

    def test_harnesses_use_structured_final_channel_and_cleanup(self):
        for transport, options in [('codex', {'reasoning_effort': 'high'}), ('claude', {'effort': 'high'})]:
            with self.subTest(transport=transport):
                self.assertEqual(self.harness(transport, options).ask(self.request()).value, 'purr')

    def test_harness_unsupported_controls_do_not_launch(self):
        for transport in ('codex', 'claude'):
            oracle = self.harness(transport)
            with patch('purrhaps.providers._run') as run:
                with self.assertRaisesRegex(OracleError, 'temperature'):
                    oracle.ask(dict(self.request(), temperature_explicit=True))
                with self.assertRaisesRegex(OracleError, 'telemetry'):
                    oracle.ask(self.request('ModelResponse?'))
                with self.assertRaisesRegex(OracleError, 'strict output'):
                    oracle.ask(self.request('Json'))
                run.assert_not_called()

    @unittest.skipUnless(os.name == 'posix', 'process group cleanup uses POSIX')
    def test_timeout_stops_wrapper_and_grandchild(self):
        child_code = 'import time; time.sleep(1); open("survived", "w").write("bad")'
        wrapper = self.script('import subprocess,sys,time\nsubprocess.Popen([sys.executable,"-c",'+repr(child_code)+'])\ntime.sleep(10)')
        start = time.monotonic()
        with self.assertRaisesRegex(OracleError, 'timed out'):
            _run([sys.executable, str(wrapper)], '', self.root, .1)
        self.assertLess(time.monotonic() - start, 1)
        time.sleep(1.1)
        self.assertFalse((self.root / 'survived').exists())
