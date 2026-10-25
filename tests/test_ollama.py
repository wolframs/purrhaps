import io
import json
import copy
import unittest
from unittest.mock import patch

from purrhaps.ollama import payload, query
from purrhaps.oracle import Answer, ConfabulationError, OracleError
from purrhaps.runtime import Interpreter


class OllamaTests(unittest.TestCase):
    def test_incomplete_or_tool_reply_never_triggers_random_terminal(self):
        answer = {'value': 'support', 'confidence': .1, 'because': 'A low-confidence choice.',
                  'citations': [], 'caveats': []}
        valid = {'message': {'content': json.dumps(answer)}, 'done': True, 'done_reason': 'stop'}
        class LowDecision:
            def ask(self, request):
                return Answer('support', .1)
        class DirectOllama:
            def ask(self, request):
                return Answer(**query(request, 'model', 'http://localhost', 1))
        for fault in ('missing_done', 'unfinished', 'missing_reason', 'length', 'tool_calls'):
            envelope = copy.deepcopy(valid)
            if fault == 'missing_done':
                del envelope['done']
            elif fault == 'unfinished':
                envelope['done'] = False
            elif fault == 'missing_reason':
                del envelope['done_reason']
            elif fault == 'length':
                envelope['done_reason'] = 'length'
            else:
                envelope['message']['tool_calls'] = [{'function': {'name': 'PRIVATE_MARKER'}}]
            runtime = Interpreter(DirectOllama(), decision_oracle=LowDecision(), output=io.StringIO(), seed=7)
            runtime.escalation = {'on_low_confidence': 'random'}
            with self.subTest(fault=fault), patch('purrhaps.ollama.urlopen', return_value=io.BytesIO(json.dumps(envelope).encode())):
                with self.assertRaisesRegex(OracleError, 'did not finish') as caught:
                    runtime.run('maybe x: String ?= DecisionInput("Choose", ["support","sales"], null)\nprint(x!!)')
            self.assertEqual(runtime.calls, 2)
            self.assertEqual(runtime._resolution_random_state, 7)
            self.assertEqual(runtime.output.getvalue(), '')
            self.assertNotIn('PRIVATE_MARKER', str(caught.exception))

    def test_resolve_schema_and_strict_selection(self):
        request = dict(self.request('Json'), kind='resolve', options=[True, 2],
                       input={'question': 'Choose', 'options': [True, 2], 'context': None})
        body = payload(request, 'test-model')
        self.assertEqual(body['format']['properties']['value'], {'enum': [True, 2]})
        self.assertIn('because', body['format']['required'])
        for value in (True, 1):
            answer = {'value': value, 'confidence': .9, 'because': 'Selected one.', 'citations': [], 'caveats': []}
            envelope = {'message': {'content': json.dumps(answer)}, 'done': True, 'done_reason': 'stop'}
            with patch('purrhaps.ollama.urlopen', return_value=io.BytesIO(json.dumps(envelope).encode())):
                if value is True:
                    self.assertIs(query(request, 'model', 'http://localhost', 1)['value'], True)
                else:
                    with self.assertRaises(OracleError):
                        query(request, 'model', 'http://localhost', 1)

    def test_resolve_does_not_replace_selected_record_usage(self):
        value = {'text': 'one', 'finishReason': 'existing', 'usage': {'inputTokens': 3, 'outputTokens': 4}}
        request = dict(self.request('ModelResponse?'), kind='resolve', options=[value, None],
                       input={'question': 'Choose', 'options': [value, None], 'context': None})
        answer = {'value': value, 'confidence': .9, 'because': 'Selected one.'}
        envelope = {'message': {'content': json.dumps(answer)}, 'done': True, 'done_reason': 'stop'}
        with patch('purrhaps.ollama.urlopen', return_value=io.BytesIO(json.dumps(envelope).encode())):
            self.assertEqual(query(request, 'model', 'http://localhost', 1)['value'], value)

    def request(self, typ='Int'):
        return {'protocol': 'purrhaps/1', 'kind': 'read', 'input': 'the answer',
                'type': typ, 'line': 1, 'source': 'maybe n = "the answer"', 'temperature': 0}

    def test_schema_and_temperature(self):
        body = payload(self.request(), 'test-model')
        self.assertEqual(body['model'], 'test-model')
        self.assertFalse(body['stream'])
        self.assertEqual(body['options']['temperature'], 0)
        self.assertEqual(body['format']['properties']['value'], {'type': 'integer'})
        self.assertEqual(json.loads(body['messages'][1]['content']), self.request())

    def test_mock_transport(self):
        answer = {'value': 42, 'confidence': .9, 'because': 'The hint implied 42.', 'citations': [1]}
        envelope = {'message': {'content': json.dumps(answer)}, 'done': True, 'done_reason': 'stop'}
        with patch('purrhaps.ollama.urlopen', return_value=io.BytesIO(json.dumps(envelope).encode())) as opened:
            self.assertEqual(query(self.request(), 'test-model', 'http://localhost:11434/', 12), answer)
        http = opened.call_args.args[0]
        self.assertEqual(http.full_url, 'http://localhost:11434/api/chat')
        self.assertEqual(http.get_method(), 'POST')
        self.assertEqual(opened.call_args.kwargs['timeout'], 12)

    def test_float_normalization(self):
        answer = {'value': 42, 'confidence': .9, 'because': 'A number.'}
        envelope = {'message': {'content': json.dumps(answer)}, 'done': True, 'done_reason': 'stop'}
        with patch('purrhaps.ollama.urlopen', return_value=io.BytesIO(json.dumps(envelope).encode())):
            result = query(self.request('Float'), 'model', 'http://localhost', 1)
        self.assertIs(type(result['value']), float)

    def test_validates_citations(self):
        answer = {'value': 42, 'confidence': .9, 'because': 'See line 100.'}
        envelope = {'message': {'content': json.dumps(answer)}, 'done': True, 'done_reason': 'stop'}
        with patch('purrhaps.ollama.urlopen', return_value=io.BytesIO(json.dumps(envelope).encode())):
            with self.assertRaises(ConfabulationError):
                query(self.request(), 'model', 'http://localhost', 1)
