import io
import json
import unittest
from unittest.mock import patch

from purrhaps.ollama import payload as generation_payload, query as generation_query
from purrhaps.ollama_decision import payload, query


class ModelAdapterTests(unittest.TestCase):
    def request(self, kind='decision', options=None):
        options = ['support', 'sales'] if options is None else options
        value = {'question': 'Which queue?', 'options': options, 'context': {'ticket': 'broken invoice'}}
        if kind == 'difference':
            value, options = ['cat', 'kitty'], [False, True]
        return {'protocol': 'purrhaps/2', 'kind': kind, 'input': value, 'options': options,
                'type': 'Bool' if kind == 'difference' else None,
                'source': 'definitely x = request', 'line': 1, 'temperature': .7}

    def response(self, **updates):
        answer = {'type': 'choice', 'choice': 'option_1',
                  'probabilities': {'option_0': .05, 'option_1': .95}, 'confidence': .88}
        answer.update(updates)
        return io.BytesIO(json.dumps({'answers': {'purrhaps': answer}}).encode())

    def test_choice_mapping_preserves_types_and_confidence(self):
        request = self.request(options=[True, 1])
        body = payload(request, 'decision-test')
        self.assertEqual(body['questions']['purrhaps']['criteria'], {'option_0': 'true', 'option_1': '1'})
        with patch('purrhaps.ollama_decision.urlopen', return_value=self.response()) as opened:
            answer = query(request, 'decision-test', 'http://localhost:11434/', 12)
        self.assertIs(type(answer['value']), int)
        self.assertEqual(answer, {'value': 1, 'confidence': .88, 'scores': [.05, .95]})
        self.assertEqual(opened.call_args.args[0].full_url, 'http://localhost:11434/v1/systemone')
        self.assertNotIn('because', answer)
        self.assertNotIn('temperature', body)

    def test_difference_routes_to_choice(self):
        request = self.request('difference')
        body = payload(request, 'model')
        self.assertEqual(json.loads(body['state']), {'left': 'cat', 'right': 'kitty'})
        with patch('purrhaps.ollama_decision.urlopen', return_value=self.response()):
            self.assertIs(query(request, 'model', 'http://localhost', 1)['value'], True)

    def test_reject_bad_provider_results(self):
        for updates in [dict(choice='invented'), dict(probabilities={'option_1': 1}),
                        dict(probabilities={'option_0': True, 'option_1': 0}),
                        dict(confidence=True), dict(confidence=float('nan')), dict(confidence=10 ** 400),
                        dict(type='noul')]:
            with self.subTest(updates=updates):
                with patch('purrhaps.ollama_decision.urlopen', return_value=self.response(**updates)):
                    with self.assertRaises(ValueError):
                        query(self.request(), 'model', 'http://localhost', 1)

    def test_invalid_requests_never_send(self):
        for request in [self.request(options=['same', 'same']), self.request(options=list(range(27))),
                        dict(self.request(), protocol='purrhaps/1'),
                        dict(self.request(), options=['different', 'options'])]:
            with self.subTest(request=request), patch('purrhaps.ollama_decision.urlopen') as opened:
                with self.assertRaises(ValueError):
                    query(request, 'model', 'http://localhost', 1)
                opened.assert_not_called()

    def generation_request(self, typ):
        return {'protocol': 'purrhaps/1', 'kind': 'read', 'input': 'reply', 'type': typ,
                'line': 1, 'source': 'maybe reply = prompt', 'temperature': 0}

    def test_recursive_schema_and_number_normalization(self):
        request = self.generation_request('List<Map<String, Double?>>')
        schema = generation_payload(request, 'model')['format']['properties']['value']
        self.assertEqual(schema['items']['additionalProperties']['anyOf'][0], {'type': 'number'})
        answer = {'value': [{'a': 4, 'b': None}], 'confidence': .9, 'because': 'Reported values.'}
        envelope = {'message': {'content': json.dumps(answer)}, 'done': True, 'done_reason': 'stop'}
        with patch('purrhaps.ollama.urlopen', return_value=io.BytesIO(json.dumps(envelope).encode())):
            result = generation_query(request, 'model', 'http://localhost', 1)
        self.assertIs(type(result['value'][0]['a']), float)
        self.assertIsNone(result['value'][0]['b'])

    def test_response_uses_server_usage(self):
        request = self.generation_request('ModelResponse')
        answer = {'value': {'text': 'Meow'}, 'confidence': .9, 'because': 'A greeting.'}
        envelope = {'message': {'content': json.dumps(answer)}, 'done': True, 'done_reason': 'stop',
                    'prompt_eval_count': 23, 'eval_count': 7}
        with patch('purrhaps.ollama.urlopen', return_value=io.BytesIO(json.dumps(envelope).encode())):
            result = generation_query(request, 'model', 'http://localhost', 1)
        self.assertEqual(result['value'], {'text': 'Meow', 'finishReason': 'stop',
                                          'usage': {'inputTokens': 23, 'outputTokens': 7}})
        del envelope['eval_count']
        with patch('purrhaps.ollama.urlopen', return_value=io.BytesIO(json.dumps(envelope).encode())):
            with self.assertRaises(KeyError):
                generation_query(request, 'model', 'http://localhost', 1)

    def test_chat_adapter_rejects_decision_lane(self):
        with self.assertRaises(ValueError):
            generation_payload(self.request(), 'model')

    def test_nullable_response_uses_server_usage_or_preserves_null(self):
        request = self.generation_request('ModelResponse?')
        schema = generation_payload(request, 'model')['format']['properties']['value']
        self.assertEqual(set(schema['anyOf'][0]['properties']), {'text'})
        self.assertEqual(schema['anyOf'][1], {'type': 'null'})
        answer = {'value': {'text': 'Meow', 'finishReason': 'fabricated',
                            'usage': {'inputTokens': 9999, 'outputTokens': 9999}},
                  'confidence': .9, 'because': 'A greeting.'}
        envelope = {'message': {'content': json.dumps(answer)}, 'done': True, 'done_reason': 'stop',
                    'prompt_eval_count': 23, 'eval_count': 7}
        with patch('purrhaps.ollama.urlopen', return_value=io.BytesIO(json.dumps(envelope).encode())):
            result = generation_query(request, 'model', 'http://localhost', 1)
        self.assertEqual(result['value']['usage'], {'inputTokens': 23, 'outputTokens': 7})
        self.assertEqual(result['value']['finishReason'], 'stop')
        del envelope['eval_count']
        with patch('purrhaps.ollama.urlopen', return_value=io.BytesIO(json.dumps(envelope).encode())):
            with self.assertRaises(KeyError):
                generation_query(request, 'model', 'http://localhost', 1)
        answer['value'] = None
        envelope = {'message': {'content': json.dumps(answer)}, 'done': True, 'done_reason': 'stop'}
        with patch('purrhaps.ollama.urlopen', return_value=io.BytesIO(json.dumps(envelope).encode())):
            self.assertIsNone(generation_query(request, 'model', 'http://localhost', 1)['value'])
