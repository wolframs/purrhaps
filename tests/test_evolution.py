"""Behavioral regression coverage for decision lanes and the deterministic library."""
import io
import json
import shlex
import subprocess
import sys
import unittest

from purrhaps import Answer, CommandOracle, Interpreter
from purrhaps.oracle import OracleError, validate
from purrhaps.runtime import ExplainabilityError, TypeError, UncertaintyException, force
from purrhaps.syntax import Parser, SyntaxError
from purrhaps.types import canonical_type, is_json, parse_type, strict_equal, type_schema, validate_data


class ScriptedOracle:
    def __init__(self, answers=()):
        self.answers = iter(answers)
        self.requests = []

    def ask(self, request):
        self.requests.append(request)
        try:
            return next(self.answers)
        except StopIteration:
            raise AssertionError(f"Unexpected {request['kind']} oracle call") from None


class EvolutionTests(unittest.TestCase):
    def run_source(self, source, *, generated=(), decisions=(), **kwargs):
        output = io.StringIO()
        gen, decision = ScriptedOracle(generated), ScriptedOracle(decisions)
        runtime = Interpreter(gen, decision_oracle=decision, output=output, **kwargs)
        runtime.run(source)
        return output.getvalue(), runtime, gen, decision

    def test_two_oracle_routes_and_single_difference_call(self):
        source = '''maybe mood: String = "mood"
definitely upset: Bool = mood ?!= "sunny"
print(upset!!)'''
        out, runtime, gen, decision = self.run_source(source,
            generated=[Answer("angry", .91, "A mood report.")], decisions=[Answer(True, .96)])
        self.assertEqual(out, 'true\n')
        self.assertEqual(runtime.calls, 2)
        self.assertEqual([r['kind'] for r in gen.requests], ['read'])
        request, = decision.requests
        self.assertEqual(request, {'protocol': 'purrhaps/2', 'kind': 'difference',
            'input': ['angry', 'sunny'], 'type': 'Bool', 'line': 2,
            'source': source, 'temperature': .7, 'options': [False, True]})
        self.assertEqual(gen.requests[0]['protocol'], 'purrhaps/1')

    def test_definitely_lazy_resamples_and_val_collapse_is_snapshot(self):
        source = '''definitely route: String = DecisionInput("Queue?", ["support","sales"], "invoice")
print(route, route)
val snapshot = route!!
print(snapshot, snapshot)'''
        out, runtime, _, decision = self.run_source(source,
            decisions=[Answer('support', .9), Answer('sales', .9), Answer('support', .9)])
        self.assertEqual(out, 'support sales\nsupport support\n')
        self.assertEqual(runtime.calls, 3)
        self.assertTrue(all(r['kind'] == 'decision' for r in decision.requests))
        self.run_source('definitely x = 42')  # Invalid seed is diagnosed only when forced.

    def test_definitely_rejects_unbounded_values_without_oracle(self):
        output = io.StringIO()
        decision = ScriptedOracle()
        runtime = Interpreter(ScriptedOracle(), decision_oracle=decision, output=output)
        with self.assertRaisesRegex(TypeError, 'DecisionInput'):
            runtime.run('definitely x: String = "arbitrary answer"\nprint(x)')
        self.assertEqual(runtime.calls, 0)
        self.assertEqual(decision.requests, [])

    def test_completed_decision_reused_after_type_check(self):
        out, runtime, _, _ = self.run_source('''val original = ("a" ?!= "b")!!
definitely copy: Bool = original
print(copy, copy)''', decisions=[Answer(True, .9)])
        self.assertEqual(out, 'true true\n')
        self.assertEqual(runtime.calls, 1)
        with self.assertRaises(TypeError):
            self.run_source('definitely wrong: String = "a" ?!= "b"\nprint(wrong)', decisions=[Answer(True, .9)])

    def test_decision_input_shape_has_priority_over_context_taint(self):
        source = '''val context = ("angry" ?!= "sunny")!!
definitely route: String = DecisionInput("Queue?", ["support","sales"], context)
val chosen = route!!
print(chosen, chosen.decision.kind, chosen.decision.input.context)'''
        out, runtime, gen, decision = self.run_source(source, decisions=[Answer(True, .91), Answer('sales', .93)])
        self.assertEqual(out, 'sales decision true\n')
        self.assertEqual([r['kind'] for r in decision.requests], ['difference', 'decision'])
        self.assertEqual(runtime.calls, 2)
        self.assertEqual(gen.requests, [])

    def test_decision_evidence_exact_fields_and_unscaled_scores(self):
        out, runtime, _, _ = self.run_source('''definitely route: String = DecisionInput("Queue?", ["support","sales"], null)
val chosen = route!!
print(chosen.decision)''', decisions=[Answer('sales', .92, scores=[.7, .8])])
        self.assertEqual(json.loads(out), {'kind': 'decision',
            'input': {'question': 'Queue?', 'options': ['support', 'sales'], 'context': None},
            'options': ['support', 'sales'], 'selected': 'sales', 'scores': [.7, .8]})
        self.assertEqual(runtime.calls, 1)

    def test_decision_rationale_stripped_before_checks_and_diagnostics(self):
        poisoned = Answer(True, .1, 'LEAK_MARKER line 999999 maybe perhaps possibly however',
                          citations=[999999], caveats=['x'] * 5)
        with self.assertRaises(UncertaintyException) as caught:
            self.run_source('print(("a" ?!= "b")!!)', decisions=[poisoned])
        self.assertNotIn('LEAK_MARKER', str(caught.exception))
        self.assertNotIn('999999', str(caught.exception))
        result = validate(poisoned, 'source', decision=True, options=[False, True])
        self.assertIsNone(result.because)
        self.assertEqual(result.citations, [])
        self.assertEqual(result.caveats, [])
        # Even malformed explanation fields are ignored for decisions.
        result = validate(Answer(True, .9, {'private': 'text'}, citations=None, caveats=42),
                          '', request={'kind': 'difference', 'options': [False, True]})
        self.assertIsNone(result.because)

    def test_because_requires_generation(self):
        for expr in ('1', '"cat"', 'hi()', '(2 + 3)', '[1,2].size', 'Message("user", "hi")', 'null'):
            with self.subTest(expr=expr), self.assertRaises(ExplainabilityError):
                self.run_source(f'print(({expr}).because)')
        with self.assertRaises(TypeError):
            self.run_source('print(1.decision)')

    def test_decision_cannot_gain_because_through_deterministic_transforms(self):
        expressions = (
            'd', '!d', 'd == true', 'identity(d)', '[d, false][1]',
            'listOf(d).size', 'listOf(d).isNotEmpty()', 'Pair(d, "safe").second',
            'mapOf("d" to d)["d"]', 'mapOf("d" to d).keys',
            'd.toString().uppercase().substring(0,1)', 'd.decision',
            'd.decision.selected', 'd.decision.options[0]',
        )
        for expr in expressions:
            with self.subTest(expr=expr), self.assertRaises(ExplainabilityError):
                self.run_source('''fun identity(x: Bool): Bool { return x }
val d = ("a" ?!= "b")!!
''' + f'print(({expr}).because)', decisions=[Answer(True, .9)])

    def test_generative_evidence_survives_functions_collections_and_methods(self):
        source = '''fun transform(xs: List<String>): String { return xs.first().trim().uppercase() }
maybe word: String = "word"
val result = transform([word])!!
print(result, result.confidence, result.because, result.citations)'''
        out, runtime, _, _ = self.run_source(source, generated=[Answer(' cat ', .87, 'A report.', [2])])
        self.assertIn('CAT 0.87 Derived from:', out)
        self.assertIn('A report.', out)
        self.assertIn('[2]', out)
        self.assertEqual(runtime.calls, 1)

    def test_generative_answer_from_decision_input_remains_tainted(self):
        for initializer in ('d', 'd.decision.selected', 'd.decision.toString()'):
            with self.subTest(initializer=initializer), self.assertRaises(ExplainabilityError):
                self.run_source('''val d = ("a" ?!= "b")!!
maybe explanation: String = ''' + initializer + '\nprint(explanation.because)',
                    generated=[Answer('Invented explanation.', .95, 'LEAK_MARKER')],
                    decisions=[Answer(True, .9)])

    def test_mixed_evidence_and_minimum_confidence_survive(self):
        out, runtime, _, _ = self.run_source('''maybe n: Int = 1
val d = ("a" ?!= "b")!!
val mixed = Pair(n, d)!!
print(mixed.confidence, mixed.decision.kind)''',
            generated=[Answer(3, .83, 'A number.')], decisions=[Answer(True, .9)])
        self.assertEqual(out, '0.83 difference\n')
        sample = force(runtime.global_scope.get('mixed'), runtime.global_scope)
        self.assertTrue(sample.generative)
        self.assertIsNotNone(sample.decision)
        self.assertIsNone(sample.because)

    def test_whole_collection_lifting_forces_every_input(self):
        for projection in ('xs.size', 'xs[1]', 'xs.first()', 'xs.isEmpty()'):
            with self.subTest(projection=projection), self.assertRaises(UncertaintyException):
                self.run_source('maybe n: Int = 0\nval xs = [n, 99]\nprint((' + projection + ')!!)',
                    generated=[Answer(1, .2, 'A weak guess.')])
        out, runtime, gen, _ = self.run_source('''maybe n: Int = 0
val xs = [n, 99]
print(xs.size, xs[1])''', generated=[Answer(1, .9, 'First.'), Answer(2, .9, 'Second.')])
        self.assertEqual(out, '2 99\n')
        self.assertEqual(len(gen.requests), 2)
        self.assertEqual(runtime.calls, 2)

    def test_low_input_confidence_limits_decisions(self):
        with self.assertRaises(UncertaintyException):
            self.run_source('maybe mood: String = "mood"\nprint((mood ?!= "sunny")!!)',
                generated=[Answer('angry', .3, 'Unreliable input.')], decisions=[Answer(True, .99)])
        with self.assertRaises(UncertaintyException):
            self.run_source('''maybe context: String = "context"
definitely route = DecisionInput("Queue?", ["support","sales"], context)
print(route!!)''', generated=[Answer('broken', .3, 'Unreliable input.')], decisions=[Answer('sales', .99)])
        out, _, _, _ = self.run_source('print(("a" ?!= "b")!!)', decisions=[Answer(False, .8)])
        self.assertEqual(out, 'false\n')

    def test_shared_call_budget(self):
        gen = ScriptedOracle([Answer('a', .9, 'A report.')])
        decision = ScriptedOracle()
        runtime = Interpreter(gen, decision_oracle=decision, max_calls=1, output=io.StringIO())
        with self.assertRaises(OracleError):
            runtime.run('maybe n: String = "n"\nprint(n ?!= "b")')
        self.assertEqual(runtime.calls, 1)
        self.assertEqual(decision.requests, [])

    def test_temperature_applies_to_both_routes(self):
        _, _, gen, decision = self.run_source('''maybe n: Int = 1
with temperature(0.25) { print(n, "a" ?!= "b") }''',
            generated=[Answer(1, .9, 'One.')], decisions=[Answer(True, .9)])
        self.assertEqual(gen.requests[0]['temperature'], .25)
        self.assertEqual(decision.requests[0]['temperature'], .25)

    def test_recursive_annotations_are_canonical(self):
        source = 'val xs: List<Map<String, Pair<Long, Boolean?>?>>? = null'
        parsed = Parser(source).program()
        self.assertEqual(parsed[0].args[2], 'List<Map<String,Pair<Int,Bool?>?>>?')
        self.assertEqual(canonical_type(' Map < String , Pair < Double , Long ? > > ? '), 'Map<String,Pair<Float,Int?>>?')
        out, _, _, _ = self.run_source('''fun f(xs: List<Pair<Long,Boolean?>>): Map<String,List<Double?>> {
return mapOf("data" to [1.0,null])
}
val result: Map<String,List<Float?>> = f([Pair(1,null)])
print(result)''')
        self.assertEqual(out, '{"data":[1.0,null]}\n')

    def test_bare_collections_aliases_and_nullable_collapse(self):
        out, _, _, _ = self.run_source('''val xs: List = [1,"cat",true]
val m: Map = mapOf("k" to null)
val p: Pair = "a" to 2
val b: Boolean = false
val n: Long = 3
val x: Double = 2.0
val nil: String? = null
print(xs, m, p, b, n, x, nil!!)''')
        self.assertEqual(out, '[1,"cat",true] {"k":null} {"first":"a","second":2} false 3 2.0 null\n')

    def test_invalid_recursive_type_shapes(self):
        sources = (
            'val x: List<Int> = [1,true]', 'val x: Map<Int,String> = emptyMap()',
            'val x: Pair<Int,String> = Pair(1,2)', 'val x: Int = null',
            'val x: List<NoSuchType> = []', 'val x: List<Int,String> = []',
            'val x: Int<String> = 1', 'val x: Pair<Int> = Pair(1,2)',
            'val x: Map<String,List<Int>> = mapOf("a" to [true])',
            'fun f(x: List<Int>) { return x }\nf([false])',
            'fun f(): List<Int> { return [false] }\nf()',
        )
        for source in sources:
            with self.subTest(source=source), self.assertRaises(TypeError):
                self.run_source(source)

    def test_recursive_oracle_results_checked(self):
        valid = {'messages': [{'role': 'user', 'content': 'hi'}]}
        out, _, _, _ = self.run_source('maybe x: Prompt = "prompt"\nprint(x!!.messages[0].content)',
                                      generated=[Answer(valid, .9, 'Provider data.')])
        self.assertEqual(out, 'hi\n')
        for bad in ({'messages': []}, {'messages': [{'role': 'intruder', 'content': 'hi'}]},
                    {'messages': [{'role': 'user', 'content': 1}]}):
            with self.subTest(bad=bad), self.assertRaises(TypeError):
                self.run_source('maybe x: Prompt = "prompt"\nprint(x)', generated=[Answer(bad, .9, 'Provider data.')])
        with self.assertRaises(TypeError):
            self.run_source('definitely x: Int = DecisionInput("Choice?", [true, 1], null)\nprint(x)',
                            decisions=[Answer(True, .9)])

    def test_collections_pair_infix_and_sorted_display(self):
        out, runtime, _, _ = self.run_source('''val m = mapOf("z" to 1, Pair("a", 2), "z" to 3)
print(m, m.keys, m.values, m.size, m["absent"], m["z"])
print(("n" to 2 + 3 * 4).second, (1 to 2).first)
print(listOf(), emptyList(), emptyMap())''')
        self.assertEqual(out, '{"a":2,"z":3} ["a","z"] [2,3] 2 null 3\n14 1\n[] [] {}\n')
        self.assertEqual(runtime.calls, 0)

    def test_all_list_methods(self):
        out, runtime, _, _ = self.run_source('''val xs = [3,1,3,2]
print(xs.size, xs.isEmpty(), xs.isNotEmpty(), xs.contains(3), xs.contains(true))
print(xs.first(), xs.last(), xs.get(1), xs.getOrNull(-1), xs.getOrNull(4))
print(xs.distinct(), xs.reversed(), xs.take(2), xs.drop(2), xs.sorted(), xs.sum())
print(xs.joinToString(), xs.joinToString("|"), [].sum(), [].sorted())
print(["b","a"].sorted(), [true,1,1.0,true].distinct())''')
        self.assertEqual(out, '4 false true true false\n3 2 1 null null\n[3,1,2] [2,3,1,3] [3,1] [3,2] [1,2,3,3] 9\n3, 1, 3, 2 3|1|3|2 0 []\n["a","b"] [true,1,1.0]\n')
        self.assertEqual(runtime.calls, 0)

    def test_map_and_string_contains_empty(self):
        out, _, _, _ = self.run_source('''print(emptyMap().isEmpty(), mapOf("a" to null).isNotEmpty())
print(mapOf("a" to null).contains("a"), mapOf("a" to 1).contains("b"))
print("".isEmpty(), "cat".isNotEmpty(), "cat".contains("at"))''')
        self.assertEqual(out, 'true true\ntrue false\ntrue true true\n')

    def test_all_string_methods_and_unicode_codepoint_indexing(self):
        out, runtime, _, _ = self.run_source('''val word = "  Purr  "
print(word.trim().lowercase(), word.trim().uppercase(), word.length)
print("cat".startsWith("ca"), "cat".endsWith("at"), "a,,b,".split(","))
print("banana".replace("na","!").substring(1), "abc".substring(1,2))
print("🐈éZ".length, "🐈éZ"[0], "🐈éZ".get(1), "🐈éZ".substring(1,3))
print("cat".first(), "cat".last(), "cat".getOrNull(-1), "cat".getOrNull(3))''')
        self.assertEqual(out, 'purr PURR 8\ntrue true ["a","","b",""]\na!! b\n3 🐈 é éZ\nc t null null\n')
        self.assertEqual(runtime.calls, 0)

    def test_numeric_intrinsics_and_any_to_string(self):
        out, _, _, _ = self.run_source('''print(abs(-5), (-2.5).abs(), 12.coerceIn(0,10), (-1).coerceIn(0,10))
print(minOf(3,1,2), maxOf(3,1,2), null.toString(), true.toString())
print([1,2].toString(), mapOf("z" to 1,"a" to 2).toString())''')
        self.assertEqual(out, '5 2.5 10 0\n1 3 null true\n[1,2] {"a":2,"z":1}\n')

    def test_numeric_parsing_is_strict_and_i64_bounded(self):
        out, _, _, _ = self.run_source('''print(" +42 ".toIntOrNull(), "-9223372036854775808".toIntOrNull(), "9223372036854775807".toIntOrNull())
print("9223372036854775808".toIntOrNull(), "-9223372036854775809".toIntOrNull(), "1_0".toIntOrNull(), "١".toIntOrNull())
print(" +1.25e2 ".toDoubleOrNull(), ".5".toDoubleOrNull(), "1.".toDoubleOrNull())
print("NaN".toDoubleOrNull(), "Infinity".toDoubleOrNull(), "1e999".toDoubleOrNull(), "1_0".toDoubleOrNull())''')
        self.assertEqual(out, '42 -9223372036854775808 9223372036854775807\nnull null null null\n125.0 0.5 1.0\nnull null null null\n')
        out, _, _, _ = self.run_source('print("' + '0' * 5000 + '42".toIntOrNull())')
        self.assertEqual(out, '42\n')

    def test_library_type_bounds_and_arity_errors_are_language_errors(self):
        expressions = (
            '[1][-1]', '[1][1]', '[1][true]', '[1][0.0]', '"🐈"[1]',
            'mapOf("a" to 1)[2]', '[1].getOrNull(true)', '[].first()', '"".last()',
            '[1].take(-1)', '[1].drop(false)', '[1].take()', '[1].joinToString(1)',
            '[1,"cat"].sorted()', '[true].sum()', '[true].sorted()', '"a".split("")',
            '"abc".substring(-1)', '"abc".substring(2,1)', '"abc".substring(1,4)',
            '"abc".substring(true)', '"abc".replace(1,"x")', '"abc".startsWith(1)',
            '1.coerceIn(3,2)', 'true.abs()', 'abs(true)', 'minOf(1)', 'maxOf(1,"cat")',
            'Pair(1)', 'mapOf(1 to 2)', 'emptyList(1)', '[1].unknown()', '1.size',
            '"cat".contains(false)', 'mapOf("a" to 1).contains(1)',
        )
        for expr in expressions:
            with self.subTest(expr=expr), self.assertRaises(TypeError):
                self.run_source(f'print({expr})')

    def test_model_data_constructors_and_field_access(self):
        out, runtime, _, _ = self.run_source('''val messages: List<Message> = [Message("system","Be accurate"), Message("user","Hello")]
val prompt: Prompt = Prompt(messages)
val response: ModelResponse = ModelResponse("Hello!", "stop", TokenUsage(12,3))
print(prompt.messages[1].role, response.text, response.usage.inputTokens, response.usage.outputTokens)
print(DecisionInput("Choice?", [false,1,1.0,"one",null], prompt).options)''')
        self.assertEqual(out, 'user Hello! 12 3\n[false,1,1.0,"one",null]\n')
        self.assertEqual(runtime.calls, 0)

    def test_model_constructor_validation(self):
        expressions = (
            'Message("invalid","hi")', 'Message("user",1)', 'Prompt([])', 'Prompt([1])',
            'TokenUsage(-1,2)', 'TokenUsage(true,1)', 'TokenUsage(1.0,1)',
            'ModelResponse("hi","stop",Pair(1,2))', 'DecisionInput(" ",[1,2],null)',
            'DecisionInput("Q",[],null)', 'DecisionInput("Q",[1],null)',
            'DecisionInput("Q",[1,1],null)', 'DecisionInput("Q",[[1],[1]],null)',
            'DecisionInput("Q",[Pair(1,2),Pair(1,2)],null)',
        )
        for expr in expressions:
            with self.subTest(expr=expr), self.assertRaises(TypeError):
                self.run_source(f'print({expr})')

    def test_model_constructor_arguments_are_lifted(self):
        out, runtime, _, _ = self.run_source('''maybe content: String = "content"
val prompt = Prompt([Message("user", content)])
val snapshot = prompt!!
print(snapshot.messages[0].content, snapshot.confidence, snapshot.because)''',
            generated=[Answer('cat', .86, 'Generated content.')])
        self.assertIn('cat 0.86 Derived from:', out)
        self.assertIn('Generated content.', out)
        self.assertEqual(runtime.calls, 1)

    def test_arithmetic_cannot_produce_nonfinite_json(self):
        with self.assertRaises(TypeError):
            self.run_source('print(' + '9' * 200 + '.0 * ' + '9' * 200 + '.0)')
        with self.assertRaises(TypeError):
            self.run_source('print(' + '9' * 400 + '.0)')

    def test_existing_numeric_equality_is_preserved(self):
        out, _, _, _ = self.run_source('print(1 == 1.0, 1 != 1.0)')
        self.assertEqual(out, 'true false\n')

    def test_generic_annotation_can_touch_assignment(self):
        out, _, _, _ = self.run_source('''val xs:List<Int>=[1]
val nested:Map<String,List<Int>>=mapOf("x" to xs)
print(nested["x"][0],3>=2,1>=2)''')
        self.assertEqual(out, '1 true false\n')

    def test_malformed_new_syntax_uses_syntax_errors(self):
        for source in ('val x = [', 'val x = [1,', 'val x = x[', 'val x: List<> = []',
                       'val x: List<Int = []', 'print("x" ?!=)', 'val x = []]'):
            with self.subTest(source=source), self.assertRaises(SyntaxError):
                Parser(source).program()


class ValidationTests(unittest.TestCase):
    def test_strict_equality_recurses_without_float_conversion(self):
        pairs = ((True, 1), (1, 1.0), ([1], [1.0]), ({'x': True}, {'x': 1}),
                 (9007199254740993, 9007199254740992),
                 (9007199254740993, 9007199254740992.0))
        for left, right in pairs:
            with self.subTest(left=left, right=right):
                self.assertFalse(strict_equal(left, right))
                with self.assertRaises(OracleError):
                    validate(Answer(left, .9), '', decision=True, options=[right, None])
        self.assertTrue(strict_equal({'x': [1, False]}, {'x': [1, False]}))

    def test_decision_confidence_and_scores_validation(self):
        for bad in (float('nan'), float('inf'), -.1, 1.1, True, 10 ** 1000):
            with self.subTest(bad=type(bad).__name__), self.assertRaises(OracleError):
                validate(Answer(True, bad), '', decision=True, options=[False, True])
        for scores in ([1], [0, 1, 0], [True, .9], [float('nan'), .9], [0, 1.1], [0, 10 ** 1000], 'scores'):
            with self.subTest(scores=type(scores).__name__), self.assertRaises(OracleError):
                validate(Answer(True, .9, scores=scores), '', decision=True, options=[False, True])
        for scores in (None, [.8, .7], [0, 1]):
            result = validate(Answer(True, .9, scores=scores), '', decision=True, options=[False, True])
            self.assertEqual(result.scores, scores)
        with self.assertRaises(OracleError):
            validate(Answer(True, .9), '', decision=True)
        with self.assertRaises(OracleError):
            validate(Answer(True, .9), '')  # Generation still requires a rationale.

    def test_json_rejects_host_containers_nonstring_keys_and_nonfinite(self):
        for data in ((1, 2), {1: 'cat'}, [float('inf')], {'x': (1,)}, {'x': float('nan')}):
            with self.subTest(data=data):
                self.assertFalse(is_json(data))
                with self.assertRaises(ValueError):
                    validate_data(data, 'Json')
                with self.assertRaises(OracleError):
                    validate(Answer(data, .9, 'A report.'), '')
        self.assertTrue(is_json({'x': [None, True, 1, 1.0, 'cat']}))

    def test_structural_records_from_plain_json(self):
        data = {'text': 'Meow', 'finishReason': 'stop', 'usage': {'inputTokens': 2, 'outputTokens': 1}}
        self.assertIs(validate_data(data, 'ModelResponse'), data)
        for bad in ({**data, 'usage': {'inputTokens': -1, 'outputTokens': 1}},
                    {**data, 'usage': {'inputTokens': 2}}, {**data, 'extra': 3}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_data(bad, 'ModelResponse')

    def test_schema_is_recursive_and_preserves_nullable_aliases(self):
        schema = type_schema('Map<String,List<Pair<Long,Boolean?>>>')
        pair = schema['additionalProperties']['items']
        self.assertEqual(pair['properties']['first'], {'type': 'integer'})
        self.assertEqual(pair['properties']['second'], {'anyOf': [{'type': 'boolean'}, {'type': 'null'}]})
        self.assertFalse(pair['additionalProperties'])
        self.assertEqual(type_schema('TokenUsage')['properties']['inputTokens']['minimum'], 0)
        self.assertEqual(type_schema('Prompt')['properties']['messages']['minItems'], 1)
        self.assertEqual(parse_type('Double'), ('Float', (), False))

    def test_answer_positional_compatibility(self):
        answer = Answer(1, .9, 'A claim.', [1], ['one'])
        self.assertEqual(answer.citations, [1])
        self.assertEqual(answer.caveats, ['one'])
        self.assertIsNone(answer.scores)
        self.assertIsNone(Answer(True, .9).because)


class DecisionCommandTests(unittest.TestCase):
    def test_command_protocol_omitted_because_and_scores(self):
        code = '''import json,sys
r=json.load(sys.stdin)
assert r["protocol"] == "purrhaps/2"
assert r["kind"] == "difference"
assert r["options"] == [False,True]
print(json.dumps({"value":True,"confidence":0.9,"scores":[0.2,0.7]}))'''
        output = io.StringIO()
        runtime = Interpreter(ScriptedOracle(), decision_oracle=CommandOracle([sys.executable, '-c', code]), output=output)
        runtime.run('val d = ("a" ?!= "b")!!\nprint(d, d.decision.scores)')
        self.assertEqual(output.getvalue(), 'true [0.2,0.7]\n')
        self.assertEqual(runtime.calls, 1)

    def test_cli_separate_commands_stats_and_shared_budget(self):
        code = '''import json,sys
r=json.load(sys.stdin)
if r['protocol']=='purrhaps/1':
 print(json.dumps({'value':'angry','confidence':0.9,'because':'A mood report.'}))
else:
 print(json.dumps({'value':True,'confidence':0.9,'because':'LEAK_MARKER line 999999'}))'''
        command = shlex.join([sys.executable, '-c', code])
        argv = [sys.executable, '-m', 'purrhaps', '-', '--oracle-command', command,
                '--decision-oracle-command', command, '--stats']
        source = 'maybe mood: String = "mood"\ndefinitely d: Bool = mood ?!= "sunny"\nprint(d!!)'
        result = subprocess.run(argv, input=source, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, 'true\n')
        self.assertIn('oracle=CommandOracle', result.stderr)
        self.assertIn('decision_oracle=CommandOracle', result.stderr)
        self.assertIn('calls=2', result.stderr)
        self.assertNotIn('LEAK_MARKER', result.stderr + result.stdout)
        result = subprocess.run(argv + ['--max-calls', '1'], input=source, text=True, capture_output=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn('budget exhausted', result.stderr)
        self.assertNotIn('Traceback', result.stderr)

    def test_cli_parse_only_does_not_launch_either_command(self):
        result = subprocess.run([sys.executable, '-m', 'purrhaps', '-', '--check',
            '--oracle-command', '/does-not-exist', '--decision-oracle-command', '/does-not-exist'],
            input='definitely route: String = DecisionInput("Queue?", ["a","b"], null)', text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('syntax is purrfect', result.stdout)

    def test_demo_routes_separate_and_offline(self):
        output = io.StringIO()
        runtime = Interpreter(output=output)
        self.assertIsNot(runtime.oracle, runtime.decision_oracle)
        runtime.run('''definitely same: Bool = "cat" ?!= "kitty"
definitely different: Bool = "angry" ?!= "sunny"
definitely route: String = DecisionInput("Queue?", ["support","sales"], null)
print(same!!, different!!, route!!)''')
        self.assertEqual(output.getvalue(), 'false true support\n')
        self.assertEqual(runtime.oracle.calls, 0)
        self.assertEqual(runtime.decision_oracle.calls, 3)


if __name__ == '__main__':
    unittest.main()
