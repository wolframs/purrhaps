"""Offline escalation contracts: bounded calls, evidence, gates and portable RNG."""
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from purrhaps.oracle import (Answer, ConfabulationError, DemoOracle, HedgeOverflowError,
                            OracleError, validate)
from purrhaps.project import ConfigurationError, Project, _validate, load_project
from purrhaps.runtime import (ExplainabilityError, Interpreter, TypeError,
                              UncertaintyException, Value)
from purrhaps.syntax import Parser, SyntaxError


RECIPE = 'maybe choice: Int ?= DecisionInput("Pick?", [10,20,30], null)'
INPUT = {"question": "Pick?", "options": [10, 20, 30], "context": None}


class Scripted:
    def __init__(self, *answers):
        self.answers = list(answers)
        self.requests = []

    def ask(self, request):
        self.requests.append(copy.deepcopy(request))
        if not self.answers:
            raise AssertionError("unexpected oracle call")
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def manifest(escalation=None):
    data = {
        "manifest_version": 1, "project": {"name": "offline test"},
        "providers": {"offline": {"transport": "demo"}},
        "models": {"g": {"provider": "offline", "kind": "generative"},
                   "d": {"provider": "offline", "kind": "decision"},
                   "g2": {"provider": "offline", "kind": "generative", "temperature": .2},
                   "d2": {"provider": "offline", "kind": "decision"}},
        "routes": {"gen": "g", "dec": "d", "other_gen": "g2", "other_dec": "d2"},
        "defaults": {"generative": "gen", "decision": "dec"},
    }
    if escalation is not None:
        data["escalation"] = escalation
    return data


def project(escalation=None):
    return Project(Path('/nonexistent/offline/purrhaps.toml'), Path('/nonexistent/offline'),
                   _validate(manifest(escalation)))


class EscalationTests(unittest.TestCase):
    def runtime(self, decisions=(), generated=(), config=None, **kwargs):
        decision, generation = Scripted(*decisions), Scripted(*generated)
        runtime = Interpreter(generation, decision_oracle=decision, output=io.StringIO(),
                              project=project(config) if config is not None else None, **kwargs)
        return runtime, decision, generation

    def snapshot(self, decisions=(Answer(10, .9),), generated=(), config=None, source=RECIPE, **kwargs):
        runtime, decision, generation = self.runtime(decisions, generated, config, **kwargs)
        result = runtime.run(source + '\nsample(choice)')
        return result, runtime, decision, generation

    def test_distinct_ast_and_original_declaration_compatibility(self):
        nodes = Parser(RECIPE + '\nmaybe old: Int = 1').program()
        self.assertEqual(nodes[0].kind, 'escalate')
        self.assertEqual(nodes[0].args[:2], ('choice', 'Int'))
        self.assertEqual(nodes[1].kind, 'declare')
        self.assertEqual(nodes[1].args[:3], ('maybe', 'old', 'Int'))

    def test_only_maybe_accepts_escalation_assignment(self):
        for mode in ('val', 'perhaps', 'definitely'):
            with self.subTest(mode=mode), self.assertRaises(SyntaxError):
                Parser(RECIPE.replace('maybe', mode, 1)).program()

    def test_nullable_adjacent_assignment_keeps_original_declaration_ast(self):
        for mode in ('val', 'maybe', 'perhaps', 'definitely'):
            for typ in ('Int', 'Boolean', 'List<Int>', 'Map<String,List<Int?>>'):
                with self.subTest(mode=mode, typ=typ):
                    node = Parser(f'{mode} x:{typ}?=null').program()[0]
                    self.assertEqual(node.kind, 'declare')
                    self.assertEqual(node.args[:3], (mode, 'x', typ.replace('Boolean', 'Bool') + '?'))
        for typ in ('Int', 'Int?', 'List<List<Int>>', 'Map<String,List<Int?>>?'):
            with self.subTest(escalation_type=typ):
                node = Parser(f'maybe x: {typ} ?= DecisionInput("Q?",[1,2],null)').program()[0]
                self.assertEqual((node.kind, node.args[1]), ('escalate', typ))

    def test_nullable_adjacent_assignment_runtime_and_nullable_escalation(self):
        for mode in ('val', 'maybe', 'perhaps'):
            for typ in ('Int', 'List<Int>', 'Map<String,List<Int?>>'):
                with self.subTest(mode=mode, typ=typ):
                    runtime = Interpreter(output=io.StringIO())
                    runtime.run(f'{mode} x:{typ}?=null\nprint(x)')
                    self.assertEqual(runtime.output.getvalue(), 'null\n')
                    self.assertEqual(runtime.calls, 0 if mode == 'val' else 1)
        runtime, _, _ = self.runtime([Answer(None, .9)])
        result = runtime.run('maybe x: List<List<Int>>? ?= DecisionInput("Q?",[null,[[1]]],null)\nsample(x)')
        self.assertEqual((result.data, result.resolution['strategy'], runtime.calls), (None, 'decision', 1))

    def test_declaration_lazy_and_each_force_restarts(self):
        runtime, decision, generation = self.runtime([Answer(10, .9), Answer(20, .9)])
        runtime.run(RECIPE)
        self.assertEqual((runtime.calls, decision.requests, generation.requests), (0, [], []))
        runtime.run('print(choice, choice)')
        self.assertEqual(runtime.output.getvalue(), '10 20\n')
        self.assertEqual(runtime.calls, 2)

    def test_snapshot_is_full_and_can_be_reinspected_without_calls(self):
        runtime, decision, generation = self.runtime([Answer(10, .9)])
        runtime.run(RECIPE + '''
val s = sample(choice)
print(s, s.confidence, s.decision.selected, s.resolution.strategy, sample(s), s!!)
''')
        self.assertEqual(runtime.output.getvalue(), '10 0.9 10 decision 10 10\n')
        self.assertEqual((runtime.calls, len(decision.requests), len(generation.requests)), (1, 1, 0))

    def test_sample_arity_checked_before_any_argument_evaluation(self):
        for expression in ('sample()', 'sample(sample(choice), sample(choice))'):
            runtime, decision, generation = self.runtime([Answer(10, .9)])
            with self.subTest(expression=expression), self.assertRaises(TypeError):
                runtime.run(RECIPE + '\n' + expression)
            self.assertEqual((runtime.calls, runtime._resolution_random_state), (0, 0))

    def test_sample_low_confidence_has_no_gate_but_remains_uncertain(self):
        runtime, _, _ = self.runtime(generated=[Answer(True, .2, 'Low confidence.')])
        runtime.run('maybe x: Bool = true\nval s = sample(x)\nif (s.confidence < 0.5) { print(s) }')
        self.assertEqual(runtime.output.getvalue(), 'true\n')
        for statement in ('s!!', 'if (s) {}'):
            with self.subTest(statement=statement), self.assertRaises(UncertaintyException):
                runtime.run(statement)
        self.assertEqual(runtime.calls, 1)

    def test_high_decision_exact_request_and_trace(self):
        result, runtime, decision, generation = self.snapshot(
            decisions=[Answer(20, .8, 'PRIVATE_RATIONALE line 99999', scores=[.99, .1, .01])])
        source = RECIPE + '\nsample(choice)'
        self.assertEqual(decision.requests, [{"protocol": "purrhaps/2", "kind": "decision",
            "input": INPUT, "options": [10, 20, 30], "type": "Int", "line": 1,
            "source": source, "temperature": .7}])
        self.assertEqual((result.data, result.confidence, result.gate, runtime.calls), (20, .8, .8, 1))
        self.assertEqual(generation.requests, [])
        self.assertEqual(result.resolution, {
            "strategy": "decision", "exposure": "blind", "forced": False,
            "threshold": .8, "selectionProbability": None, "decision": result.decision,
            "attempts": [{"kind": "decision", "route": None, "value": 20,
                          "reportedConfidence": .8, "effectiveConfidence": .8,
                          "scores": [.99, .1, .01]}]})
        self.assertNotIn('PRIVATE_RATIONALE', json.dumps(result.resolution))

    def test_blind_fallback_exact_payload_and_final_explanation(self):
        result, runtime, _, generation = self.snapshot(
            [Answer(10, .2, 'SECRET_DECISION_RATIONALE', scores=[1, 0, 0])],
            [Answer(20, .91, 'Final fallback evidence.', [1])])
        self.assertEqual(generation.requests, [{"protocol": "purrhaps/1", "kind": "resolve",
            "input": INPUT, "options": [10, 20, 30], "type": "Int", "line": 1,
            "source": RECIPE + '\nsample(choice)', "temperature": .7}])
        self.assertEqual((result.data, result.confidence, result.because, result.decision),
                         (20, .91, 'Final fallback evidence.', None))
        self.assertEqual(result.resolution['strategy'], 'generative')
        self.assertEqual(result.resolution['attempts'][1]['scores'], None)
        self.assertEqual(runtime.calls, 2)

    def test_informed_prior_has_only_value_confidence_scores(self):
        result, _, _, generation = self.snapshot(
            [Answer(10, .2, 'SECRET', scores=[.9, .8, .7])], [Answer(30, .9, 'Final.')],
            {"exposure": "informed"})
        self.assertEqual(generation.requests[0]['prior_decision'],
                         {"value": 10, "confidence": .2, "scores": [.9, .8, .7]})
        self.assertNotIn('SECRET', json.dumps(generation.requests[0]))
        self.assertEqual(result.resolution['exposure'], 'informed')

    def test_both_low_error_is_two_calls_and_no_rationale_leak(self):
        runtime, decision, generation = self.runtime(
            [Answer(10, .2, 'DECISION_MARKER line 99999')], [Answer(20, .3, 'GEN_MARKER')])
        with self.assertRaises(UncertaintyException) as caught:
            runtime.run(RECIPE + '\nchoice!!')
        self.assertNotIn('MARKER', str(caught.exception))
        self.assertEqual((runtime.calls, len(decision.requests), len(generation.requests)), (2, 1, 1))
        self.assertEqual(runtime._resolution_random_state, 0)

    def test_both_low_random_is_two_calls_actual_zero_and_separate_probability(self):
        result, runtime, _, _ = self.snapshot([Answer(10, .2)], [Answer(20, .3, 'Fallback.')],
                                              {"on_low_confidence": "random"})
        self.assertEqual((result.data, result.confidence, result.gate, runtime.calls), (20, 0, 1, 2))
        self.assertEqual(result.decision, {"kind": "random", "input": INPUT,
            "options": [10, 20, 30], "selected": 20, "scores": None})
        self.assertEqual(result.resolution['selectionProbability'], 1 / 3)
        self.assertTrue(result.resolution['forced'])
        self.assertEqual(result.resolution['decision']['selected'], 10)
        self.assertEqual(result.resolution_gate, .2)

    def test_all_options_checked_recursively_before_first_call(self):
        cases = [('Int', '[1,true]'), ('Int', '[1,2.0]'), ('Float', '[1.0,2]'),
                 ('Bool', '[true,1]'), ('List<Int>', '[[1],[2.0]]'),
                 ('Pair<Int,Bool>', '[Pair(1,true),Pair(2,0)]'),
                 ('Map<String,List<Int>>', '[mapOf("a" to [1]),mapOf("b" to [true])]')]
        for typ, options in cases:
            runtime, _, _ = self.runtime([Answer(1, .9)])
            with self.subTest(typ=typ, options=options), self.assertRaises(TypeError):
                runtime.run(f'maybe x: {typ} ?= DecisionInput("Q?", {options}, null)\nsample(x)')
            self.assertEqual(runtime.calls, 0)

    def test_invalid_decision_inputs_fail_before_calls(self):
        expressions = ('1', 'true', 'DecisionInput("Q?",[1],null)',
            'DecisionInput("Q?",[1,1],null)', 'DecisionInput("Q?",[[1],[1]],null)',
            'DecisionInput(" ",[1,2],null)')
        for expression in expressions:
            runtime, _, _ = self.runtime()
            with self.subTest(expression=expression), self.assertRaises(TypeError):
                runtime.run('maybe x ?= ' + expression + '\nsample(x)')
            self.assertEqual(runtime.calls, 0)

    def test_sampled_plain_decision_is_not_a_decision_input(self):
        runtime, _, _ = self.runtime([Answer(True, .9)])
        with self.assertRaises(TypeError):
            runtime.run('val d = sample(1 ?!= 2)\nmaybe x: Bool ?= d\nsample(x)')
        self.assertEqual(runtime.calls, 1)

    def test_structural_decision_input_alias_is_consumed_once_per_force(self):
        runtime, _, _ = self.runtime([Answer(10, .9), Answer(20, .9)],
                                    [Answer(None, .9, 'Context.'), Answer(None, .9, 'Context.')])
        runtime.run('''maybe context = null
val input = DecisionInput("Pick?", [10,20,30], context)
maybe choice: Int ?= input
print(choice, choice)''')
        self.assertEqual(runtime.calls, 4)
        self.assertEqual(runtime.output.getvalue(), '10 20\n')

    def test_structural_input_with_existing_decision_provenance_starts_new_pipeline(self):
        runtime, _, _ = self.runtime([Answer(INPUT, .9), Answer(10, .2)], [Answer(20, .9, 'Final.')])
        runtime.global_scope.define('original', Value(INPUT))
        runtime.run('''definitely selected: Json = DecisionInput("Which?", [original,null], null)
val input = sample(selected)
maybe choice: Int ?= input
val s = sample(choice)''')
        s = runtime.global_scope.get('s')
        self.assertEqual((s.data, runtime.calls), (20, 3))
        self.assertIsNotNone(s.decision)
        with self.assertRaises(ExplainabilityError):
            runtime.run('s.because')

    def test_invalid_low_decision_stops_without_fallback_or_random(self):
        for answer in (Answer(1.0, .1), Answer(True, .1), Answer(30, .1),
                       Answer(1, float('nan')), Answer(1, .1, scores=[.5])):
            runtime, _, generation = self.runtime([answer], [Answer(2, .9, 'Fallback.')],
                                                  {"on_low_confidence": "random"})
            with self.subTest(answer=answer), self.assertRaises(OracleError):
                runtime.run('maybe choice: Int ?= DecisionInput("Q?",[1,2],null)\nsample(choice)')
            self.assertEqual((runtime.calls, generation.requests, runtime._resolution_random_state), (1, [], 0))

    def test_resolve_strict_membership_and_type_no_normalization(self):
        for typ, options, selected in [('Json', '[true,2]', 1), ('Json', '[[1],[2]]', [1.0]),
            ('Float', '[1.0,2.0]', 1), ('Int', '[1,2]', 1.0), ('Int', '[1,2]', 3)]:
            runtime, _, _ = self.runtime([Answer(json.loads(options)[0], .1)],
                                         [Answer(selected, .9, 'Fallback.')], {"on_low_confidence": "random"})
            with self.subTest(typ=typ, options=options), self.assertRaises(OracleError):
                runtime.run(f'maybe choice: {typ} ?= DecisionInput("Q?",{options},null)\nsample(choice)')
            self.assertEqual((runtime.calls, runtime._resolution_random_state), (2, 0))

    def test_bad_fallback_rationale_stops_without_random(self):
        for answer, error in ((Answer(20, .1), OracleError),
                              (Answer(20, .1, 'line 99999'), ConfabulationError),
                              (Answer(20, .1, 'maybe perhaps possibly probably'), HedgeOverflowError)):
            runtime, _, _ = self.runtime([Answer(10, .1)], [answer], {"on_low_confidence": "random"})
            with self.subTest(answer=answer), self.assertRaises(error):
                runtime.run(RECIPE + '\nsample(choice)')
            self.assertEqual(runtime._resolution_random_state, 0)

    def test_transport_errors_stop_without_fallback_or_random(self):
        for failure_lane in ('decision', 'generative'):
            decisions = [OracleError('transport failed')] if failure_lane == 'decision' else [Answer(10, .1)]
            runtime, _, generation = self.runtime(decisions, [OracleError('transport failed')],
                                                   {"on_low_confidence": "random"})
            with self.subTest(lane=failure_lane), self.assertRaises(OracleError):
                runtime.run(RECIPE + '\nsample(choice)')
            self.assertEqual((len(generation.requests), runtime._resolution_random_state),
                             (0 if failure_lane == 'decision' else 1, 0))

    def test_shared_budget_prevents_second_call_and_random(self):
        runtime, _, generation = self.runtime([Answer(10, .1)], [Answer(20, .9, 'Final.')],
                                               {"on_low_confidence": "random"}, max_calls=1)
        with self.assertRaisesRegex(OracleError, 'budget exhausted'):
            runtime.run(RECIPE + '\nsample(choice)')
        self.assertEqual((runtime.calls, generation.requests, runtime._resolution_random_state), (1, [], 0))

    def test_initializer_calls_share_budget_with_pipeline(self):
        runtime, decision, generation = self.runtime([Answer(10, .1)], [Answer(None, .9, 'Context.')],
                                                     {"on_low_confidence": "random"}, max_calls=2)
        with self.assertRaisesRegex(OracleError, 'budget exhausted'):
            runtime.run('maybe c = null\nmaybe choice: Int ?= DecisionInput("Pick?",[10,20,30],c)\nsample(choice)')
        self.assertEqual((runtime.calls, len(decision.requests), len(generation.requests)), (2, 1, 1))
        self.assertEqual(runtime._resolution_random_state, 0)

    def test_threshold_honors_initializer_confidence_not_option_scores(self):
        runtime, _, _ = self.runtime([Answer(10, .99, scores=[1, 1, 1])],
            [Answer(None, .4, 'Context.'), Answer(20, .99, 'Final.')], {"on_low_confidence": "random"})
        result = runtime.run('maybe c = null\nmaybe choice: Int ?= DecisionInput("Pick?",[10,20,30],c)\nsample(choice)')
        self.assertEqual(result.resolution['strategy'], 'random')
        self.assertEqual([a['effectiveConfidence'] for a in result.resolution['attempts']], [.4, .4])
        self.assertEqual(runtime.calls, 3)

    def test_threshold_boundary_and_ordinary_collapse_gate_are_separate(self):
        for confidence, threshold in ((.8, .8), (0, 0), (1, 1), (.6, .5)):
            result, runtime, _, generation = self.snapshot([Answer(10, confidence)], config={"threshold": threshold})
            self.assertEqual((result.data, runtime.calls, generation.requests), (10, 1, []))
            if confidence < .8:
                with self.assertRaises(UncertaintyException):
                    runtime.collapse(result, runtime.global_scope)

    def test_rejected_original_context_taint_survives_final_fallback(self):
        runtime, _, _ = self.runtime([Answer(True, .9), Answer(10, .1)], [Answer(20, .9, 'Final.')])
        runtime.run('''val context = sample(1 ?!= 2)
maybe choice: Int ?= DecisionInput("Pick?", [10,20,30], context)
val s = sample(choice)''')
        self.assertEqual(runtime.calls, 3)
        with self.assertRaises(ExplainabilityError):
            runtime.run('s.because')
        self.assertEqual(runtime.global_scope.get('s').decision['kind'], 'difference')

    def test_resolution_history_is_tainted_while_final_fallback_explains(self):
        runtime, _, _ = self.runtime([Answer(10, .2)], [Answer(20, .9, 'Final.')])
        runtime.run(RECIPE + '\nval s = sample(choice)\nval history = s.resolution\nprint(s.because)')
        self.assertEqual(runtime.output.getvalue(), 'Final.\n')
        history = runtime.global_scope.get('history')
        self.assertEqual(history.decision, history.data['decision'])
        self.assertEqual(history.gate, .2)
        self.assertEqual(history.confidence, .9)
        self.assertEqual(history.decision['selected'], 10)
        for expression in ('history!!', 'history.attempts[0].value!!', 'history.resolution!!'):
            with self.subTest(expression=expression), self.assertRaises(UncertaintyException):
                runtime.run(expression)
        with self.assertRaises(ExplainabilityError):
            runtime.run('history.because')
        self.assertEqual(runtime.calls, 2)

    def test_attempt_reported_confidence_is_not_shadowed(self):
        runtime, _, _ = self.runtime([Answer(10, .2)], [Answer(20, .9, 'Final.')])
        runtime.run(RECIPE + '\nval s=sample(choice)\nprint(s.resolution.attempts[0].reportedConfidence)')
        self.assertEqual(runtime.output.getvalue(), '0.2\n')

    def test_trace_gate_keeps_unrelated_low_operand_on_first_and_repeat_extraction(self):
        runtime, _, _ = self.runtime([Answer(10, .9)], [Answer(2, .1, 'Low.')])
        runtime.run(RECIPE + '\nval s=sample(choice)\nmaybe low=2\nval combined=s + sample(low)')
        for expr in ('combined.resolution!!', 'combined.resolution.resolution!!'):
            with self.subTest(expr=expr), self.assertRaises(UncertaintyException):
                runtime.run(expr)
        self.assertEqual(runtime.calls, 2)

    def test_resolution_requires_trace_and_forces_once(self):
        with self.assertRaises(TypeError):
            Interpreter().run('1.resolution')
        runtime, _, _ = self.runtime([Answer(10, .9)])
        result = runtime.run(RECIPE + '\nchoice.resolution')
        self.assertEqual((runtime.calls, result.data['strategy']), (1, 'decision'))

    def test_first_trace_propagates_through_derived_values(self):
        runtime, _, _ = self.runtime([Answer(10, .9), Answer(20, .9)])
        runtime.run(RECIPE + '\nval a=sample(choice)\nval b=sample(choice)\nval both=a+b')
        both = runtime.global_scope.get('both')
        self.assertIs(both.resolution, runtime.global_scope.get('a').resolution)
        self.assertEqual(both.data, 30)

    def test_random_gate_allows_branch_and_deterministic_descendants(self):
        runtime, _, _ = self.runtime([Answer(True, .1)], [Answer(False, .2, 'Low.')],
                                     {"on_low_confidence": "random"})
        runtime.run('''maybe choice: Bool ?= DecisionInput("Q?",[false,true],null)
val s=sample(choice)
fun identity(v: Bool): Bool { return v }
if (identity([Pair(!s,s)].first().second)) { print(s.confidence, s!!) }''')
        self.assertEqual(runtime.output.getvalue(), '0.0 true\n')
        self.assertEqual(runtime.calls, 2)

    def test_random_gate_never_authorizes_unrelated_low_operands(self):
        expressions = ('s+low', '[s,low][0]', 'Pair(s,low).first',
                       'identity(s+low)', 's + [low][0]', '(s to low).first')
        for expression in expressions:
            runtime, _, _ = self.runtime([Answer(10, .1)],
                [Answer(20, .2, 'Low fallback.'), Answer(1, .1, 'Unrelated low.')],
                {"on_low_confidence": "random"})
            with self.subTest(expression=expression), self.assertRaises(UncertaintyException):
                runtime.run(RECIPE + '\nval s=sample(choice)\nmaybe low: Int=1\nfun identity(v: Int): Int { return v }\n(' + expression + ')!!')
            self.assertEqual(runtime.calls, 3)

    def test_random_gate_cannot_launder_through_if_condition(self):
        runtime, _, _ = self.runtime([Answer(True, .1)],
            [Answer(False, .1, 'Low.'), Answer(True, .1, 'Unrelated.')], {"on_low_confidence": "random"})
        with self.assertRaises(UncertaintyException):
            runtime.run('maybe x: Bool ?= DecisionInput("Q?",[false,true],null)\nval s=sample(x)\nmaybe low: Bool=true\nif (s && low) {}')

    def test_random_never_gains_explanation_through_ops_history_or_generation(self):
        expressions = ('s.because', '(s+1).because', 'identity(s).because',
                       's.decision.because', 's.resolution.because', 'sample(relay).because')
        for expression in expressions:
            runtime, _, _ = self.runtime([Answer(10, .1)],
                [Answer(20, .1, 'Fallback.'), Answer(20, .99, 'Fresh rationale.')],
                {"on_low_confidence": "random"})
            with self.subTest(expression=expression), self.assertRaises(ExplainabilityError):
                runtime.run(RECIPE + '\nval s=sample(choice)\nfun identity(v) { return v }\nmaybe relay=s\n' + expression)

    def test_history_never_gains_explanation_through_function_list_generation(self):
        for expression in ('identity(h).because', '[h][0].because', 'sample(relay).because'):
            runtime, _, _ = self.runtime([Answer(10, .2)],
                [Answer(20, .9, 'Final.'), Answer(1, .99, 'Fresh rationale.')])
            with self.subTest(expression=expression), self.assertRaises(ExplainabilityError):
                runtime.run(RECIPE + '\nval s=sample(choice)\nval h=s.resolution\nfun identity(v) {return v}\nmaybe relay=h\n' + expression)

    def test_random_history_uses_original_attempt_gate_and_decision(self):
        runtime, _, _ = self.runtime([Answer(10, .2)], [Answer(20, .3, 'Low.')],
                                     {"on_low_confidence": "random"})
        runtime.run(RECIPE + '\nval s=sample(choice)\nval h=s.resolution')
        history = runtime.global_scope.get('h')
        self.assertEqual((history.gate, history.confidence, history.decision['selected']), (.2, 0, 10))
        with self.assertRaises(UncertaintyException):
            runtime.run('h.attempts[0].value!!')

    def test_blind_keeps_original_context_and_source_without_first_response(self):
        runtime, _, generation = self.runtime([Answer(10, .2, 'DYNAMIC_RESPONSE_MARKER')],
                                               [Answer(20, .9, 'Final.')])
        source = 'maybe choice: Int ?= DecisionInput("Pick?", [10,20,30], "existing decision context")\nsample(choice)'
        runtime.run(source)
        request = generation.requests[0]
        self.assertEqual((request['source'], request['input']['context']), (source, 'existing decision context'))
        self.assertNotIn('DYNAMIC_RESPONSE_MARKER', json.dumps(request))
        self.assertNotIn('prior_decision', request)

    def test_route_scopes_use_current_lanes_and_restore_on_success_return_error(self):
        records = []
        class Routed:
            def __init__(self, route): self.route = route
            def ask(self, request):
                records.append((self.route, copy.deepcopy(request)))
                if self.route in ('dec', 'other_dec'): return Answer(10, .1)
                return Answer(20, .9, 'Final.')
        p = project({})
        with mock.patch.object(Project, 'oracle', side_effect=lambda route, **kw: Routed(route)):
            runtime = Interpreter(project=p, output=io.StringIO())
            runtime.run(RECIPE + '\nwith model("other_dec") { with model("other_gen") { print(choice) } }')
            self.assertEqual([r for r, _ in records], ['other_dec', 'other_gen'])
            self.assertEqual(records[1][1]['temperature'], .2)
            runtime.run('fun f() { with model("other_gen") { with temperature(0) { return sample(choice) } } }\nf()')
            self.assertEqual(records[-1][1]['temperature'], 0)
            with self.assertRaises(TypeError):
                runtime.run('with model("other_dec") { with model("other_gen") { with temperature(0) { val bad: Int=true } } }')
        self.assertEqual((runtime.generative_route, runtime.decision_route, runtime.temperature, runtime._temperature_depth),
                         ('gen', 'dec', .7, 0))

    def test_explicit_route_overrides_scope_without_mutation(self):
        records = []
        class Routed:
            def __init__(self, route): self.route=route
            def ask(self, request):
                records.append((self.route, copy.deepcopy(request)))
                return Answer(10, .1) if self.route == 'dec' else Answer(20, .9, 'Final.')
        with mock.patch.object(Project, 'oracle', side_effect=lambda route, **kw: Routed(route)):
            runtime = Interpreter(project=project({"decision": "dec", "generative": "gen"}), output=io.StringIO())
            runtime.run(RECIPE + '\nwith model("other_dec") { with model("other_gen") { print(choice) } }')
        self.assertEqual([r for r, _ in records], ['dec', 'gen'])
        self.assertEqual(records[1][1]['temperature'], .7)
        self.assertEqual((runtime.generative_route, runtime.decision_route), ('gen', 'dec'))

    def test_route_temperature_restore_after_fallback_transport_error(self):
        runtime, _, _ = self.runtime([Answer(10, .1)], [OracleError('transport')], {})
        # Model blocks use configured fixtures; no provider or environment access.
        with mock.patch.object(Project, 'oracle', return_value=Scripted(OracleError('transport'))):
            with self.assertRaises(OracleError):
                runtime.run(RECIPE + '\nwith model("other_gen") { with temperature(0) { sample(choice) } }')
        self.assertEqual((runtime.generative_route, runtime.decision_route, runtime.temperature, runtime._temperature_depth),
                         ('gen', 'dec', .7, 0))

    def test_external_defaults_have_no_false_named_route_attribution_in_either_lane(self):
        class Routed:
            def __init__(self, route): self.route = route
            def ask(self, request):
                return Answer(10, .1) if self.route == 'other_dec' else Answer(20, .9, 'Final.')
        for scoped_lanes, expected in (((), [None, None]),
                                      (('other_dec',), ['other_dec', None]),
                                      (('other_gen',), [None, 'other_gen']),
                                      (('other_dec', 'other_gen'), ['other_dec', 'other_gen'])):
            generation = Scripted(Answer(20, .9, 'External final.'), Answer(20, .9, 'External final.'))
            decision = Scripted(Answer(10, .1), Answer(10, .1))
            with self.subTest(scoped=scoped_lanes), mock.patch.object(
                    Project, 'oracle', side_effect=lambda route, **kw: Routed(route)):
                runtime = Interpreter(generation, decision_oracle=decision, project=project({}))
                body = 'return sample(choice)'
                for route in scoped_lanes:
                    body = 'with model("' + route + '") { ' + body + ' }'
                result = runtime.run(RECIPE + '\nfun scoped() { ' + body + ' }\nscoped()')
                self.assertEqual([a['route'] for a in result.resolution['attempts']], expected)
                restored = runtime.run('sample(choice)')
                self.assertEqual([a['route'] for a in restored.resolution['attempts']], [None, None])
                self.assertIs(runtime.oracle, generation)
                self.assertIs(runtime.decision_oracle, decision)
                for external in (generation, decision):
                    self.assertTrue(external.requests)
                    self.assertTrue(all('temperature_explicit' not in req for req in external.requests))

    def test_explicit_escalation_routes_are_attributed_despite_external_defaults(self):
        class Routed:
            def __init__(self, route): self.route = route
            def ask(self, request):
                return Answer(10, .1) if self.route == 'dec' else Answer(20, .9, 'Final.')
        generation, decision = Scripted(), Scripted()
        with mock.patch.object(Project, 'oracle', side_effect=lambda route, **kw: Routed(route)):
            runtime = Interpreter(generation, decision_oracle=decision,
                                  project=project({'decision': 'dec', 'generative': 'gen'}))
            result = runtime.run(RECIPE + '\nsample(choice)')
        self.assertEqual([a['route'] for a in result.resolution['attempts']], ['dec', 'gen'])
        self.assertEqual((generation.requests, decision.requests), ([], []))
        self.assertIs(runtime.oracle, generation)
        self.assertIs(runtime.decision_oracle, decision)


class RandomStreamTests(unittest.TestCase):
    def test_splitmix64_known_outputs_and_low64_seed(self):
        fixtures = [(0, 3, [1,0,1,1,1,0]), (1, 5, [0,4,0,0,1,3]),
                    (-1, 3, [2,0,1,0,0,1]), (19, 2, [0,0,1,1,0,1]),
                    (2**64+1, 5, [0,4,0,0,1,3])]
        for seed, count, expected in fixtures:
            runtime = Interpreter(seed=seed)
            with self.subTest(seed=seed, count=count):
                self.assertEqual([runtime.random_option_index(count) for _ in expected], expected)

    def test_rejection_discards_low_u64_before_modulo(self):
        # Seed 3 first emits 0x1d0b14e4db018fed (below threshold), then
        # 0xb3466f8a7b81a989. The first result must be discarded.
        runtime = Interpreter(seed=3)
        self.assertEqual(runtime.random_option_index(2**63+1), 3694763184872335752)
        self.assertEqual(runtime._resolution_random_state, 0x3c6ef372fe94f82d)

    def test_terminal_stream_independent_of_demo_samples(self):
        for seed in (0, 1, -1, 19):
            selections = []
            for extra in (0, 8):
                runtime = Interpreter(DemoOracle(seed), decision_oracle=Scripted(Answer(10, .1)),
                                      project=project({"on_low_confidence": "random"}), seed=seed,
                                      output=io.StringIO())
                runtime.run('maybe demo: Int="answer to everything"\n' + '\n'.join('sample(demo)' for _ in range(extra)))
                # The resolve fallback remains valid but low, independent of demo randomness.
                runtime.oracle = Scripted(Answer(20, .1, 'Fallback.'))
                result = runtime.run(RECIPE + '\nsample(choice)')
                selections.append(result.data)
            self.assertEqual(selections[0], selections[1])

    def test_successful_pipeline_does_not_advance_terminal_rng(self):
        for first_confidence, fallback in ((.9, []), (.1, [Answer(20, .9, 'Final.')])):
            runtime = Interpreter(Scripted(*fallback), decision_oracle=Scripted(Answer(10, first_confidence)),
                                  output=io.StringIO(), seed=-1)
            runtime.run(RECIPE + '\nsample(choice)')
            self.assertEqual(runtime._resolution_random_state, 2**64-1)


class EscalationManifestTests(unittest.TestCase):
    def test_defaults_remain_implicit_and_loading_is_read_only(self):
        data = manifest()
        self.assertNotIn('escalation', _validate(data))
        self.assertEqual(_validate(manifest({}))['escalation'], {})
        runtime = Interpreter(project=project({}), oracle=Scripted(), decision_oracle=Scripted())
        self.assertEqual(runtime.escalation, {})

    def test_strict_threshold_rejects_bool_nonfinite_out_of_range_and_bad_type(self):
        for value in (True, False, -.1, 1.1, float('nan'), float('inf'), '0.8', [], {}):
            with self.subTest(value=value), self.assertRaises(ConfigurationError):
                _validate(manifest({"threshold": value}))
        for value in (0, 1, .8):
            self.assertEqual(_validate(manifest({"threshold": value}))['escalation']['threshold'], value)

    def test_route_reference_kind_and_type_validation(self):
        for config in ({"decision": "gen"}, {"generative": "dec"}, {"decision": "missing"},
                       {"generative": []}, {"decision": True}, {"generative": ""}):
            with self.subTest(config=config), self.assertRaises(ConfigurationError):
                _validate(manifest(config))
        self.assertEqual(_validate(manifest({"decision": "other_dec", "generative": "other_gen"}))['escalation'],
                         {"decision": "other_dec", "generative": "other_gen"})

    def test_unknown_keys_tables_and_policy_values_rejected_without_echo(self):
        for config in ([], True, 'human', {"PRIVATE_MARKER": True}, {"exposure": 'PRIVATE_MARKER'},
                       {"exposure": True}, {"on_low_confidence": []}, {"on_low_confidence": 'human'}):
            with self.subTest(config=config), self.assertRaises(ConfigurationError) as caught:
                _validate(manifest(config))
            self.assertNotIn('PRIVATE_MARKER', str(caught.exception))

    def test_human_always_unsupported_and_no_humans_does_not_require_backend(self):
        for no_humans in (True, False):
            data = manifest({"on_low_confidence": 'human'})
            data['project']['no_humans_allowed'] = no_humans
            with self.subTest(no_humans=no_humans), self.assertRaises(ConfigurationError):
                _validate(data)
            data['escalation']['on_low_confidence'] = 'random'
            self.assertIs(_validate(data), data)

    def test_manifest_relocation_preserves_escalation_without_provider_construction(self):
        data = manifest({"decision": 'other_dec', "generative": 'other_gen', 'threshold': .7,
                         'exposure': 'informed', 'on_low_confidence': 'random'})
        # Inline TOML fixture supports every JSON value in this manifest.
        def toml(value):
            if isinstance(value, dict): return '{' + ','.join(json.dumps(k)+'='+toml(v) for k,v in value.items()) + '}'
            return json.dumps(value)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            text = '\n'.join(k+'='+toml(v) for k,v in data.items())
            for name in ('first', 'relocated'):
                target = root / name
                target.mkdir()
                (target/'purrhaps.toml').write_text(text)
                with mock.patch('purrhaps.providers.ConfiguredOracle', side_effect=AssertionError('provider construction')):
                    self.assertEqual(load_project(target).data, data)


class ResolveOracleTests(unittest.TestCase):
    def test_demo_resolve_first_option_and_honest_rationale(self):
        oracle = DemoOracle(9)
        request = {"kind": 'resolve', 'input': INPUT, 'options': [10,20,30], 'type': 'Int',
                   'protocol': 'purrhaps/1', 'line': 1, 'source': 'source', 'temperature': .7}
        answer = validate(oracle.ask(request), 'source', request=request)
        self.assertEqual((answer.value, answer.confidence, oracle.calls), (10, .92, 1))
        self.assertIn('offline demo', answer.because)

    def test_resolve_validation_keeps_generative_envelope(self):
        request = {'kind': 'resolve', 'options': [1,2]}
        result = validate(Answer(1, .8, 'Final.', [1], ['Caveat.']), 'source', request=request)
        self.assertEqual((result.because, result.citations, result.caveats), ('Final.', [1], ['Caveat.']))
        for answer in (Answer(True, .1, 'Final.'), Answer(1.0, .1, 'Final.'), Answer(3, .1, 'Final.'), Answer(1, .1)):
            with self.subTest(answer=answer), self.assertRaises(OracleError):
                validate(answer, 'source', request=request)

    def test_resolve_requires_valid_options_even_when_answer_is_high_confidence(self):
        for options in (None, [], [float('inf')], 'options'):
            with self.subTest(options=options), self.assertRaises(OracleError):
                validate(Answer(1, .99, 'Final.'), 'source',
                         request={'kind': 'resolve', 'options': options})


if __name__ == '__main__':
    unittest.main()
