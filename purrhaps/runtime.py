"""Lazy uncertainty, lexical environments, and a very small ecology."""
from dataclasses import dataclass
import builtins
import operator
import sys
from .syntax import Parser, PurrhapsError
from .oracle import DemoOracle, validate, OracleError
from .types import validate_data, is_json
from . import stdlib


class UncertaintyException(PurrhapsError):
    pass


class TypeError(PurrhapsError):
    pass


class NameError(PurrhapsError):
    pass


class ExplainabilityError(PurrhapsError):
    pass


@dataclass
class Value:
    data: object
    confidence: float = 1.0
    because: str | None = None
    citations: tuple = ()
    generative: bool = False
    decision: dict | None = None
    resolution: dict | None = None
    gate_confidence: float | None = None
    resolution_gate: float | None = None

    @property
    def gate(self):
        return self.confidence if self.gate_confidence is None else self.gate_confidence


class Scope:
    def __init__(self, label, parent=None):
        self.label, self.parent, self.bindings = label, parent, {}

    def get(self, name):
        if name in self.bindings:
            return self.bindings[name]
        if self.parent:
            return self.parent.get(name)
        raise NameError(f"unknown name {name!r}")

    def define(self, name, value):
        if name in self.bindings:
            raise NameError(f"{name!r} is already declared in this scope")
        self.bindings[name] = value


class Maybe:
    def __init__(self, sample, cached=False):
        self.sample, self.cached = sample, cached
        self.cache = {}
        self.owner = None

    def read(self, scope):
        self.owner = scope.label
        if not self.cached:
            return self.sample(scope)
        if scope not in self.cache:
            self.cache[scope] = self.sample(scope)
        return self.cache[scope]


@dataclass
class Function:
    params: list
    result_type: str | None
    body: list
    closure: Scope


class Returned(Exception):
    def __init__(self, value):
        self.value = value


def force(value, scope):
    return value.read(scope) if isinstance(value, Maybe) else value


def check_type(value, typ):
    try:
        validate_data(value.data, typ)
    except ValueError as exc:
        raise TypeError(str(exc)) from None
    return value


def display(data):
    return stdlib.display(data)


def derived(data, values, *, because=None):
    """Whole-value evidence propagation; keep the first decision/resolution trace.

    This is conservative value taint, not a full evidence graph or a tracker of
    implicit control-flow dependencies.
    """
    generative = any(v.generative for v in values)
    decision = next((v.decision for v in values if v.decision is not None), None)
    traced = next((v for v in values if v.resolution is not None), None)
    if not generative or decision is not None:
        because = None
    elif because is None:
        because = "Derived from: " + " | ".join(v.because for v in values if v.because)
    return Value(data, min((v.confidence for v in values), default=1), because,
                 tuple(dict.fromkeys(n for v in values for n in v.citations)), generative, decision,
                 traced.resolution if traced is not None else None,
                 min((v.gate for v in values), default=1),
                 traced.resolution_gate if traced is not None else None)


class Interpreter:
    def __init__(self, oracle=None, *, decision_oracle=None, output=None, threshold=.8, max_calls=None,
                 project=None, generative_route=None, decision_route=None, seed=0, timeout_override=None):
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        if max_calls is None:
            max_calls = project.data.get("runtime", {}).get("max_calls", 1000) if project else 1000
        if type(max_calls) is not int or max_calls < 1:
            raise ValueError("max_calls must be positive")
        self.project, self.seed, self.timeout_override = project, seed, timeout_override
        # Portable terminal-choice stream, deliberately separate from demo RNGs.
        self._resolution_random_state = seed & ((1 << 64) - 1)
        self.escalation = project.data.get("escalation", {}) if project else {}
        self.generative_route, self.decision_route = generative_route, decision_route
        self._route_oracles = {}
        self._profile_temperature = None
        if project is not None:
            self.generative_route = (generative_route if generative_route is not None
                                     else project.data["defaults"]["generative"])
            self.decision_route = (decision_route if decision_route is not None
                                   else project.data["defaults"]["decision"])
            _, model = project.route(self.generative_route, "generative")
            self._profile_temperature = model.get("temperature") if oracle is None else None
            project.route(self.decision_route, "decision")
            if oracle is None:
                oracle = self.route_oracle(self.generative_route)
            if decision_oracle is None:
                decision_oracle = self.route_oracle(self.decision_route)
        elif generative_route is not None or decision_route is not None:
            from .project import ConfigurationError
            raise ConfigurationError("named model routes require a project manifest")
        self.oracle = oracle if oracle is not None else DemoOracle(seed)
        self.decision_oracle = decision_oracle if decision_oracle is not None else DemoOracle(seed)
        self.output = output if output is not None else sys.stdout
        self.threshold, self.max_calls = threshold, max_calls
        self.temperature, self.calls, self.scope_count = .7, 0, 0
        self._temperature_depth = 0
        self.source = ""
        self.global_scope = Scope("module")
        self.global_scope.define("this", Value({"self": "purrhaps"}))

    def route_oracle(self, route):
        if route not in self._route_oracles:
            self._route_oracles[route] = self.project.oracle(route, seed=self.seed,
                                                            timeout_override=self.timeout_override)
        return self._route_oracles[route]

    def child(self, parent, label):
        self.scope_count += 1
        return Scope(f"{label}#{self.scope_count}", parent)

    def ask(self, kind, value, typ, line, *, options=None, prior_decision=None, route=None):
        if self.calls >= self.max_calls:
            raise OracleError(f"oracle call budget exhausted ({self.max_calls})")
        self.calls += 1
        decision = kind in ("decision", "difference")
        oracle = self.decision_oracle if decision else self.oracle
        profile_temperature = self._profile_temperature
        if route is not None:
            _, model = self.project.route(route, "decision" if decision else "generative")
            oracle = self.route_oracle(route)
            if not decision:
                profile_temperature = model.get("temperature")
        request = {
            "protocol": "purrhaps/2" if decision else "purrhaps/1", "kind": kind, "input": value,
            "type": typ, "line": line, "source": self.source,
            "temperature": (profile_temperature if not decision and self._temperature_depth == 0
                            and profile_temperature is not None else self.temperature),
        }
        if decision or kind == "resolve":
            request["options"] = options
        if prior_decision is not None:
            request["prior_decision"] = prior_decision
        if self.project is not None and any(oracle is configured for configured in self._route_oracles.values()):
            request["temperature_explicit"] = self._temperature_depth > 0
        answer = validate(oracle.ask(request), self.source, decision=decision, options=options)
        evidence = {"kind": kind, "input": value, "options": options,
                    "selected": answer.value, "scores": answer.scores} if decision else None
        result = Value(answer.value, answer.confidence, answer.because, tuple(answer.citations),
                       not decision, evidence)
        return check_type(result, typ)

    def random_option_index(self, count):
        """SplitMix64 plus rejection sampling, shared with the native contract."""
        mask = (1 << 64) - 1
        threshold = (1 << 64) % count
        while True:
            self._resolution_random_state = (self._resolution_random_state + 0x9e3779b97f4a7c15) & mask
            z = self._resolution_random_state
            z = ((z ^ (z >> 30)) * 0xbf58476d1ce4e5b9) & mask
            z = ((z ^ (z >> 27)) * 0x94d049bb133111eb) & mask
            z ^= z >> 31
            if z >= threshold:
                return z % count

    def resolve(self, seed, typ, line):
        check_type(seed, "DecisionInput")
        options = seed.data["options"]
        # Reject an invalid *unselected* option before spending any calls.
        for option in options:
            check_type(Value(option), typ)
        threshold = self.escalation.get("threshold", .8)
        exposure = self.escalation.get("exposure", "blind")
        attempts = []

        def attempt(kind, route, prior=None):
            answer = self.ask(kind, seed.data, typ, line, options=options,
                              prior_decision=prior, route=route)
            effective = min(answer.confidence, seed.confidence)
            active_route = self.decision_route if kind == "decision" else self.generative_route
            active_oracle = self.decision_oracle if kind == "decision" else self.oracle
            # CLI/injected adapters override the active defaults without being
            # the configured route. Attribute a name only to its actual adapter.
            attributed_route = (active_route if self._route_oracles.get(active_route) is active_oracle else None)
            attempts.append({"kind": "decision" if kind == "decision" else "generative",
                             "route": route if route is not None else attributed_route,
                             "value": answer.data, "reportedConfidence": float(answer.confidence),
                             "effectiveConfidence": float(effective),
                             "scores": answer.decision["scores"] if answer.decision else None})
            return answer, effective

        decision, confidence = attempt("decision", self.escalation.get("decision"))
        if confidence >= threshold:
            result = derived(decision.data, [decision, seed])
            strategy = "decision"
        else:
            prior = ({"value": decision.data, "confidence": decision.confidence,
                      "scores": decision.decision["scores"]} if exposure == "informed" else None)
            answer, confidence = attempt("resolve", self.escalation.get("generative"), prior)
            if confidence >= threshold:
                # The rejected decision is history, never a dependency of this
                # answer. Original context evidence still propagates normally.
                result = derived(answer.data, [seed, answer], because=answer.because)
                strategy = "generative"
            elif self.escalation.get("on_low_confidence", "error") == "random":
                selected = options[self.random_option_index(len(options))]
                evidence = {"kind": "random", "input": seed.data, "options": options,
                            "selected": selected, "scores": None}
                result = Value(selected, 0.0, citations=seed.citations,
                               decision=evidence, gate_confidence=1.0)
                strategy = "random"
            else:
                raise UncertaintyException("escalation answers are below the configured confidence threshold")
        result.resolution = {"strategy": strategy, "exposure": exposure,
                             "forced": strategy == "random", "threshold": float(threshold),
                             "selectionProbability": 1.0 / len(options) if strategy == "random" else None,
                             "decision": decision.decision, "attempts": attempts}
        result.resolution_gate = min(seed.gate, *(a["reportedConfidence"] for a in attempts))
        return result

    def collapse(self, value, scope):
        sample = force(value, scope)
        if sample.gate < self.threshold:
            reason = f": {sample.because}" if sample.generative and sample.decision is None else ""
            raise UncertaintyException(
                f"confidence {sample.confidence:.2f} is below {self.threshold:.2f}{reason}")
        return sample

    def run(self, source):
        self.source = source
        body = Parser(source).program()
        try:
            return self.execute(body, self.global_scope)
        except Returned:
            raise PurrhapsError("return is only valid inside a function") from None
        except RecursionError:
            raise PurrhapsError("the cat chased its own tail: recursion limit exceeded") from None

    def execute(self, body, scope):
        result = Value(None)
        for statement in body:
            try:
                result = self.statement(statement, scope)
            except PurrhapsError as exc:
                if not str(exc).startswith("line "):
                    exc.args = (f"line {statement.line}: {exc}",)
                raise
        return result

    def statement(self, node, scope):
        kind, args = node.kind, node.args
        if kind == "escalate":
            name, typ, expr = args
            scope.define(name, Maybe(lambda reader: self.resolve(
                force(self.eval(expr, scope), reader), typ, node.line)))
            return Value(None)
        if kind == "declare":
            mode, name, typ, expr = args
            if mode == "val":
                value = self.eval(expr, scope)
                if isinstance(value, Maybe):
                    original = value
                    value = Maybe(lambda reader: check_type(force(original, reader), typ))
                else:
                    check_type(value, typ)
            else:
                def sample(reader):
                    seed = force(self.eval(expr, scope), reader)
                    if mode == "definitely":
                        if type(seed.data) is dict and seed.data.keys() == {"question", "options", "context"}:
                            check_type(seed, "DecisionInput")
                            answer = self.ask("decision", seed.data, typ, node.line, options=seed.data["options"])
                            # New choice evidence takes priority over evidence in its context.
                            return derived(answer.data, [answer, seed])
                        if seed.decision is not None:
                            return check_type(seed, typ)
                        raise TypeError("definitely needs a DecisionInput(question, options, context) or a decision expression such as ?!=; meow, supply bounded choices")
                    answer = self.ask("read", seed.data, typ, node.line)
                    return derived(answer.data, [seed, answer], because=answer.because)
                value = Maybe(sample, cached=mode == "perhaps")
            scope.define(name, value)
            return Value(None)
        if kind == "function":
            name, params, typ, body = args
            if len({p[0] for p in params}) != len(params):
                raise NameError("duplicate function parameter")
            scope.define(name, Function(params, typ, body, scope))
            return Value(None)
        if kind == "expression":
            return self.eval(args[0], scope)
        if kind == "return":
            raise Returned(self.eval(args[0], scope))
        if kind == "if":
            cond, yes, no = args
            value = self.collapse(self.eval(cond, scope), scope)
            check_type(value, "Bool")
            return self.execute(yes if value.data else no, self.child(scope, "branch"))
        if kind == "temperature":
            temp, body = args
            value = self.eval(temp, scope)
            if isinstance(value, Maybe):
                raise TypeError("temperature needs a definite number; use !!")
            if type(value.data) not in (float, int) or not 0 <= value.data <= 2:
                raise TypeError("temperature must be a number between 0 and 2")
            previous = self.temperature
            self.temperature = value.data
            self._temperature_depth += 1
            try:
                return self.execute(body, self.child(scope, "temperature"))
            finally:
                self.temperature = previous
                self._temperature_depth -= 1
        if kind == "model":
            from .project import ConfigurationError
            if self.project is None:
                raise ConfigurationError("with model requires a project manifest")
            route, body = args
            value = self.eval(route, scope)
            if isinstance(value, Maybe) or type(value.data) is not str:
                raise TypeError("model route needs a definite String; use !!")
            _, model = self.project.route(value.data)
            oracle = self.route_oracle(value.data)
            if model["kind"] == "generative":
                previous = self.oracle, self.generative_route, self._profile_temperature
                self.oracle, self.generative_route = oracle, value.data
                self._profile_temperature = model.get("temperature")
                try:
                    return self.execute(body, self.child(scope, "model"))
                finally:
                    self.oracle, self.generative_route, self._profile_temperature = previous
            previous = self.decision_oracle, self.decision_route
            self.decision_oracle, self.decision_route = oracle, value.data
            try:
                return self.execute(body, self.child(scope, "model"))
            finally:
                self.decision_oracle, self.decision_route = previous
        raise PurrhapsError(f"unsupported statement {kind}")

    def lift(self, values, operation):
        def sample(scope):
            concrete = [force(v, scope) for v in values]
            try:
                data = operation(*[v.data for v in concrete])
                if not is_json(data):
                    raise TypeError("operation must produce finite JSON data")
            except (ArithmeticError, builtins.TypeError, ValueError, KeyError, IndexError) as exc:
                raise TypeError(f"invalid operation: {exc}") from None
            return derived(data, concrete)
        return Maybe(sample) if any(isinstance(v, Maybe) for v in values) else sample(self.global_scope)

    def eval(self, node, scope):
        kind, args = node.kind, node.args
        if kind == "literal":
            return check_type(Value(args[0], citations=(node.line,)), None)
        if kind == "list":
            return self.lift([self.eval(item, scope) for item in args[0]], lambda *items: list(items))
        if kind == "index":
            return self.lift([self.eval(arg, scope) for arg in args], stdlib.index)
        if kind == "name":
            value = scope.get(args[0])
            if isinstance(value, Function):
                raise TypeError("functions must be called")
            return value
        if kind == "collapse":
            return self.collapse(self.eval(args[0], scope), scope)
        if kind == "attribute":
            expr, name = args
            value = self.eval(expr, scope)
            if name == "owner":
                return Value(value.owner if isinstance(value, Maybe) else "certain")
            if name in ("because", "confidence", "citations", "decision", "resolution"):
                sample = force(value, scope)
                if name == "because" and (not sample.generative or sample.decision is not None):
                    raise ExplainabilityError(".because requires generative evidence without decision evidence; this value has no generative explanation, meow")
                if name == "decision":
                    if sample.decision is None:
                        raise TypeError(".decision requires decision evidence")
                    return derived(sample.decision, [sample])
                if name == "resolution":
                    if sample.resolution is None:
                        raise TypeError(".resolution requires escalation evidence")
                    history = derived(sample.resolution, [sample])
                    history.decision = sample.resolution["decision"]
                    history.because = None
                    history.gate_confidence = min(sample.gate, sample.resolution_gate)
                    return history
                if name == "citations":
                    return Value(list(sample.citations))
                return Value(getattr(sample, name), because="Metadata of one sampled value.")
            return self.lift([value], lambda data: stdlib.attribute(data, name))
        if kind == "unary":
            op, expr = args
            def unary(x):
                if op == "!":
                    if type(x) is not bool:
                        raise TypeError("! requires Bool")
                    return not x
                if type(x) not in (int, float):
                    raise TypeError("unary - requires a number")
                return -x
            return self.lift([self.eval(expr, scope)], unary)
        if kind == "binary":
            op, left, right = args
            a, b = self.eval(left, scope), self.eval(right, scope)
            if op in ("==", "!=") and any(isinstance(v, Maybe) for v in (a, b)):
                raise TypeError("== and != are undefined on maybes; use ~= or collapse with !!")
            if op in ("~=", "?!="):
                def judge(reader):
                    av, bv = force(a, reader), force(b, reader)
                    answer = self.ask("difference" if op == "?!=" else "equivalence", [av.data, bv.data], "Bool", node.line,
                                      options=[False, True] if op == "?!=" else None)
                    return derived(answer.data, [answer, av, bv], because=answer.because)
                return Maybe(judge)
            if op == "to":
                return self.lift([a, b], lambda x, y: stdlib.intrinsic("Pair", [x, y]))
            operations = {"+": operator.add, "-": operator.sub, "*": operator.mul,
                          "/": operator.truediv, "%": operator.mod, "==": operator.eq,
                          "!=": operator.ne, "<": operator.lt, ">": operator.gt,
                          "<=": operator.le, ">=": operator.ge,
                          "&&": lambda x, y: x and y, "||": lambda x, y: x or y}
            def binary(x, y):
                if op in ("&&", "||") and (type(x) is not bool or type(y) is not bool):
                    raise TypeError(f"{op} requires Bool operands")
                if op in ("+", "-", "*", "/", "%"):
                    numeric = type(x) in (int, float) and type(y) in (int, float)
                    strings = op == "+" and type(x) is str and type(y) is str
                    if not numeric and not strings:
                        raise TypeError(f"{op} requires numbers (or two strings for +)")
                return operations[op](x, y)
            return self.lift([a, b], binary)
        if kind == "call":
            target, expressions = args
            if target.kind == "attribute":
                receiver, name = target.args
                values = [self.eval(receiver, scope)] + [self.eval(expr, scope) for expr in expressions]
                return self.lift(values, lambda data, *items: stdlib.method(data, name, items))
            if target.kind != "name":
                raise TypeError("only named functions or member methods can be called")
            name = target.args[0]
            if name == "sample" and len(expressions) != 1:
                raise TypeError("sample expects exactly one argument")
            values = [self.eval(expr, scope) for expr in expressions]
            if name == "sample":
                return force(values[0], scope)
            if name == "hi":
                if values:
                    raise TypeError("hi() takes no arguments")
                return Value("hi! 🐈‍⬛", because="The standard library's sole certainty.")
            if name == "print":
                print(*[display(force(v, scope).data) for v in values], file=self.output)
                return Value(None, because="The print intrinsic completed.")
            if name in stdlib.INTRINSICS:
                return self.lift(values, lambda *items: stdlib.intrinsic(name, items))
            function = scope.get(name)
            if not isinstance(function, Function):
                raise TypeError(f"{name!r} is not callable")
            if len(values) != len(function.params):
                raise TypeError(f"{name} expects {len(function.params)} arguments, got {len(values)}")
            child = self.child(function.closure, name)
            for (param, typ), value in zip(function.params, values):
                if isinstance(value, Maybe):
                    wrapped = Maybe(lambda reader, v=value, t=typ: check_type(force(v, reader), t))
                    child.define(param, wrapped)
                else:
                    child.define(param, check_type(value, typ))
            try:
                self.execute(function.body, child)
                result = Value(None)
            except Returned as returned:
                result = returned.value
            if isinstance(result, Maybe):
                return Maybe(lambda reader: check_type(force(result, reader), function.result_type))
            return check_type(result, function.result_type)
        raise PurrhapsError(f"unsupported expression {kind}")
