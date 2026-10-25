"""Oracles return structured claims; the runtime decides whether to trust them."""
from dataclasses import dataclass, field, replace
import json
import math
import random
import re
import subprocess
from .syntax import PurrhapsError
from .types import is_json, strict_equal


class OracleError(PurrhapsError):
    pass


class ConfabulationError(PurrhapsError):
    pass


class HedgeOverflowError(PurrhapsError):
    pass


@dataclass
class Answer:
    value: object
    confidence: float
    because: str | None = None
    citations: list[int] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    scores: list[float] | None = None


def validate(answer, source, *, decision=False, options=None, request=None):
    """Validate a generative claim or an explicitly bounded decision response.

    Supplying a decision request also selects decision validation. Decisions
    discard provider rationale rather than validating or exposing it.
    """
    if request is not None:
        decision = decision or request.get("kind") in ("decision", "difference")
        options = request.get("options", options)
    if not isinstance(answer, Answer):
        raise OracleError("oracle must return an Answer")
    if (type(answer.confidence) not in (int, float)
            or not 0 <= answer.confidence <= 1 or not math.isfinite(answer.confidence)):
        raise OracleError("confidence must be a finite number between 0 and 1")
    if not is_json(answer.value):
        raise OracleError("oracle value must be finite JSON data")
    # Resolve keeps the generative rationale envelope while enforcing the same
    # strict, recursive option membership as the decision lane.
    if decision or options is not None or (request is not None and request.get("kind") == "resolve"):
        if type(options) is not list or not options or not is_json(options):
            raise OracleError("decision validation requires finite JSON request options")
        if not any(strict_equal(answer.value, option) for option in options):
            raise OracleError("decision value must strictly match one supplied option")
        if decision and answer.scores is not None and (
                type(answer.scores) is not list or len(answer.scores) != len(options)
                or any(type(s) not in (int, float) or not 0 <= s <= 1 or not math.isfinite(s) for s in answer.scores)):
            raise OracleError("decision scores must align with options and be finite numbers between 0 and 1")
        if decision:
            return replace(answer, because=None, citations=[], caveats=[])
    if not isinstance(answer.because, str) or not answer.because.strip():
        raise OracleError("because must be a nonempty string")
    if not isinstance(answer.citations, list) or not all(type(x) is int for x in answer.citations):
        raise OracleError("citations must be a list of integer source line numbers")
    if not isinstance(answer.caveats, list) or not all(isinstance(x, str) for x in answer.caveats):
        raise OracleError("caveats must be a list of strings")
    # The protocol requires structured citations; catch simple prose citations too.
    prose_lines = [int(n) for n in re.findall(r'\bline\s+(\d+)\b', answer.because, re.I)]
    for line in answer.citations + prose_lines:
        if line < 1 or line > len(source.splitlines()):
            raise ConfabulationError(f"oracle cited nonexistent line {line}")
    prose_hedges = re.findall(r'\b(?:perhaps|maybe|possibly|probably|however|although|uncertain)\b', answer.because, re.I)
    if max(len(answer.caveats), len(prose_hedges)) > 3:
        raise HedgeOverflowError("rationale contains more than three caveats; please meow decisively")
    try:
        json.dumps(answer.value, allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise OracleError("oracle value must be finite JSON data") from exc
    return answer


class DemoOracle:
    """An offline toy, not an LLM. Seeded variety at nonzero temperature."""
    def __init__(self, seed=0):
        self.random = random.Random(seed)
        self.calls = 0

    def ask(self, request):
        self.calls += 1
        value = request["input"]
        typ = request.get("type")
        if request["kind"] == "decision":
            # A deliberately modest offline demo: first permitted choice.
            return Answer(request["options"][0], .92)
        if request["kind"] == "resolve":
            return Answer(request["options"][0], .92,
                          "The offline demo selected the first permitted option; no model inference was performed.")
        if request["kind"] in ("equivalence", "difference"):
            a, b = value
            def normalize(x):
                return str(x).casefold().strip().strip(".!?")
            synonyms = ({"sunny", "happy", "cheerful", "☀️"}, {"cat", "kitty", "feline"})
            a, b = normalize(a), normalize(b)
            result = a == b or any(a in group and b in group for group in synonyms)
            if request["kind"] == "difference":
                return Answer(not result, .94)
            return Answer(result, .94, "The demo oracle compared normalized words and its tiny synonym table.", [request["line"]])
        text = str(value).casefold()
        confidence = .92
        rationale = "The demo oracle preserved the declaration's input."
        if value == {"self": "purrhaps"}:
            value, confidence = "I can't introspect reliably", .26
            rationale = "The demo oracle has no introspective machinery."
        elif typ == "Int" and isinstance(value, str):
            numbers = re.findall(r'-?\d+', value)
            if "answer to everything" in text or "meaning of life" in text:
                value = 42
                rationale = "The declaration strongly implied 42."
            elif numbers:
                value = int(numbers[0])
                rationale = "The demo oracle extracted the first integer."
            else:
                value, confidence = 0, .4
                rationale = "The demo oracle could not infer an integer."
            if request["temperature"] > 0 and confidence >= .8:
                value += self.random.choice([0, 0, 0, -1, 1])
        elif typ == "String":
            value = str(value)
        elif typ == "Float" and type(value) in (int, float):
            value = float(value)
        return Answer(value, confidence, rationale, [request["line"]])


class CommandOracle:
    """One JSON request on stdin, one JSON answer on stdout. Never uses a shell."""
    def __init__(self, command, timeout=30):
        if not command:
            raise OracleError("oracle command cannot be empty")
        self.command, self.timeout, self.calls = command, timeout, 0

    def ask(self, request):
        self.calls += 1
        try:
            completed = subprocess.run(self.command, input=json.dumps(request), text=True,
                                       capture_output=True, timeout=self.timeout, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise OracleError(f"oracle command failed ({type(exc).__name__})") from None
        if completed.returncode:
            # Child stderr may contain secrets. Do not echo it into diagnostics.
            raise OracleError(f"oracle command exited with status {completed.returncode}")
        try:
            data = json.loads(completed.stdout)
            if not isinstance(data, dict):
                raise ValueError()
            return Answer(**data)
        except (TypeError, ValueError):
            raise OracleError("oracle command returned an invalid answer object") from None
