"""Local Ollama System One adapter for Purrhaps's finite-choice decision route.

Uses /v1/systemone, never chat generation. See docs/oracles.md for the mapping.
No model installation or credential discovery is performed.
"""
import argparse
import json
import math
import sys
from urllib.request import Request, urlopen

from .types import strict_equal, validate_data
from .syntax import PurrhapsError


def payload(request, model):
    if request.get("protocol") != "purrhaps/2":
        raise ValueError("decision adapter requires purrhaps/2")
    kind, value = request["kind"], request["input"]
    options = request["options"]
    if kind == "decision":
        validate_data(value, "DecisionInput")
        if not strict_equal(options, value["options"]):
            raise ValueError("request options disagree with DecisionInput")
        question, context = value["question"], value["context"]
    elif kind == "difference":
        if type(value) is not list or len(value) != 2 or not strict_equal(options, [False, True]):
            raise ValueError("difference needs two operands and Boolean options")
        question = "Are left and right meaningfully different?"
        context = {"left": value[0], "right": value[1]}
    else:
        raise ValueError("unsupported decision kind")
    if not 2 <= len(options) <= 26:
        raise ValueError("Ollama choice requires 2 to 26 options")
    validate_data(context, "Json")
    # Stable opaque labels preserve heterogeneous JSON values, including true vs 1.
    criteria = {f"option_{i}": json.dumps(option, ensure_ascii=False, allow_nan=False)
                for i, option in enumerate(options)}
    return {"model": model, "state": json.dumps(context, ensure_ascii=False, allow_nan=False),
            "questions": {"purrhaps": {"type": "choice", "instructions": question,
                                        "criteria": criteria}}}


def _score(value):
    if type(value) not in (int, float) or not 0 <= value <= 1 or not math.isfinite(value):
        raise ValueError("decision score must be finite and between 0 and 1")
    return value


def query(request, model, host, timeout):
    body = json.dumps(payload(request, model), allow_nan=False).encode()
    http = Request(host.rstrip('/') + '/v1/systemone', data=body,
                   headers={"Content-Type": "application/json"})
    with urlopen(http, timeout=timeout) as response:
        envelope = json.load(response)
    answer = envelope["answers"]["purrhaps"]
    if answer.get("type") != "choice":
        raise ValueError("expected a choice response")
    labels = [f"option_{i}" for i in range(len(request["options"]))]
    choice, probabilities = answer["choice"], answer["probabilities"]
    if choice not in labels or type(probabilities) is not dict or set(probabilities) != set(labels):
        raise ValueError("decision returned unknown or missing options")
    scores = [_score(probabilities[label]) for label in labels]
    value = request["options"][labels.index(choice)]
    validate_data(value, request.get("type"))
    return {"value": value, "confidence": _score(answer["confidence"]), "scores": scores}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='an installed System One decision model')
    parser.add_argument('--host', default='http://127.0.0.1:11434')
    parser.add_argument('--timeout', type=float, default=120)
    options = parser.parse_args(argv)
    if not math.isfinite(options.timeout) or options.timeout <= 0:
        parser.error('timeout must be finite and positive')
    try:
        request = json.load(sys.stdin)
        print(json.dumps(query(request, options.model, options.host, options.timeout), allow_nan=False))
        return 0
    except (OSError, ValueError, TypeError, KeyError, AttributeError, PurrhapsError) as exc:
        print(f'Ollama decision failed ({type(exc).__name__}). Check the server and model.', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
