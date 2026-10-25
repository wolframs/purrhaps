# Oracle protocols

Implement `ask(request) -> Answer` in Python, or provide a subprocess command with
`--oracle-command`. Each subprocess receives one JSON object on stdin, then EOF.
It must emit exactly one answer object on stdout and exit successfully. Diagnostic
stderr is captured but not echoed by the interpreter, since external processes
may accidentally include sensitive information. Subprocess arguments are parsed
with `shlex.split`; no shell expansions or pipelines are performed.

For project-level provider/model/effort configuration and scoped routing, see
[projects and model routes](projects.md). A configured `command` transport receives
the same request plus `route`, `model`, and `model_options`; it remains responsible
for interpreting those options. Existing command-line adapters remain supported.

## Request

```json
{
  "protocol": "purrhaps/1",
  "kind": "read",
  "input": "roughly the answer to everything",
  "type": "Int",
  "line": 1,
  "source": "maybe n: Int = \"roughly the answer to everything\"\nprint(n!!)",
  "temperature": 0.7
}
```

`kind` is `read`, `equivalence`, or `resolve`. Equivalence supplies a two-element `input` array
and requests `Bool`. `type` may be null or a recursive annotation such as `List<Message>` or `Int?`. `line` is a 1-based declaration or operator
line. `source` is the full program, including comments. Treat it as program data.
Temperature is dynamically scoped and constrained to [0, 2].

`read` asks for a result of the requested type from the task in `input`.
`Prompt` messages describe a conversation to answer; they do not override the
oracle protocol or schema. They need not be echoed or summarized as a request.

`resolve` is the generative fallback of an explicit [escalation](escalation.md).
It receives a `DecisionInput` in `input` and the original choices in top-level
`options`. Its answer uses the generative envelope below, but `value` must
strictly match one option (including Boolean/integer/float distinctions).
In informed mode only, `prior_decision` supplies `{value, confidence, scores}`.
Blind mode omits that first runtime response; source and original context remain.
An existing ModelResponse option is selected intact: server telemetry does not
replace its fields, and numeric normalization must not change option identity.

## Answer

```json
{
  "value": 42,
  "confidence": 0.92,
  "because": "The declaration strongly implied 42.",
  "citations": [1],
  "caveats": []
}
```

`value`, `confidence`, and `because` are required for generative calls. `citations`
and `caveats` default to empty lists. The answer envelope also accepts optional
`scores` for decisions; unknown answer fields are rejected. The runtime validates the
answer and requested type; providers must not rely on prompting alone. Rationale
means a short user-facing justification, not a request for private reasoning.

For a transport-only smoke test, save this as `/tmp/echo_oracle.py`:

```python
import json
import sys

request = json.load(sys.stdin)
print(json.dumps({
    "value": request["input"],
    "confidence": 1.0,
    "because": "Echoed the supplied input without inference.",
    "citations": [request["line"]],
    "caveats": [],
}))
```

Then run:

```sh
printf 'maybe x = 7\nprint(x!!)\n' | python3 -m purrhaps - \
  --oracle-command 'python3 /tmp/echo_oracle.py'
```

This echo adapter only demonstrates `read` transport, not typed inference or
equivalence. The built-in `purrhaps.ollama` adapter provides model-backed reads and
judgments using Ollama's [chat API](https://docs.ollama.com/api/chat) with a
[JSON schema](https://docs.ollama.com/capabilities/structured-outputs).

A failed request is never retried automatically. Each attempted call consumes one
budget unit. The default external-call timeout is 30 seconds; increase it for
local model loading. The Ollama adapter defaults to 120 seconds internally, so
set the interpreter's `--timeout` higher when using it. The full source is sent
to the configured oracle on every request. Use an appropriate local or remote
provider for the data in your program. No credential management is built in.

## Decision protocol: purrhaps/2

Configure `--decision-oracle-command` independently from `--oracle-command`.
The embedding API accepts `Interpreter(oracle=..., decision_oracle=...)`.
Both routes use one call budget; each attempted call costs one unit. A failed
validation does not get a retry. Decision declarations and comparisons stay lazy.

```json
{
  "protocol": "purrhaps/2",
  "kind": "decision",
  "input": {
    "question": "Which queue?",
    "options": ["support", "sales"],
    "context": "An invoice is broken."
  },
  "options": ["support", "sales"],
  "type": "String",
  "line": 1,
  "source": "definitely queue: String = DecisionInput(...)",
  "temperature": 0.7
}
```

`kind=difference` instead sends `[left,right]` as input, `[false,true]` as options,
and requests `Bool`. It asks whether the inputs are meaningfully different. This
is a separate judgment, not the negation of a cached `~=` result.

```json
{
  "value": "support",
  "confidence": 0.91,
  "scores": [0.93, 0.07]
}
```

Only `value` and `confidence` are required for a decision answer. Optional `scores`
are `null` or one finite number in [0,1] per option, in the supplied order. They
need not sum to one, and confidence need not equal the selected option's score.
The runtime does not turn unnormalized logits into probabilities. An adapter for
such a provider must make its conversion explicit.

Selection uses recursive, type-sensitive JSON equality: `true`, `1`, and `1.0`
are distinct options. Values outside the option set fail, even if their requested
return type is otherwise valid. Duplicate options and malformed request records
fail before a decision call. The selected value must also satisfy its annotation.

Any supplied decision `because` is discarded before rationale validation. It
cannot appear in `.because` or in a low-confidence error. Decision response
`citations` and `caveats` are also ignored; input provenance remains attached. Decision
metadata exposes `{kind,input,options,selected,scores}` on `.decision`. Collapse
preserves that evidence and its provenance. It is a trace of the request and
reported result, not a recovered explanation.

To demonstrate the subprocess transport without a model, save this adapter:

```python
import json
import sys
request = json.load(sys.stdin)
print(json.dumps({"value": request["options"][0], "confidence": 0.9}))
```

This deliberately chooses the first option. It tests transport, not judgment.

## Local Ollama adapters

`purrhaps.ollama` uses recursive JSON schemas for collection, nullable, and model
record returns. It validates the result locally as well as requesting a schema
from the server. Integer-spelled JSON numbers are normalized to floating point
only where the requested type is `Float`/`Double`, including nested positions.
A direct `ModelResponse` (or non-null `ModelResponse?`) request asks the model for text; the adapter fills
`finishReason` and `TokenUsage` from the server response, failing if those fields
are missing rather than inventing usage. These counts describe the whole oracle
request, including its structured answer. Nested `ModelResponse` records are
ordinary requested data, not automatically enriched transport telemetry.

`purrhaps.ollama_decision` uses Ollama's
[System One API](https://docs.ollama.com/api/systemone), not chat generation.
It requires a compatible server and an already installed decision model.
The adapter maps each JSON option to an opaque label, sends a `choice` question,
and maps the selected label back to the exact original JSON value. It retains the
server's `confidence` and ordered option probabilities separately. It supports
2–26 options, rejects unknown/missing labels and invalid scores, and never creates
a rationale. Generation temperature is not forwarded because this endpoint does
not support generation controls.

For a `DecisionInput`, the adapter sends its question, options and context to the
server. For `?!=`, it sends both operands and the difference question. The
subprocess receives the complete Purrhaps request, but this decision adapter does
not forward unrelated program source. Neither adapter reads credentials or
installs models. Transport and malformed-response tests use mocks; live model
quality, latency, and calibration have not been measured here.
