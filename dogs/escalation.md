# Decision-first escalation

`maybe … ?=` tries a bounded decision first, then asks a generative model if the
decision's confidence is too low. It changes how a sample is obtained; the
variable still behaves like a `maybe`.

```kotlin
maybe queue: String ?= DecisionInput("Which queue should handle this?", ["support", "sales"], "The customer cannot sign in.")
val answer = sample(queue)
print(answer, answer.confidence, answer.resolution.strategy)
```

Only `maybe` accepts `?=`. Its initializer must produce a valid `DecisionInput`,
including through a function or alias. Each option must match the declaration's
type before either pipeline model is called. An already completed decision is
not reused: a `DecisionInput` value always starts a new request. Other values,
including a Boolean from `?!=`, are not escalation requests.

Put whitespace between a type annotation and the escalation operator:
`maybe x: Int ?= ...`. Existing compact nullable declarations such as
`val x:Int?=null` keep their meaning (`Int?` followed by ordinary `=`).
For nullable escalation, write `maybe x: Int? ?= ...`.

Like ordinary `maybe`, each forced read starts again. The initializer is sampled
once for that pipeline. No call occurs merely by declaring the variable.

## Freeze a sample without requiring confidence

`sample(value)` forces exactly once and returns a stable snapshot, preserving
its confidence and provenance. It works for ordinary `maybe`, `perhaps`, and
`definitely` values too. Sampling an existing snapshot makes no model call.

```kotlin
definitely verdict: String = DecisionInput("Which queue?", ["support", "sales"], "A billing question")
val heard = sample(verdict)
print(heard, heard.confidence, heard.decision.scores)
if (heard.confidence >= 0.8) {
    print(heard!!)
} else {
    print("The recorded answer needs another policy.")
}
```

`.confidence` describes the snapshot and is definite metadata. The selected
value remains uncertain: `sample` alone does not authorize `!!` or branching on
its contents. This avoids repeated reads accidentally asking for new decisions.

## Project policy

An optional section in `purrhaps.toml` configures escalation:

```toml
[escalation]
decision = "triage"
generative = "careful"
threshold = 0.8
exposure = "blind"
on_low_confidence = "random"
```

`decision` and `generative` name routes of the corresponding kind. When omitted,
the pipeline uses the routes active at the time it is forced, including scoped
`with model` routes and command-line overrides. Explicit escalation routes take
precedence for that pipeline; surrounding routes are restored afterward.

`threshold` defaults to 0.8 and accepts finite numbers in [0,1]. For either
attempt, effective confidence is the minimum of the answer's confidence and the
initializer's confidence. Option scores do not substitute for confidence.
An effective confidence equal to the threshold is accepted.

`exposure` defaults to `blind`: the fallback receives the original question,
options and context, without this pipeline's first runtime response. `informed`
adds that decision's selected value, reported confidence and scores. Full source
and the original context are still sent in both modes; blind mode does not
redact earlier material already present there. Informed mode may introduce
anchoring. Neither mode lets the LLM explain the earlier model's reasoning.

`on_low_confidence` defaults to `error`. If both valid answers fall below the
threshold, the read raises `UncertaintyException`. Set it to `random` to select
uniformly from the original options instead. This repository explicitly enables
random fallback; other projects must opt in independently.

Transport errors, malformed answers, invalid citations, unsupported controls and
budget exhaustion remain errors. They do not trigger fallback or randomness.
The pipeline spends one decision call, plus one generation call if needed;
initializer sampling may spend additional calls. All use the shared budget.
Random selection makes no model call. There are no implicit retries.

`project.no_humans_allowed = true` prohibits a future human escalation step.
This release has no human transport or notification mechanism; `human` is not
an accepted terminal policy, regardless of the flag.

## Resolution and evidence

The final value retains a `.resolution` record:

| Field | Meaning |
| --- | --- |
| `strategy` | `decision`, `generative`, or `random` |
| `exposure` | Configured `blind` or `informed` mode |
| `forced` | True only for the random terminal selection |
| `threshold` | Pipeline acceptance threshold |
| `selectionProbability` | `1/K` for uniform selection from K options; otherwise null |
| `decision` | Original decision evidence: input, options, selected value and scores |
| `attempts` | The one or two model attempts, in order |

Each attempt contains `kind`, `route` (null for an unconfigured external/demo
lane), `value`, `reportedConfidence`, `effectiveConfidence`, and `scores` (null
for generation). Use `reportedConfidence` for the attempt's raw number;
`.confidence` is reserved for the language's value metadata.

```kotlin
maybe choice: String ?= DecisionInput("Choose a nap spot", ["box", "sofa"], null)
val frozen = sample(choice)
print(frozen.resolution.attempts[0].reportedConfidence)
print(frozen.resolution.strategy, frozen.resolution.forced)
```

A successful fallback can provide `.because` for its own generated choice.
The rejected decision is recorded as history, rather than being treated as the
source of the fallback's explanation. Original decision-tainted context still
prevents `.because`, as it does for ordinary generation.

The history itself is decision-tainted. Feeding
`frozen.resolution.attempts[0].value` or scores into generation cannot turn the
rejected decision into something with a generative explanation. Historical
attempts also do not inherit the random selection's permission to bypass the
confidence gate. Decision and random final values reject `.because`.

Deterministic operations retain the first available resolution trace, as with
existing decision evidence. This is whole-value provenance, not a complete
graph of all decisions; control-flow provenance remains untracked.

## Randomness authorizes progress, not confidence

A random terminal result has `.confidence == 0.0`: it asserts no evidential
confidence. Its `selectionProbability` describes the draw, not whether the chosen
answer is correct. It carries explicit permission to pass `!!` and Boolean
branching. That permission propagates through deterministic operations, but
does not authorize any unrelated uncertain operand combined with it.

Normal accepted model answers still use the usual collapse threshold, 0.8 by
default. Setting the pipeline threshold to 0.5 could accept a 0.6 answer that
`!!` rejects. Only the explicit random terminal has the bypass described above.

Random fallback uses a separate seeded SplitMix64 stream and rejection sampling
for uniform option indices. The same seed and sequence of random terminals
produce the same results in Python and native Rust, independently of demo-oracle
draws. This is reproducibility machinery, not cryptographic randomness.

Run the [offline example](../examples/projects/escalation/main.prr) with:

```sh
python3 -m purrhaps --project examples/projects/escalation --seed 7 --stats
```

Its deliberately high threshold rejects both demo answers, exercising the random
terminal without credentials, network access or a local model.
