# Purrhaps 🐈‍⬛

**A programming language in which uncertainty is a first-class liability.**

Files end in `.prr`. Values may change their minds. Equality requires an opinion.
The borrow checker has been replaced by the ecology.

Purrhaps includes a working tree-walking interpreter and a **native Rust
transpiler**, sharing a handwritten parser, lazy uncertain values, provenance,
and pluggable oracles. The Python tools run on Python 3.11+ with **no runtime
dependencies**. Generated Rust projects use Cargo with `serde_json`, `toml`, and `url`.

## First contact

This is an experimental language under active development. Start with the offline
demo; live models are optional and configured through a project manifest.

```sh
git clone https://github.com/wolframs/purrhaps.git
cd purrhaps
python3 -m purrhaps examples/hello.prr --stats
python3 -m purrhaps examples/ecology.prr
python3 -m unittest discover -s tests -v
```

The default oracle is an **offline demo, not an LLM**. It has a small synonym table,
a fondness for 42, and seeded variation. No network requests or credentials are
needed. The language's semantics are real; this oracle's intelligence is cardboard.

```kotlin
print(hi())

maybe n: Int = "roughly the answer to everything"
print(n, n)                 // independent samples; agreement is not guaranteed
val definite = n!!         // confidence below 0.8 throws UncertaintyException
print(definite.because)     // explanation for that exact collapsed sample

perhaps answer: Int = "the meaning of life"
print(answer, answer)      // one cached sample in this scope

val contagious = n + 1     // remains uncertain, and lazy
print(contagious!!)

fun greet(user: Human) {
    maybe mood = user.lastMessage
    if (mood ~= "sunny") print("☀️")
}
greet(Human("cheerful"))

with temperature(0.0) {
    print(n!!, n!!)         // stable with the demo; model reproducibility varies
}
```

Optional installation into your own virtual environment:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/purrhaps examples/hello.prr
```

## Enter Rust-land

```sh
python3 -m purrhaps examples/hello.prr --emit-rust build/rust-hello
cargo run --manifest-path build/rust-hello/Cargo.toml -- --stats
```

This emits a new Cargo project containing generated Rust functions and a native
runtime. The compiled executable needs **no Python installation**. All current
language constructs are supported, including lazy oracle reads, scope caches,
functions, provenance validation, and the ecology. Existing output directories
are never overwritten; choose a fresh destination for each transpilation.

Native `Int` is checked signed 64-bit; the Python interpreter uses arbitrary-sized
integers. Nonzero-temperature demo sequences differ between backends. Objects
and arrays print as compact JSON in both backends. See the [Rust backend guide](dogs/rust.md)
for builds, oracle configuration, verification, and the remaining differences.

## The laws of uncertain cats

| Construct | What happens |
| --- | --- |
| `maybe x: Int = "a hint"` | Saves an initializer. Each forced read evaluates it and asks the oracle. |
| `perhaps x = "a hint"` | Same, but caches one successful answer per reading scope. |
| `definitely x: String = DecisionInput(...)` | Lazily asks a separate decision oracle to select from supplied options. |
| `maybe x: String ?= DecisionInput(...)` | Tries a decision, escalates to an LLM on low confidence, then applies the project's terminal policy. |
| `sample(x)` | Freezes one sample with its confidence and evidence, without requiring acceptance. |
| `val x = expression` | Immutable binding. Does **not** remove uncertainty. |
| `x + 1` | Deferred if `x` is uncertain; its confidence cannot improve through arithmetic. |
| `x!!` | Returns a snapshot after its confidence gate; an explicitly configured random terminal can authorize progress without confidence. |
| `a ~= b` | Asks the generative oracle for an uncertain Boolean with rationale. No transitivity is assumed or cached. |
| `a ?!= b` | Asks the decision oracle whether the inputs are meaningfully different. |
| `a == b`, `a != b` | Reject lazy operands. Sample or collapse first to compare snapshots; the comparison retains their uncertainty. |
| `x.because` | Inspect a generative sample’s rationale. Plain and decision-derived values raise `ExplainabilityError`. |
| `x.confidence`, `x.citations`, `x.decision` | Inspect one sample; `.decision` contains decision evidence, never a rationale. Separate reads may disagree. |
| `x.resolution` | Inspect escalation history and final strategy; historical decision data retains its provenance. |
| `x.owner` | Last reading scope, or `null` before a read. Inspection does not sample. |
| `if (condition)` | Collapses an uncertain Boolean at the same confidence threshold. |
| `with temperature(t) { ... }` | Dynamically sets temperature for calls in the block; restores it even on failure. |
| `hi()` | Always returns the definite string `hi! 🐈‍⬛`. No oracle involved. |

A **forced read** is consumption by `print`, `sample`, `!!`, metadata inspection, a branch,
or an oracle request that needs the value. Merely constructing a deferred
expression or passing it to a function does not sample. `val x = n + n` samples
`n` twice whenever `x` is consumed. `perhaps` makes both reads share an answer
inside the same scope.

See [decision-first escalation](dogs/escalation.md) for `?=`, blind/informed
fallback, and the explicit random terminal policy. The
[tribunal field notes](dogs/field-notes.md) record the real program
that exposed the need for stable low-confidence samples and escalation.

An oracle claiming a nonexistent source line causes `ConfabulationError`. More
than three structured caveats or recognized hedge words causes
`HedgeOverflowError`. These are mechanical checks, not proof that a rationale is
true. Confidence is an oracle's claim, not a calibrated probability.

## Two kinds of opinion

```kotlin
maybe mood: String = "cheerful"
definitely upset: Bool = mood ?!= "sunny"
val snapshot = upset!!
print(snapshot, snapshot.confidence, snapshot.decision.options)
// snapshot.because raises ExplainabilityError. Confidence is not an explanation.

val request = DecisionInput("Which animal?", ["cat", "dog"], "cat")
definitely animal: String = request
print(animal!!)
```

`definitely` names the decision route; it does not grant certainty. The initializer
must be a `DecisionInput` or an existing decision-backed expression. Wrapping
`?!=` makes no extra decision call. Both routes share the call budget and preserve
input uncertainty. A decision's optional scores stay attached to its snapshot;
a later generative transformation cannot manufacture `.because` for that evidence.
The runtime does not track provenance through control-flow choices.

## A useful standard litter

```kotlin
val labels: List<String> = ["cat", "box", "cat"]
print(labels.distinct().sorted().joinToString(" / "))
val counts: Map<String, Int> = mapOf("cats" to 3, "boxes" to 2)
print(counts["cats"], counts.keys)
val missing: Int? = "many cats".toIntOrNull()
val prompt: Prompt = Prompt([Message("user", "Hello!")])
val reply: ModelResponse = ModelResponse("Meow!", "stop", TokenUsage(12, 3))
```

Collections, string and numeric operations are deterministic and preserve lazy
uncertainty. Recursive annotations validate model inputs and returned data.
`Message`, `Prompt`, `DecisionInput`, `TokenUsage`, and `ModelResponse` are validated
JSON records. The example reply above is constructed data, not a live model result.

See the [standard-library reference](dogs/standard-library.md) for the complete
surface, Kotlin differences, and remaining work. This is a practical Kotlin-style
core, not the entire Kotlin library. Try `examples/standard_library.prr`,
`examples/model_data.prr`, and `examples/decisions.prr`.

## Configure a project

A `purrhaps.toml` manifest separates provider connections, model profiles, named
routes, and generation/decision defaults. Configure OpenRouter or another
OpenAI-compatible endpoint, local Ollama, an existing Codex/Claude login, or your
own subprocess adapter. API credentials are referenced by environment name.

```sh
python3 -m purrhaps --project examples/projects/routing --check
python3 -m purrhaps --project examples/projects/routing --stats
```

```kotlin
with model("careful") {
    maybe answer: Int = "the answer to everything"
    print(answer!!)
}
```

Routing happens when a lazy value is sampled. Named blocks change only their
model's lane and restore the previous route afterward. Effort, timeout, model
options and endpoint live in the manifest; unsupported controls fail explicitly.
See [project configuration](dogs/projects.md) for the schema, provider examples,
CLI precedence, subscription harness boundaries and native deployment.

## Give it a real oracle

An included adapter speaks to an existing local Ollama server. Choose a model
already installed on that server:

```sh
python3 -m purrhaps examples/hello.prr \
  --oracle-command 'python3 -m purrhaps.ollama --model YOUR_INSTALLED_MODEL' \
  --timeout 150 --max-calls 30 --stats
```

This sends program source and sampled inputs to that server. The adapter uses
Ollama's [chat API](https://docs.ollama.com/api/chat) and
[structured output format](https://docs.ollama.com/capabilities/structured-outputs).
It neither installs models nor reads credentials. Adapter transport is tested with
mock responses; a live model is not required by the test suite.

Decision models have their own adapter and command:

```sh
python3 -m purrhaps examples/decisions.prr \
  --oracle-command 'python3 -m purrhaps.ollama --model YOUR_CHAT_MODEL' \
  --decision-oracle-command 'python3 -m purrhaps.ollama_decision --model YOUR_DECISION_MODEL' \
  --timeout 150 --stats
```

The decision adapter uses Ollama's [System One endpoint](https://docs.ollama.com/api/systemone)
with an already installed compatible model. Both adapters are covered by mocked
transport tests; no live model result is claimed. The default for each route is
an independent offline demo oracle.

Other models can connect through `--oracle-command` or `--decision-oracle-command`: one JSON request on stdin,
one JSON answer on stdout, one process per call. Commands run without a shell.
See [the protocol](dogs/oracles.md) for the exact contract and a minimal example.
There are no automatic retries, so a failed call does not silently multiply cost.

## Useful controls

```sh
python3 -m purrhaps examples/hello.prr --check   # syntax only, zero oracle calls
python3 -m purrhaps examples/hello.prr --seed 7  # reproducible demo PRNG sequence
python3 -m purrhaps examples/uncertainty.prr    # deliberate collapse failure
```

`--stats` reports oracle backends and the shared call count on stderr. `--max-calls` defaults
to 1,000; subprocess calls have a configurable timeout. Diagnostics include a
source line and exit with status 1. `--check` only parses; it does not type-check.

## Boundaries of this first litter

Supported: immutable declarations, optional runtime type annotations, arithmetic,
comparisons, eager Boolean operators, functions, `return`, lexical scopes,
`if`/`else`, temperature blocks, field/index access, collection literals, recursive
runtime types, library methods, project manifests, scoped model routes, and provenance.

Not implemented: loops, assignment, imports, classes, lambdas,
exceptions in source code, static type inference, async execution, or a REPL.
The self-introspection joke is represented by low-confidence demo answers, not
an empirical 74% distribution. Ownership is observable last-reader metadata;
there is no borrow enforcement or memory-safety claim.

Read the [language specification](dogs/language.md) for evaluation order, caching,
and confidence propagation. Tests use scripted oracles to verify behavior
independently of model output. No live LLM result is claimed by the offline demos.

*Certainty is a side effect. Please pet responsibly.*
