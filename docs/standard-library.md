# The growing standard litter

Purrhaps borrows useful shapes from Kotlin's [collections](https://kotlinlang.org/docs/collections-overview.html)
and [nullable types](https://kotlinlang.org/docs/null-safety.html). This is a native
Purrhaps library, not a Kotlin runtime or a promise of source compatibility.
It runs in the Python interpreter and generated Rust executables.

`sample(value)` is a runtime intrinsic for freezing one uncertain sample without
requiring confidence. Its result retains the original confidence and provenance;
it does not itself grant permission to collapse or branch. See
[sampling and escalation](escalation.md) for examples and resolution metadata.

## Types that travel across a model boundary

Annotations validate concrete values recursively, including model replies:

```kotlin
val labels: List<String> = ["support", "sales"]
val counts: Map<String, Int> = mapOf("cats" to 3, "boxes" to 2)
val fallback: String? = null
val pair: Pair<String, Int> = Pair("cats", 3)
```

`Boolean`, `Double`, and `Long` alias `Bool`, `Float`, and `Int`. `Json` accepts
finite JSON data. `List<T>`, `Map<String,T>`, `Pair<A,B>`, and nullable `T?` nest.
Bare `List`, `Map`, and `Pair` leave their element types unrestricted. Map keys
are strings. `Int` retains its existing Python/native range difference; these
aliases do not introduce Kotlin's distinct fixed-width numeric types.

Nullability is checked at runtime. There are no smart casts, safe-call `?.`,
Elvis `?:`, or static null guarantees. Purrhaps's `!!` still checks **confidence**;
it is not Kotlin's null assertion. A sufficiently confident nullable result can
collapse to `null`.

## Deterministic operations

All these operations preserve uncertainty and the minimum input confidence.
Constructing a collection with a lazy value does not call a model. Consuming it
forces its operands from left to right; repeated consumption can resample them.
A collapsed collection is a snapshot. Operations retain the collection's aggregate
provenance, so selecting one element does not discard evidence from its siblings.

| Area | Available operations |
| --- | --- |
| Construction | `[a, b]`, `listOf(...)`, `emptyList()`, `Pair(a,b)`, `a to b`, `mapOf(...)`, `emptyMap()` |
| Access | `xs[i]`, `map["key"]`, `.size`, string `.length`, pair `.first` / `.second`, map `.keys` / `.values` |
| Membership | `.isEmpty()`, `.isNotEmpty()`, `.contains(x)` (map keys) |
| Lists and strings | `.first()`, `.last()`, `.get(i)`, `.getOrNull(i)` |
| Lists | `.joinToString(separator)`, `.distinct()`, `.reversed()`, `.take(n)`, `.drop(n)`, `.sorted()`, `.sum()` |
| Strings | `.trim()`, `.lowercase()`, `.uppercase()`, `.startsWith(s)`, `.endsWith(s)`, `.split(separator)`, `.replace(old,new)`, `.substring(start[,end])` |
| Conversion | `.toString()`, string `.toIntOrNull()`, `.toDoubleOrNull()` |
| Numbers | `.abs()`, `.coerceIn(min,max)`, `abs(n)`, `minOf(...)`, `maxOf(...)` |

`joinToString()` defaults to `", "`. Lists and records display as compact JSON;
record keys and map key/value views are sorted by key. Duplicate map keys keep
the last supplied value. Collection `contains` and `distinct` use type-sensitive equality: `true`, `1`,
and `1.0` remain distinct. Ordinary `==`/`!=` retain their existing numeric
coercion behavior. Missing map keys return `null`. List/string indexing
requires a nonnegative integer and fails outside the bounds; `getOrNull` returns
`null` outside the bounds. Negative `take`/`drop` counts are errors.

String positions count Unicode code points, unlike Kotlin's UTF-16 indexing.
Case conversion follows each backend's Unicode tables; Unicode version differences
may affect unusual characters. Empty split delimiters are rejected. Sorting
accepts numbers or strings, not mixed strings and numbers. Numeric parsing rejects
non-finite results; `toIntOrNull()` uses signed 64-bit range in both backends.

## Model inputs and outputs

Constructors produce ordinary, validated JSON records. The same names can be used
as structural type annotations on parameters, returns, and oracle declarations.
They are not nominal classes and do not make a network call.

| Constructor / type | Fields and checks |
| --- | --- |
| `Message(role, content)` | `role`: system/user/assistant/tool; `content`: String |
| `Prompt(messages)` | Nonempty `List<Message>` |
| `DecisionInput(question, options, context)` | Nonempty question; at least two distinct finite JSON options; arbitrary finite JSON context |
| `TokenUsage(inputTokens, outputTokens)` | Nonnegative integer counts |
| `ModelResponse(text, finishReason, usage)` | String text and finish reason; validated `TokenUsage` |

```kotlin
val prompt: Prompt = Prompt(listOf(Message("user", "Explain this cat.")))
maybe reply: ModelResponse = prompt
// A configured oracle must return a record matching ModelResponse.
// Usage must be supplied by the oracle; the runtime does not invent token counts.

val request = DecisionInput("Which queue?", ["support", "sales"], "An invoice is broken.")
definitely queue: String = request
val chosen = queue!!
print(chosen, chosen.confidence)
print(chosen.decision.options, chosen.decision.scores)
```

The Ollama adapter populates direct `ModelResponse` and non-null `ModelResponse?`
usage and finish reason from its server envelope. Other adapters are responsible
for reporting those fields honestly; structural validation cannot establish their
truth. The default demo cannot author arbitrary structured model responses. Use
`examples/model_data.prr` for an offline record example and configure a real or
scripted generative oracle for a `maybe reply: ModelResponse` request.

Decision results remain the selected, typed value with attached evidence. They
are not wrapped in a special object requiring an unwrap at every branch. Inspect
`.decision` on a snapshot for `kind`, `input`, `options`, `selected`, and `scores`.
Scores may be `null`: a provider need not supply them. They are reported scores,
not an inferred explanation or a guarantee of calibrated probabilities.

## Where this library grows next

This release supplies the data handling needed for model requests and replies.
It does not reproduce Kotlin's entire standard library. Remaining major areas:

- First-class functions and lambdas, then `map`, `filter`, `fold`, grouping and
  sequences with explicit rules for lazy model calls.
- Real sets, mutable collections, ranges and iteration.
- User-defined records and sealed results, schema descriptions, tool calls,
  multimodal content, provider error/refusal/abstention outcomes and streaming.
- Regex, dates/durations, files, networking and async execution with explicit
  effect boundaries.
- Distinct fixed-width primitives, static type checking, and ergonomic nullable
  operations.

Decision abstention is currently an explicit option such as `"escalate"`, or a
low-confidence collapse failure. It is not inferred from scores. None of these
constructors certifies that a model made a good decision. Please pet responsibly.
