# Purrhaps language contract

## Grammar and evaluation

Statements are separated by a newline or semicolon. Blocks use braces; comments
start with `//`. Strings are double-quoted with JSON escapes and support Unicode.
Expressions and function signatures currently occupy a single logical line;
blocks can contain multiple lines. Primitive types are `Int`, `Float`, `String`,
`Bool`, `Unit`, `Any`, and `Json`; `Long`, `Double`, and `Boolean` are aliases.
Recursive `List<T>`, `Map<String,T>`, `Pair<A,B>`, and nullable `T?` annotations and
validated model records are described in the [library reference](standard-library.md).
An omitted type means no type restriction. Booleans are not integers. Types are
checked at runtime when a concrete value is available. `Human` requires exactly
one String field, `lastMessage`. Model record annotations also validate their
complete nested structure.

Precedence, tightest first: calls/fields/indexing/`!!`; unary `-`/`!`; `* / %`; `+ -`;
`< <= > >= to`; `== != ~= ?!=`; `&&`; `||`. Binary operators associate left.
Arithmetic accepts numbers; `+` also concatenates two strings. `/` uses floating
point division. There is no implicit string/number coercion in arithmetic.
Both Boolean operands are evaluated; `&&` and `||` do not short-circuit.

Name lookup is lexical. Declarations are immutable and may shadow names in outer
scopes. Redeclaration in the same scope is an error. Function declarations capture
their lexical environment. Calls create a fresh scope and check arity. `return`
exits a function; falling off its end returns `null`. Functions are not first-class.
Intrinsic and library names are reserved call targets. Member calls dispatch to
the deterministic library; arbitrary methods and first-class functions are absent.

## Values and recipes

A definite value holds data, confidence, source citations, and origin metadata.
Generative samples retain rationales; decision samples retain choice evidence.
Literals have confidence 1 and their source line as provenance, but no `.because`. An uncertain value holds a
recipe. Constructing a recipe does not call an oracle. Forcing it returns a
concrete sample. This internal materialization does not change the recipe into a
definite source-level value; `!!` creates an accepted snapshot, while `sample(x)`
creates a snapshot without requiring confidence.

A `maybe` initializer is evaluated in its declaration environment on every forced
read. Its resulting input is sent to the oracle, with its type annotation and
source location. If that input was itself uncertain, it is sampled first; the
outer answer cannot increase its confidence. `val` preserves a recipe's
uncertainty. Arithmetic and ordinary field access defer computation if any
operand is uncertain. Collections and library calls use the same whole-value
lifting: even `xs.size` or `xs[1]` samples every lazy constructor operand. Operands
are sampled left to right. For an expression with
multiple explicit collapses, those collapses execute while constructing the
expression, before remaining lazy operands are forced.

A `definitely` initializer is also lazy and resampled per forced read. A
`DecisionInput` invokes the decision route with a finite option set; an already
decision-backed expression is reused, with its result annotation checked, without
an extra decision call. A request containing prior decision evidence in its
context still requires a fresh selection. Other initializers fail. `?!=` asks
whether its two operands are meaningfully different, with options `[false,true]`.

`maybe x: T ?= DecisionInput(...)` creates a decision-first escalation recipe.
Only `maybe` supports this assignment form. It tries a generative route if the
decision's effective confidence is below the project threshold, then applies
the configured error or random terminal policy. See [escalation](escalation.md)
for routing, evidence and the distinction between acceptance and sampling.

The `perhaps` cache belongs to the declaration but is keyed by **reading scope
identity**. A function invocation, chosen branch, or temperature block creates a
fresh scope; the module is also a scope. Cache hits still transfer ownership.
Failure does not populate the cache. Returning to an earlier scope reuses that
scope's sample. Temperature changes do not invalidate existing cached samples.
An uncertain value returned from a function is consumed in the caller's reading
scope; its lexical bindings remain captured.

`with model("route") { ... }` selects a configured route for its lane during
forced reads in the block; it requires a [project manifest](projects.md). It
creates a reading scope and restores the previous route after normal completion,
errors, and function returns. Lazy values escaping the block use the route active
when later sampled; collapse inside the block to retain that result. The other
lane is unaffected. Profile temperature defaults yield to an explicit temperature
block, except on transports that cannot honor temperature (which reject it).

An initializer is lazy, so `maybe x = x` is recursive. It eventually fails with a
language recursion diagnostic. This interpreter does not attempt cycle inference.

## Collapse and provenance

`!!` requires confidence at least 0.8 by default, except for an explicitly
authorized random terminal result from escalation. It preserves the sample's score
and provenance after collapse. The Python embedding API can set another threshold.
Deterministic calculations inherit the minimum input confidence. Nested oracle
reads and oracle equivalence use the minimum of input and answer confidence.
This is a conservative propagation rule, not a statistical independence model.
Random terminal results keep confidence zero and carry a separate permission
to pass the gate. Combining them with an unrelated low-confidence operand does
not give that operand permission. Historical attempts do not inherit this bypass.

`.because`, `.confidence`, `.citations`, `.decision`, and `.resolution` each inspect a fresh sample
unless the receiver is definite or cached. Their results are definite metadata.
`.because` requires generative origin without decision origin; plain deterministic
values and anything derived from decision evidence raise `ExplainabilityError`.
The restriction follows samples through copies, operations and function boundaries,
not variable spelling. `.decision` retains decision provenance so using its fields
as fresh model input cannot manufacture an explanation. Other metadata describes
the sample rather than claiming to explain its value. To get
coherent fields for one sample, use `val snapshot = sample(x)` (no confidence gate)
or `val snapshot = x!!` (accepted snapshot) and inspect the snapshot,
or inspect a `perhaps` repeatedly within one scope. Arithmetic combines input
rationales and deduplicates citations; equivalence uses its judgment rationale
and retains input citations. Derived values carrying several decisions currently
retain one trace, not a complete evidence graph. Inspect separate collapsed
decisions when their individual option sets or scores matter.
Escalation's `.resolution` records the attempts and final selection strategy.
The history is decision-tainted even when the final LLM answer has its own
generative explanation. Reading historical decision data through this record
cannot manufacture `.because`.

`if` is a deliberate implicit-collapse boundary to support the original
`if (mood ~= "sunny")` syntax. A low-confidence condition throws; neither branch
runs. A branch must receive a Boolean, never a truthy string or number. This does
not implement control-flow information tracking: a literal returned from a chosen
branch is still a definite literal.

Ownership is the last scope to force a recipe, including metadata reads. Reads
of derived values also transfer ownership of the source recipes they consume.
`.owner` itself is non-forcing and returns a scope label (`module`, `visit#2`,
etc.), `null` before the first read, or `certain` for definite values. Failed
attempted reads still transfer ownership. Ownership does not revoke access.

## Oracle checks

Oracle results must be finite JSON data and have a finite numeric confidence in
[0, 1]. Generative results require a nonempty rationale, integer citations, and
string caveats. Decision results need no rationale; any supplied rationale is discarded
before rationale-specific checks or diagnostics, along with provider-supplied
citations and caveats; input provenance is retained. A decision must strictly match a
supplied option, recursively distinguishing Booleans, integers and floats. Optional
scores must align with options and be finite numbers in [0,1]; normalization is
not required. Provider scores and confidence have no calibration guarantee. Typed results
are checked without coercion by the interpreter. Citation validation checks that
each line exists, not that it supports the conclusion. Structured citations and
simple prose mentions of `line N` are checked.

Caveats are counted as the larger of the structured caveat count and the count of
these case-insensitive whole words in the rationale: `perhaps`, `maybe`,
`possibly`, `probably`, `however`, `although`, `uncertain`. More than three fails.
This is intentionally a specified lexical heuristic, not general natural-language
understanding. Derived rationales can concatenate several already validated
answers; the hedge limit applies to each oracle answer, not that concatenation.

An oracle-call budget bounds model calls, not arbitrary CPU usage. The interpreter
is not a process sandbox. An external oracle command is operator-selected code.
Use a fresh `Interpreter` instance for each source program; cross-program state
and interactive incremental execution are not supported.
