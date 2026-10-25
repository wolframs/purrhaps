# Native Rust backend

Purrhaps can lower a `.prr` program into a Cargo project. Python is needed to
transpile; the resulting executable contains its own Rust runtime and does not
invoke Python or reparse Purrhaps source. Generated functions directly call native
runtime operations. The full source remains embedded for oracle requests and
source-line diagnostics.

## Build and run

```sh
python3 -m purrhaps examples/hello.prr --emit-rust build/rust-hello
cargo build --manifest-path build/rust-hello/Cargo.toml --release
./build/rust-hello/target/release/purrhaps-program --stats
```

The output directory must not exist. This protects hand-edited generated projects;
use a new destination to regenerate. Generation parses and validates the source
before creating files. It performs no model calls and needs no Rust installation.
Compilation requires Cargo, rustc, and the `serde_json`, `toml`, and `url` crates. Add `--offline` to
Cargo commands when the dependencies are already cached. First-time dependency
fetching requires network access. Cargo creates `Cargo.lock`; retain it when you
want reproducible dependency resolution. This backend was verified with rustc
1.96.1 and cached serde_json 1.0.151.

The generated layout is:

```text
Cargo.toml
src/main.rs       # Generated expression, statement, and block functions
src/runtime.rs    # Native values, recipes, environments
src/project.rs    # TOML configuration and route validation
src/providers.rs  # Configured provider transports
.gitignore
```

Generated symbol names are compiler-assigned, so Purrhaps identifiers cannot
collide with Rust keywords or inject Rust code. Strings use safely delimited Rust
raw literals. Parsing JSON literals in generated code handles data only; there is
no embedded syntax tree interpreter.

## Project configuration

Native programs support the same [purrhaps.toml manifest](projects.md), named
`with model` blocks, provider profiles and configuration validation as Python.
They also support [decision-first escalation](escalation.md), including `sample`,
resolution histories, and seeded random terminal choices. Random terminal draws
match Python for the same seed and sequence of terminal choices.
Transpiling a configured project copies its manifest into the emitted Cargo
project. The executable reads it at runtime; pass `--project FILE_OR_DIR` from
another directory or `--no-project` for standalone behavior. `--model ROUTE` and
`--decision-model ROUTE` select default routes. Build-time CLI overrides are not
baked into the copied manifest; pass execution overrides to the binary.

Configured HTTP providers use `curl` on PATH with its user configuration disabled;
other native routes do not need curl. Subscription harness routes need their
respective executable. The Purrhaps native runtime does not invoke Python for any
built-in transport. A custom Python command adapter naturally requires Python.

## Native runtime controls

Arguments belong to the compiled program, not the transpilation step:

```sh
./build/rust-hello/target/release/purrhaps-program \
  --seed 7 --max-calls 30 --timeout 150 --stats
```

Supported options are `--seed`, `--max-calls`, `--timeout`, `--stats`, `--oracle`,
`--oracle-arg`, `--decision-oracle`, `--decision-oracle-arg`, and `--help`. The call budget defaults to 1,000; the per-call timeout
defaults to 30 seconds. Native timeouts must be finite and in (0, 86400] seconds.
The seed accepts signed decimal integers, reduced to their low 64 bits for
native random streams. Invalid arguments exit with status 2;
language/runtime failures exit with status 1 and a source-line diagnostic.

Native subprocess oracles use the same [generative and decision JSON protocols](oracles.md) as Python.
Specify an executable and each argument separately; no shell is involved:

```sh
./build/rust-hello/target/release/purrhaps-program \
  --oracle python3 \
  --oracle-arg -m --oracle-arg purrhaps.ollama \
  --oracle-arg --model --oracle-arg YOUR_INSTALLED_MODEL \
  --timeout 150 --stats
```

This particular optional adapter needs Python and the Purrhaps package available
to that Python interpreter. The compiled language runtime itself does not.
An oracle can instead be any executable that implements the protocol. Child
stderr is discarded to avoid exposing credentials in diagnostics. The process is
killed and reaped on timeout; calls are never retried automatically. As in Python,
full program source and sampled inputs are sent to the configured oracle.

To configure the separate decision route, add:

```sh
--decision-oracle python3 \
--decision-oracle-arg -m --decision-oracle-arg purrhaps.ollama_decision \
--decision-oracle-arg --model --decision-oracle-arg YOUR_DECISION_MODEL
```

Both routes consume the same `--max-calls` budget. Leaving either route
unconfigured uses its offline demo; configuring one does not configure the other.
The [standard library and model records](standard-library.md) run natively too.

## Semantics and deliberate differences

Both runtimes support all currently parsed language constructs. Recipes preserve
uncertainty through arithmetic, fields, arguments, and return values. Each reading
scope has its own `perhaps` cache. `!!` and branches require confidence at least
0.8. Native results validate oracle confidence and typed values before use; generative
results additionally validate citations and caveats. Decision values must match
the supplied options and cannot expose `.because`.
Temperature is restored when a block finishes, fails, or returns from a function.
Name lookup is lexical; ownership follows the latest attempted reader.

The native runtime stores scopes in an arena for the lifetime of one program.
Recipes hold scope indices and reference-counted values. It deliberately does not
attempt garbage collection or expose Rust borrowing rules as Purrhaps semantics.

Known backend differences:

- Native `Int` is signed 64-bit. Oversized integer literals fail during generation;
  overflow in arithmetic produces a language error. Python integers are unbounded.
- Both backends reject non-finite arithmetic results. Float formatting and mixed
  integer/float comparisons at the precision boundary can still differ.
- The demo oracle uses a native seeded generator, so nonzero-temperature samples
  need not match Python's PRNG. The same native seed and program repeat the same
  sequence. Temperature zero removes demo integer variation in both backends.
- Collections and records now print as compact JSON in both runtimes, with sorted
  record keys. Float formatting within that JSON can differ at extreme magnitudes.
  Unicode case conversion can differ when backend Unicode tables differ.
- Recursion is limited by a native evaluation-depth guard (128 nested evaluation
  or sampling operations), rather than Python's interpreter recursion limit.
- Type/protocol failures share error categories; diagnostic wording can differ.
  The Rust backend uses serde_json's numeric representation and rejects `Int`
  oracle answers that do not fit signed 64-bit.

The demo oracle remains a toy. No native or Python runtime can prove that a model's
rationale is true or its self-reported confidence is calibrated.

## Verification

```sh
python3 -m unittest discover -s tests -v
```

When Cargo is available, the suite generates projects, builds them offline, and
runs the binaries. Tests compare stdout, oracle call counts, and failure categories
against Python for arithmetic, lazy reads, cache scope, ownership, functions,
branching, metadata, collapse, and scripted external oracles. All shipped examples
are compiled and executed. Additional tests cover overflow, recursion, timeouts,
call budgets, CLI generation, and execution with an empty `PATH`.

Cargo tests skip if Cargo is absent. If Cargo exists but its dependency cache is
empty, integration tests fail with Cargo's missing-dependency diagnostic; fetch
with a generated project's `cargo fetch` first. No live model is required or
claimed by these tests.
