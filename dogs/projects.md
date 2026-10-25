# Purrhaps projects and model routing

A project has a versioned **`purrhaps.toml`**. It describes provider connections,
model configurations, named routes, and the defaults for generation and decisions.
Source code names a job such as `writing`, `careful`, or `triage`; the manifest
chooses the vendor, model, effort, and endpoint behind it.

```toml
manifest_version = 1

[project]
name = "my-cat"
entry = "main.prr"

[runtime]
max_calls = 100
timeout = 60

[providers.router]
transport = "openai-chat"
base_url = "https://openrouter.ai/api/v1"
api_key_env = "OPENROUTER_API_KEY"
response_format = "json_schema"

[providers.local_decisions]
transport = "ollama-decision"
base_url = "http://127.0.0.1:11434"

[models.writer]
provider = "router"
kind = "generative"
model = "YOUR_PROVIDER_MODEL_ID"
[models.writer.options]
reasoning = { effort = "high" }
max_tokens = 2048
provider = { require_parameters = true }

[models.classifier]
provider = "local_decisions"
kind = "decision"
model = "YOUR_INSTALLED_DECISION_MODEL"

[routes]
writing = "writer"
triage = "classifier"

[defaults]
generative = "writing"
decision = "triage"
```

Model IDs above are placeholders, not downloaded or selected automatically.
`reasoning` and provider preferences are OpenRouter request options; other services
may use different keys. Purrhaps forwards supported configuration exactly and does
not translate effort levels or silently substitute another model. Provider-side
support and any harness-side normalization remain the provider's responsibility.

The runtime checks unknown keys, versions, route references, lane compatibility,
finite numeric settings, and reserved option names. A typo fails rather than
falling back to the demo. Configure `transport = "demo"` explicitly for offline
work, as in [the runnable routing project](../examples/projects/routing/purrhaps.toml).

## Human escalation policy

```toml
[project]
name = "my-cat"
entry = "main.prr"
no_humans_allowed = true
```

`project.no_humans_allowed` records a project-wide prohibition on human
escalation. It accepts only a TOML boolean; omission is equivalent to `false`
(no prohibition), which does not itself request human involvement.

The flag is validated and preserved in Python and emitted native projects.
Human escalation is not implemented, so this flag currently changes no execution
behavior. Decision-to-LLM escalation is available through `maybe … ?=` and an
optional `[escalation]` section; see [escalation and random fallback](escalation.md).
Its terminal policy can raise an error or explicitly select a random option.
Any future human step must honor this prohibition. The flag itself does not
lower confidence thresholds or force acceptance of a model's answer.
This is a Purrhaps policy, not a sandbox for external commands or harnesses.

## Run, select, and check

```sh
# Discover purrhaps.toml from this directory upward and use project.entry.
python3 -m purrhaps

# Discover upward from the explicit source file's directory.
python3 -m purrhaps path/to/main.prr

# Select a specific manifest or directory, even when running elsewhere.
python3 -m purrhaps --project examples/projects/routing
python3 -m purrhaps --project examples/projects/routing --check

# Override a default route for this invocation.
python3 -m purrhaps --project examples/projects/routing --model careful

# Preserve the original standalone-script behavior.
python3 -m purrhaps examples/hello.prr --no-project
```

`--model` selects a generative route; `--decision-model` selects a decision route.
Each conflicts with its corresponding legacy `--oracle-command` override.
A command override replaces only that default lane; an explicit named block can
still select a manifest route. `--project` and `--no-project` are mutually exclusive.
With no manifest, standalone programs retain their original offline defaults.

Timeout precedence is CLI `--timeout`, then model `timeout`, then runtime
`timeout`, then 30 seconds. Timeouts must be positive, finite, and at most 86400
seconds. Call-budget precedence is CLI `--max-calls`, runtime `max_calls`, then
1000. Both lanes and all routes share that budget. A harness invocation consumes
one Purrhaps call even if the harness internally uses several model turns.

`--check` validates the selected manifest and parses source. It does not look up
keys, launch an adapter or model, test connectivity, or establish model capability.
Emission likewise does not authenticate or call providers. There is no dotenv
loader, secret interpolation, credential storage, or implicit retry/fallback.
The explicit `?=` pipeline escalates only on valid low-confidence answers.
Inject credentials into the process using your existing secret manager; manifests
contain only environment-variable **names**. Missing keys fail on the first
actual HTTP call. Subscription harnesses use their own existing login state.

## Select a model for part of a program

```kotlin
maybe summary: String = "Explain this cat"
with model("writing") {
    val frozen = summary!!
    print(frozen.because)
    with model("triage") {
        definitely different: Bool = frozen ?!= "a cardboard box"
        print(different!!)
    }
}
```

A route changes only its declared lane; a decision block does not change the
current generative route. Blocks nest and restore the previous route after normal
completion, exceptions, and `return`. An unknown route fails before calling a model.
The route expression must be a definite String.

Routing is **dynamic at the forced read**. Declaring a `maybe` or returning a lazy
recipe from inside a route block does not pin it to that model. Collapse inside
the block to retain that result. `perhaps` keeps its existing per-reading-scope
cache; a model block creates a new reading scope. Named routes are configuration,
not a change to confidence thresholds or provenance rules.

A generative model's optional `temperature` provides a default. `with temperature`
overrides it while active. OpenAI-compatible HTTP requests omit the temperature
parameter entirely unless a profile or explicit block sets it. Codex, Claude,
and Ollama decision transports reject explicit temperature settings before
invocation; those adapters cannot honor that control.

## Provider transports

| Transport | Configuration | Behavior |
| --- | --- | --- |
| `demo` | No connection settings | Existing offline toy; either lane |
| `openai-chat` | `base_url`, optional `api_key_env`, optional `response_format` | `/chat/completions`; either lane, with typed structured output |
| `ollama` | `base_url`, optional `api_key_env` | `/api/chat`; generative lane |
| `ollama-decision` | `base_url`, optional `api_key_env` | `/v1/systemone`; decision lane, 2–26 choices |
| `command` | `command = ["executable", "arg", ...]` | Purrhaps JSON stdin/stdout protocol; either lane |
| `codex` | Optional command prefix, default `["codex"]` | Noninteractive `codex exec`; generative lane |
| `claude` | Optional command prefix, default `["claude"]` | Noninteractive `claude -p`; generative lane |

HTTP base URLs cannot contain userinfo, query strings, or fragments. The adapter
adds the endpoint suffix shown above. Authorization uses `api_key_env`, never a
literal token or header in TOML. HTTP redirects are rejected. Transport diagnostics
omit response bodies and child logs, which can contain sensitive data.

An inference reseller works through `openai-chat` **if its API implements that
protocol**, with its documented base URL and model IDs. This includes configurable
reseller endpoints rather than a hard-coded vendor list. The supplied Surplus
Intelligence website did not establish an API base URL during implementation;
none is guessed here. Use `command` for any incompatible protocol.

`openai-chat` with a decision profile makes a bounded structured choice through a
chat API. That does not turn a chat model into a specialist decision model. For a
dedicated endpoint use its adapter, such as `ollama-decision`, or `command`.

## Model settings and structured output

Every model profile has `provider` and `kind`; non-demo/non-command transports
also require `model`. Optional `timeout` overrides the project default.

- OpenAI-compatible `options` are top-level request extras. For example,
  OpenRouter uses `reasoning = { effort = "high" }`; a compatible reseller might
  use `reasoning_effort = "high"`. Use that service's documented schema.
- Ollama generative `options` go into its API `options` object.
- Codex accepts `options.reasoning_effort` and `options.service_tier`.
- Claude accepts `options.effort`.
- Generic commands receive `model`, `model_options`, and `route` alongside the
  normal oracle request. They implement their own option handling.
- Demo and Ollama decision profiles accept no arbitrary options.

Model options cannot override the runtime's model, messages, streaming mode,
response schema, temperature, tool controls, or authentication/header fields.
Provider errors do not trigger an automatic downgrade, retry, or endpoint switch.

The default HTTP response mode is `json_schema`. Portable strict schemas require
concrete closed types, such as `String`, `List<Message>`, or `ModelResponse?`.
Unconstrained `Json`/`Any`, dynamic maps, and bare generic collections fail before
invocation in strict mode and in the built-in harness adapters. Choose an explicit
`response_format = "json_object"` for compatible HTTP models needing those shapes;
the prompt still contains the schema and the runtime still validates the answer.
Schema acceptance depends on the particular provider/model, even in strict mode.

Configured HTTP and harness adapters require every field requested by their
answer schema, including explicit nullable decision scores. Generic command
adapters retain the base protocol’s optional-field defaults. HTTP adapters reject
refusals, tool calls, truncated completions, malformed JSON, and invalid typed
answers. HTTP deadlines cover the entire invocation, not just socket inactivity. A direct `ModelResponse`/`ModelResponse?` takes token
counts and finish reason from actual HTTP response metadata. Nested response
records remain ordinary requested data. The built-in harness adapters currently
reject direct `ModelResponse` because they do not extract trustworthy usage
telemetry; use `String`, another typed record, or a custom adapter.

## Subscription harness examples

These are fragments to add to a manifest with both default lanes configured:

```toml
[providers.subscription_codex]
transport = "codex"
[models.editor]
provider = "subscription_codex"
kind = "generative"
model = "YOUR_CODEX_MODEL_ID"
timeout = 180
[models.editor.options]
reasoning_effort = "high"

[providers.subscription_claude]
transport = "claude"
[models.reviewer]
provider = "subscription_claude"
kind = "generative"
model = "YOUR_CLAUDE_MODEL_ID"
timeout = 180
[models.reviewer.options]
effort = "high"
```

Map route names to `editor` and `reviewer` in `[routes]`. No API key is read by
Purrhaps for these transports. Installed harnesses manage their own authentication;
subscription availability, model access and usage limits remain account-dependent.

Each call uses a fresh temporary workspace, passes the request on stdin, and reads
only the structured final-result channel. Temporary artifacts are removed. Codex
uses an ephemeral run, ignores user configuration, uses a read-only sandbox and a
noninteractive approval policy. **Read-only is not tool-free**: the adapter asks
Codex not to use tools but does not claim to disable all its tools. Claude disables
its tool set and MCP configuration, uses safe mode, and disables session persistence.
Neither adapter enables a permission bypass or reuses a prior conversation.

For `command`, all relative paths are relative to the manifest directory. For
harnesses, an explicit relative executable path is resolved against that directory
before entering the temporary workspace; other arguments are literal and are not
rewritten. Use absolute paths for wrapper-script arguments. Commands are argv
arrays and never passed through a shell. Timeouts terminate the invocation's
process group on supported Unix platforms.

## Native executables and deployment

Emit a configured project through the normal CLI:

```sh
python3 -m purrhaps --project examples/projects/routing --emit-rust build/routed-cat
cargo run --manifest-path build/routed-cat/Cargo.toml -- --project build/routed-cat
```

The emitted project includes `purrhaps.toml`, retaining credential variable names,
not resolved values. Native programs read configuration at runtime. Use `--project`
to select it explicitly when launching elsewhere; no original absolute manifest
path is baked into the executable. The executable runs its compiled source;
`project.entry` is used by the Python launcher, not to replace that source.
Build-time CLI execution overrides are not baked into the copied manifest; pass
those overrides to the native executable when launching it. Bare native execution discovers from its
current directory upward. Native `--model`, `--decision-model`, `--no-project`,
`--timeout`, and `--max-calls` follow the same rules as Python; native legacy command
overrides retain `--oracle` / `--oracle-arg` and their decision counterparts.

Relative command paths in the emitted manifest now refer to its **new directory**.
Deploy adapters with it or use deployment-appropriate absolute paths; emission does
not copy arbitrary executables, source files, account state, or credentials.
The native runtime uses the `toml` and `url` crates. Native HTTP transports additionally need
`curl` on PATH; its user configuration is disabled. Native transport stdout is capped at 8 MiB. Demo, subprocess, and harness
routes need no Python runtime. Harness routes require their respective CLI.

## Verification and future extensions

Tests use local HTTP fixture servers, synthetic environment entries and fake
harness executables. They verify request mapping, output validation, credential
resolution timing, route behavior, redaction, timeouts and native parity. They do
not claim live OpenRouter, reseller, subscription, or GPU model success.

Provider capability discovery, aliases for deployment environments, monetary
budgets, batched/streaming calls, retry policies, and a dependency lockfile can be
added as versioned features. The current manifest is deliberately strict about
what it can execute today.

References: [OpenRouter structured output](https://openrouter.ai/docs/guides/features/structured-outputs),
[OpenRouter reasoning options](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens),
[Codex noninteractive execution](https://developers.openai.com/codex/noninteractive),
and [Claude CLI reference](https://code.claude.com/docs/en/cli-reference).
