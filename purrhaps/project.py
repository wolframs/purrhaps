"""Project manifests and named model routes, validated without resolving secrets."""
from dataclasses import dataclass
import math
from pathlib import Path
import re
import tomllib
from urllib.parse import urlsplit

from .syntax import PurrhapsError
from .types import is_json


class ConfigurationError(PurrhapsError):
    """A malformed manifest or unavailable named route.

    Diagnostics deliberately describe the schema rather than echoing supplied
    values, unknown keys, URLs, or TOML parser snippets.
    """


_HTTP = {"openai-chat", "ollama", "ollama-decision"}
_TRANSPORTS = _HTTP | {"demo", "command", "codex", "claude"}
_KINDS = {"generative", "decision"}
_RESERVED = {"model", "messages", "stream", "response_format", "temperature", "tools", "tool_choice"}
_CREDENTIAL_KEYS = {
    "apikey", "apikeyenv", "authorization", "proxyauthorization", "headers", "header",
    "httpheaders", "extraheaders", "defaultheaders", "credential", "credentials", "password",
    "secret", "clientsecret", "token", "accesstoken", "refreshtoken", "authtoken",
    "bearertoken", "auth", "authentication",
}


def _fail(message):
    raise ConfigurationError(message)


def _table(value, allowed=None, required=(), context="configuration"):
    if type(value) is not dict:
        _fail(f"{context} must be a table")
    if allowed is not None and value.keys() - allowed:
        _fail(f"{context} contains unknown or unsupported keys")
    if set(required) - value.keys():
        _fail(f"{context} is missing required fields")
    return value


def _string(value, context):
    if type(value) is not str or not value.strip():
        _fail(f"{context} must be a nonempty string")


def _number(value, context, *, maximum=86400, allow_zero=False):
    if (type(value) not in (int, float) or not (0 <= value <= maximum if allow_zero else 0 < value <= maximum)
            or not math.isfinite(value)):
        _fail(f"{context} must be a finite number within the supported range")


def _argv(value):
    if type(value) is not list or not value or any(type(arg) is not str or "\0" in arg for arg in value):
        _fail("provider command must be a nonempty array of argument strings")
    if not value[0].strip():
        _fail("provider command requires a nonempty executable")


def _url(value):
    _string(value, "provider base_url")
    try:
        parsed = urlsplit(value)
        valid = (parsed.scheme in ("http", "https") and bool(parsed.netloc) and bool(parsed.hostname)
                 and parsed.username is None and parsed.password is None and "@" not in parsed.netloc
                 and "?" not in value and "#" not in value and "\\" not in value
                 and not any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in value))
        # Parsing the port also diagnoses nonnumeric/out-of-range port values.
        parsed.port
    except ValueError:
        valid = False
    if not valid:
        _fail("provider base_url must be an absolute HTTP(S) URL without userinfo, query or fragment")


def _credential_keys(value):
    if type(value) is dict:
        for key, item in value.items():
            normalized = re.sub(r"[^a-z0-9]", "", key.lower())
            if (normalized in _CREDENTIAL_KEYS or any(part in normalized for part in
                    ("apikey", "authorization", "password", "secret", "credential", "header"))):
                _fail("model options cannot contain credential or header fields")
            _credential_keys(item)
    elif type(value) is list:
        for item in value:
            _credential_keys(item)


def _validate(data):
    _table(data, {"manifest_version", "project", "runtime", "providers", "models", "routes", "defaults", "escalation"},
           ("manifest_version", "project", "providers", "models", "routes", "defaults"), "manifest")
    if type(data["manifest_version"]) is not int or data["manifest_version"] != 1:
        _fail("manifest_version must be the supported integer version 1")
    project = _table(data["project"], {"name", "entry", "no_humans_allowed"}, ("name",), "project")
    _string(project["name"], "project name")
    if "no_humans_allowed" in project and type(project["no_humans_allowed"]) is not bool:
        _fail("project no_humans_allowed must be a boolean")
    if "entry" in project:
        _string(project["entry"], "project entry")
        entry = Path(project["entry"])
        if entry.is_absolute() or entry == Path(".") or "\0" in project["entry"]:
            _fail("project entry must be a relative file path")
    runtime = _table(data.get("runtime", {}), {"max_calls", "timeout"}, context="runtime")
    if "max_calls" in runtime and (type(runtime["max_calls"]) is not int or runtime["max_calls"] <= 0):
        _fail("runtime max_calls must be a positive integer")
    if "timeout" in runtime:
        _number(runtime["timeout"], "runtime timeout")
    providers = _table(data["providers"], context="providers")
    for name, provider in providers.items():
        _string(name, "provider name")
        _table(provider, required=("transport",), context="provider")
        transport = provider["transport"]
        if type(transport) is not str or transport not in _TRANSPORTS:
            _fail("provider transport is unsupported")
        allowed = {"transport"}
        required = set()
        if transport in _HTTP:
            allowed |= {"base_url", "api_key_env"}
            required.add("base_url")
            if transport == "openai-chat":
                allowed.add("response_format")
        elif transport in ("command", "codex", "claude"):
            allowed.add("command")
            if transport == "command":
                required.add("command")
        _table(provider, allowed, required, "provider")
        if transport in _HTTP:
            _url(provider["base_url"])
            if "api_key_env" in provider and (type(provider["api_key_env"]) is not str
                    or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", provider["api_key_env"])):
                _fail("provider api_key_env must be an environment variable identifier")
        if "response_format" in provider and (type(provider["response_format"]) is not str
                or provider["response_format"] not in ("json_schema", "json_object")):
            _fail("provider response_format must be json_schema or json_object")
        if "command" in provider:
            _argv(provider["command"])
    models = _table(data["models"], context="models")
    for name, model in models.items():
        _string(name, "model profile name")
        _table(model, {"provider", "kind", "model", "timeout", "temperature", "options"},
               ("provider", "kind"), "model profile")
        _string(model["provider"], "model provider reference")
        if model["provider"] not in providers:
            _fail("model profile references an unknown provider")
        kind = model["kind"]
        if type(kind) is not str or kind not in _KINDS:
            _fail("model kind must be generative or decision")
        transport = providers[model["provider"]]["transport"]
        if "model" in model:
            _string(model["model"], "model identifier")
        elif transport not in ("demo", "command"):
            _fail("model profile requires a model identifier for this transport")
        if "timeout" in model:
            _number(model["timeout"], "model timeout")
        if "temperature" in model:
            _number(model["temperature"], "model temperature", maximum=2, allow_zero=True)
            if kind != "generative" or transport in ("codex", "claude", "ollama-decision"):
                _fail("model temperature is unsupported by this lane or transport")
        if transport in ("ollama", "codex", "claude") and kind != "generative":
            _fail("provider transport only supports generative models")
        if transport == "ollama-decision" and kind != "decision":
            _fail("provider transport only supports decision models")
        options = _table(model.get("options", {}), context="model options")
        if not is_json(options):
            _fail("model options must contain only finite JSON values")
        _credential_keys(options)
        if transport == "openai-chat" and options.keys() & _RESERVED:
            _fail("model options contain reserved request fields")
        if transport == "ollama" and "temperature" in options:
            _fail("model options cannot override profile temperature")
        if transport in ("demo", "ollama-decision") and options:
            _fail("provider transport does not support model options")
        if transport in ("codex", "claude"):
            allowed = {"reasoning_effort", "service_tier"} if transport == "codex" else {"effort"}
            _table(options, allowed, context="model options")
            for option in options.values():
                _string(option, "harness model option")
    routes = _table(data["routes"], context="routes")
    for name, target in routes.items():
        _string(name, "route name")
        _string(target, "route model reference")
        if target not in models:
            _fail("route references an unknown model profile")
    defaults = _table(data["defaults"], _KINDS, _KINDS, "defaults")
    for kind, route in defaults.items():
        _string(route, "default route reference")
        if route not in routes:
            _fail("default references an unknown route")
        if models[routes[route]]["kind"] != kind:
            _fail("default route model kind does not match its lane")
    escalation = _table(data.get("escalation", {}),
                        {"decision", "generative", "threshold", "exposure", "on_low_confidence"},
                        context="escalation")
    for kind in _KINDS:
        if kind in escalation:
            route = escalation[kind]
            _string(route, "escalation route reference")
            if route not in routes or models[routes[route]]["kind"] != kind:
                _fail("escalation route must exist and match its lane")
    if "threshold" in escalation:
        _number(escalation["threshold"], "escalation threshold", maximum=1, allow_zero=True)
    for key, allowed in (("exposure", {"blind", "informed"}),
                         ("on_low_confidence", {"error", "random"})):
        if key in escalation and (type(escalation[key]) is not str or escalation[key] not in allowed):
            _fail("escalation " + key + " is unsupported")
    return data


@dataclass(frozen=True)
class Project:
    path: Path
    root: Path
    data: dict

    def route(self, name, kind=None):
        if type(name) is not str or name not in self.data["routes"]:
            _fail("unknown model route")
        if kind is not None and (type(kind) is not str or kind not in _KINDS):
            _fail("unknown model lane")
        model = self.data["models"][self.data["routes"][name]]
        if kind is not None and model["kind"] != kind:
            _fail("model route kind does not match the requested lane")
        return self.data["providers"][model["provider"]], model

    def oracle(self, route, seed=0, timeout_override=None):
        provider, model = self.route(route)
        timeout = (timeout_override if timeout_override is not None else model.get("timeout",
                   self.data.get("runtime", {}).get("timeout", 30)))
        _number(timeout, "oracle timeout")
        # Provider construction is intentionally absent from load/check paths.
        from .providers import ConfiguredOracle
        return ConfiguredOracle(provider, model, self.root, route, seed=seed, timeout=float(timeout))


def load_project(path):
    """Load and validate a manifest without reading environment/auth or making calls."""
    path = Path(path).resolve()
    if path.is_dir():
        path = path / "purrhaps.toml"
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except (OSError, ValueError):
        raise ConfigurationError("cannot read or parse project manifest") from None
    return Project(path, path.parent, _validate(data))


def discover_project(start):
    """Find the nearest manifest, starting at a directory (or a source file)."""
    start = Path(start).resolve()
    if start.is_file():
        start = start.parent
    for directory in (start, *start.parents):
        candidate = directory / "purrhaps.toml"
        if candidate.is_file():
            return candidate
    return None
