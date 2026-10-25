"""Offline project validation, lazy route semantics, and command-line selection."""
import copy
from datetime import date, datetime, time
import io
import json
import math
import os
from pathlib import Path
import shlex
import sys
import tempfile
import unittest
from unittest import mock

from purrhaps.__main__ import main
from purrhaps.oracle import Answer, OracleError
from purrhaps.project import ConfigurationError, discover_project, load_project
from purrhaps.runtime import Interpreter, TypeError
from purrhaps.syntax import Parser, PurrhapsError


def manifest():
    return {
        "manifest_version": 1,
        "project": {"name": "test cats", "entry": "main.prr"},
        "providers": {"offline": {"transport": "demo"}},
        "models": {
            "writer": {"provider": "offline", "kind": "generative"},
            "careful": {"provider": "offline", "kind": "generative", "temperature": .2},
            "judge": {"provider": "offline", "kind": "decision"},
            "other_judge": {"provider": "offline", "kind": "decision"},
        },
        "routes": {"writing": "writer", "careful": "careful", "triage": "judge", "audit": "other_judge"},
        "defaults": {"generative": "writing", "decision": "triage"},
    }


def toml_value(value):
    """Tiny test fixture writer; also emits invalid TOML JSON values on purpose."""
    if type(value) is dict:
        return "{ " + ", ".join(json.dumps(k) + " = " + toml_value(v) for k, v in value.items()) + " }"
    if type(value) is list:
        return "[" + ", ".join(map(toml_value, value)) + "]"
    if type(value) in (date, datetime, time):
        return value.isoformat()
    if type(value) is float and not math.isfinite(value):
        return str(value)
    return json.dumps(value, ensure_ascii=False)


class ProjectFixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "purrhaps.toml"

    def write(self, data=None, source='print(hi())'):
        data = manifest() if data is None else data
        self.path.write_text("\n".join(k + " = " + toml_value(v) for k, v in data.items()), encoding="utf-8")
        (self.root / "main.prr").write_text(source, encoding="utf-8")
        return load_project(self.path)

    def bad(self, data):
        with self.assertRaises(ConfigurationError) as caught:
            self.write(data)
        self.assertIsInstance(caught.exception, PurrhapsError)
        self.assertNotIn("PRIVATE_MARKER", str(caught.exception))

    def cli(self, *args, cwd=None, source=""):
        # Run the CLI in-process so tests can prove provider constructors were not
        # entered during --check without inspecting any real environment values.
        old_cwd = Path.cwd()
        self.addCleanup(os.chdir, old_cwd)
        if cwd is not None:
            os.chdir(cwd)
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch("sys.stdin", io.StringIO(source)), mock.patch("sys.stdout", stdout), mock.patch("sys.stderr", stderr):
            try:
                status = main(list(args))
            except SystemExit as exc:
                status = exc.code
        os.chdir(old_cwd)
        return status, stdout.getvalue(), stderr.getvalue()


class ManifestTests(ProjectFixture):
    def test_no_humans_allowed_is_an_optional_strict_boolean(self):
        self.assertNotIn("no_humans_allowed", self.write().data["project"])
        for value in (False, True):
            data = manifest()
            data["project"]["no_humans_allowed"] = value
            self.assertIs(self.write(data).data["project"]["no_humans_allowed"], value)
        for value in (0, 1, "true", "PRIVATE_MARKER", [], {}):
            with self.subTest(value=value):
                data = manifest()
                data["project"]["no_humans_allowed"] = value
                self.bad(data)

    def test_public_api_and_no_environment_or_transport_during_load(self):
        self.write()
        with mock.patch("os._Environ.__getitem__", side_effect=AssertionError("environment read")), \
                mock.patch("purrhaps.providers.ConfiguredOracle", side_effect=AssertionError("transport constructed")):
            project = load_project(self.path)
        self.assertEqual(project.path, self.path.resolve())
        self.assertEqual(project.root, self.root.resolve())
        self.assertEqual(project.data, manifest())
        self.assertEqual(project.route("writing", "generative"),
                         (manifest()["providers"]["offline"], manifest()["models"]["writer"]))
        self.assertEqual(load_project(self.root).path, project.path)

    def test_discovery_nearest_and_absent(self):
        self.write()
        child = self.root / "nested"
        child.mkdir()
        source = child / "source.prr"
        source.write_text("print(hi())")
        self.assertEqual(discover_project(child), self.path)
        self.assertEqual(discover_project(source), self.path)
        nested_manifest = child / "purrhaps.toml"
        nested_manifest.write_text("invalid")
        self.assertEqual(discover_project(child), nested_manifest)
        self.assertIsNone(discover_project(Path("/")))

    def test_invalid_toml_is_redacted_and_missing_file_is_configuration_error(self):
        for text in ('PRIVATE_MARKER = "unterminated', 'x = 1\nx = "PRIVATE_MARKER"'):
            self.path.write_text(text)
            with self.assertRaises(ConfigurationError) as caught:
                load_project(self.path)
            self.assertNotIn("PRIVATE_MARKER", str(caught.exception))
            self.assertNotIn(text, str(caught.exception))
        with self.assertRaises(ConfigurationError):
            load_project(self.root / "missing.toml")

    def test_unknown_keys_in_every_fixed_schema_table(self):
        for path in ((), ("project",), ("runtime",), ("providers", "offline"),
                     ("models", "writer"), ("defaults",)):
            with self.subTest(path=path):
                data = manifest()
                node = data
                for key in path:
                    node = node.setdefault(key, {})
                node["PRIVATE_MARKER"] = "PRIVATE_MARKER"
                self.bad(data)

    def test_required_fields(self):
        for path in (("manifest_version",), ("project",), ("providers",), ("models",), ("routes",), ("defaults",),
                     ("project", "name"), ("providers", "offline", "transport"),
                     ("models", "writer", "provider"), ("models", "writer", "kind"),
                     ("defaults", "generative"), ("defaults", "decision")):
            with self.subTest(path=path):
                data = manifest()
                node = data
                for key in path[:-1]:
                    node = node[key]
                del node[path[-1]]
                self.bad(data)

    def test_table_types(self):
        for path in (("project",), ("runtime",), ("providers",), ("providers", "offline"),
                     ("models",), ("models", "writer"), ("routes",), ("defaults",)):
            for value in (1, True, "PRIVATE_MARKER", []):
                with self.subTest(path=path, value=value):
                    data = manifest()
                    node = data
                    for key in path[:-1]:
                        node = node[key]
                    node[path[-1]] = value
                    self.bad(data)

    def test_versions_names_and_string_fields(self):
        for value in (True, False, "1", 1.0, 2, 0, [], {}):
            data = manifest()
            data["manifest_version"] = value
            self.bad(data)
        for path in (("project", "name"), ("project", "entry"), ("models", "writer", "provider"),
                     ("models", "writer", "kind"), ("models", "writer", "model"),
                     ("routes", "writing"), ("defaults", "generative")):
            for value in (True, 1, 1.2, [], {}, "", "   "):
                data = manifest()
                node = data
                for key in path[:-1]:
                    node = node[key]
                node[path[-1]] = value
                self.bad(data)
        for section in ("providers", "models", "routes"):
            data = manifest()
            data[section][""] = next(iter(data[section].values()))
            self.bad(data)
        for entry in ("/absolute/main.prr", ".", "a\0b"):
            data = manifest()
            data["project"]["entry"] = entry
            self.bad(data)

    def test_references_and_kind_errors(self):
        for path in (("models", "writer", "provider"), ("routes", "writing"), ("defaults", "generative")):
            data = manifest()
            node = data
            for key in path[:-1]:
                node = node[key]
            node[path[-1]] = "PRIVATE_MARKER"
            self.bad(data)
        data = manifest()
        data["defaults"]["generative"] = "triage"
        self.bad(data)
        data = manifest()
        data["models"]["writer"]["kind"] = "PRIVATE_MARKER"
        self.bad(data)
        project = self.write()
        for route, kind in (("PRIVATE_MARKER", None), ("triage", "generative"), ([], None), ("writing", [])):
            with self.assertRaises(ConfigurationError) as caught:
                project.route(route, kind)
            self.assertNotIn("PRIVATE_MARKER", str(caught.exception))

    def test_numbers_reject_bools_nonfinite_out_of_range_and_wrong_scalar_types(self):
        for path in (("runtime", "timeout"), ("models", "writer", "timeout"), ("models", "writer", "temperature")):
            for value in (True, False, "30", [], {}, float("nan"), float("inf"), float("-inf"), -1, 86401):
                data = manifest()
                node = data
                for key in path[:-1]:
                    node = node.setdefault(key, {})
                node[path[-1]] = value
                self.bad(data)
        for field in ("timeout", "max_calls"):
            for value in (0, True, "2", [], {}, -1):
                data = manifest()
                data["runtime"] = {field: value}
                self.bad(data)
        data = manifest()
        data["runtime"] = {"max_calls": 1.0}
        self.bad(data)
        for value in (2.01,):
            data = manifest()
            data["models"]["writer"]["temperature"] = value
            self.bad(data)
        data = manifest()
        data["runtime"] = {"max_calls": 1, "timeout": 86400}
        data["models"]["writer"].update(timeout=.01, temperature=0)
        self.assertEqual(self.write(data).data, data)

    def test_timeout_precedence_and_constructor_contract(self):
        for runtime_timeout, model_timeout, override, expected in (
                (None, None, None, 30), (17, None, None, 17), (17, 12, None, 12), (17, 12, 4, 4)):
            data = manifest()
            if runtime_timeout is not None:
                data["runtime"] = {"timeout": runtime_timeout}
            if model_timeout is not None:
                data["models"]["writer"]["timeout"] = model_timeout
            project = self.write(data)
            with mock.patch("purrhaps.providers.ConfiguredOracle") as constructor:
                self.assertIs(project.oracle("writing", seed=19, timeout_override=override), constructor.return_value)
                constructor.assert_called_once_with(project.data["providers"]["offline"], project.data["models"]["writer"],
                                                    project.root, "writing", seed=19, timeout=float(expected))
        for value in (0, True, float("nan"), 86401):
            with self.assertRaises(ConfigurationError):
                project.oracle("writing", timeout_override=value)

    def external(self, transport="openai-chat", kind="generative"):
        data = manifest()
        provider = {"transport": transport}
        if transport in ("openai-chat", "ollama", "ollama-decision"):
            provider["base_url"] = "https://example.invalid/v1"
        if transport == "command":
            provider["command"] = ["adapter", ""]
        data["providers"]["external"] = provider
        data["models"]["external"] = {"provider": "external", "kind": kind, "model": "vendor/exact-model"}
        return data

    def test_all_transports_valid_capabilities(self):
        for transport, kind in (("demo", "generative"), ("demo", "decision"), ("command", "generative"),
                                ("command", "decision"), ("openai-chat", "generative"), ("openai-chat", "decision"),
                                ("ollama", "generative"), ("ollama-decision", "decision"), ("codex", "generative"),
                                ("claude", "generative")):
            with self.subTest(transport=transport, kind=kind):
                data = self.external(transport, kind)
                self.assertEqual(self.write(data).data, data)

    def test_provider_scalar_types_and_invalid_transport(self):
        for value in (True, 1, 1.5, [], {}, "PRIVATE_MARKER"):
            data = manifest()
            data["providers"]["offline"]["transport"] = value
            self.bad(data)
        for value in (True, 1, 1.5, [], {}, "", "PRIVATE_MARKER"):
            data = self.external()
            data["providers"]["external"]["base_url"] = value
            self.bad(data)
        for value in (True, 1, 1.5, [], {}, "", "a-b", "A=B", "2NAME", "PRIVATE_MARKER\n"):
            data = self.external()
            data["providers"]["external"]["api_key_env"] = value
            self.bad(data)
        for value in (True, 1, 1.5, [], {}, "", "PRIVATE_MARKER"):
            data = self.external()
            data["providers"]["external"]["response_format"] = value
            self.bad(data)
        data = self.external()
        data["providers"]["external"].update(api_key_env="DUMMY_TEST_ENV", response_format="json_object")
        self.write(data)

    def test_http_url_constraints(self):
        for value in ("ftp://example.invalid", "//example.invalid", "https:///path", "https://",
                      "https://user:PRIVATE_MARKER@example.invalid", "https://user@example.invalid",
                      "https://example.invalid/?PRIVATE_MARKER", "https://example.invalid/#PRIVATE_MARKER",
                      "https://example.invalid/?", "https://example.invalid/#", "https://example.invalid:bad",
                      "https://example.invalid:70000", "https://example.invalid/with space", "https://[bad",
                      "https://example.invalid/\nPRIVATE_MARKER", "https://example.invalid\\PRIVATE_MARKER"):
            data = self.external()
            data["providers"]["external"]["base_url"] = value
            self.bad(data)
        for value in ("http://127.0.0.1:1234/v1", "https://example.invalid", "http://[::1]:1234"):
            data = self.external()
            data["providers"]["external"]["base_url"] = value
            self.write(data)

    def test_missing_transport_fields_and_model_identifier(self):
        for transport in ("openai-chat", "ollama", "ollama-decision"):
            data = self.external(transport, "decision" if transport == "ollama-decision" else "generative")
            del data["providers"]["external"]["base_url"]
            self.bad(data)
        data = self.external("command")
        del data["providers"]["external"]["command"]
        self.bad(data)
        for transport in ("openai-chat", "ollama", "ollama-decision", "codex", "claude"):
            data = self.external(transport, "decision" if transport == "ollama-decision" else "generative")
            del data["models"]["external"]["model"]
            self.bad(data)

    def test_command_and_harness_prefix_argv_types_and_literal_arguments(self):
        for transport in ("command", "codex", "claude"):
            for value in ("PRIVATE_MARKER", [], {}, True, [1], [""], ["  "], ["adapter", True], ["a\0b"]):
                data = self.external(transport)
                data["providers"]["external"]["command"] = value
                self.bad(data)
            data = self.external(transport)
            data["providers"]["external"]["command"] = ["./wrapper", "", "$(PRIVATE_MARKER)", "$HOME"]
            self.write(data)

    def test_no_transport_cross_fields(self):
        for transport, field, value in (("demo", "base_url", "https://example.invalid"),
                                       ("demo", "command", ["adapter"]), ("command", "base_url", "https://example.invalid"),
                                       ("codex", "api_key_env", "DUMMY"), ("claude", "response_format", "json_schema"),
                                       ("ollama", "response_format", "json_schema"),
                                       ("openai-chat", "command", ["adapter"]),
                                       ("openai-chat", "headers", {"PRIVATE_MARKER": "PRIVATE_MARKER"})):
            data = self.external(transport)
            data["providers"]["external"][field] = value
            self.bad(data)

    def test_model_capability_validation(self):
        for transport, kind in (("ollama", "decision"), ("codex", "decision"), ("claude", "decision"),
                                ("ollama-decision", "generative")):
            self.bad(self.external(transport, kind))
        for transport, kind in (("demo", "decision"), ("command", "decision"), ("openai-chat", "decision"),
                                ("codex", "generative"), ("claude", "generative"), ("ollama-decision", "decision")):
            data = self.external(transport, kind)
            data["models"]["external"]["temperature"] = .5
            self.bad(data)

    def test_options_require_tables_of_finite_json_recursively(self):
        for value in (True, "PRIVATE_MARKER", 1, [], [1]):
            data = self.external("command")
            data["models"]["external"]["options"] = value
            self.bad(data)
        for value in (date(2026, 1, 1), time(12, 0), datetime(2026, 1, 1), float("nan"), float("inf")):
            data = self.external("command")
            data["models"]["external"]["options"] = {"nested": [{"value": value}]}
            self.bad(data)
        data = self.external("command")
        options = {"nested": [{"bool": True, "number": 1, "decimal": .2, "str": "literal", "array": []}], "empty": {}}
        data["models"]["external"]["options"] = options
        self.assertEqual(self.write(data).data["models"]["external"]["options"], options)

    def test_nested_credential_and_header_keys_rejected(self):
        for transport in ("command", "openai-chat", "ollama"):
            for key in ("api_key", "API-KEY", "x-api-key", "Authorization", "headers", "extra_headers", "access_token",
                        "refresh_token", "auth_token", "authentication", "password", "secret", "client_secret"):
                data = self.external(transport)
                data["models"]["external"]["options"] = {"provider": {"nested": [{key: "PRIVATE_MARKER"}]}}
                self.bad(data)

    def test_transport_specific_options_and_exact_effort_spelling(self):
        for transport in ("demo", "ollama-decision"):
            data = self.external(transport, "decision" if transport == "ollama-decision" else "generative")
            data["models"]["external"]["options"] = {"effort": "max"}
            self.bad(data)
        for reserved in ("model", "messages", "stream", "response_format", "temperature", "tools", "tool_choice"):
            data = self.external()
            data["models"]["external"]["options"] = {reserved: "PRIVATE_MARKER"}
            self.bad(data)
        data = self.external("ollama")
        data["models"]["external"]["options"] = {"temperature": .5}
        self.bad(data)
        for transport, allowed in (("codex", {"reasoning_effort": "max", "service_tier": "priority"}),
                                   ("claude", {"effort": "max"})):
            data = self.external(transport)
            data["models"]["external"]["options"] = allowed
            self.assertEqual(self.write(data).data["models"]["external"]["options"], allowed)
            for value in ("", "  ", 1, True, {}, {"nested": "max"}, ["max"]):
                bad = copy.deepcopy(data)
                bad["models"]["external"]["options"] = {next(iter(allowed)): value}
                self.bad(bad)
            data["models"]["external"]["options"] = {"PRIVATE_MARKER": "max"}
            self.bad(data)
        data = self.external()
        data["models"]["external"]["options"] = {
            "reasoning": {"effort": "max"}, "reasoning_effort": "exact-custom-value",
            "max_tokens": 10, "max_completion_tokens": 20,
            "provider": {"order": ["reseller"], "allow_fallbacks": False},
        }
        self.assertEqual(self.write(data).data, data)


class RecordingOracle:
    def __init__(self, route, records):
        self.route, self.records = route, records

    def ask(self, request):
        self.records.append((self.route, dict(request)))
        if request["kind"] in ("decision", "difference"):
            return Answer(request["options"][0], .99)
        return Answer(request["input"], .99, "Recorded sample.")


class RoutingTests(ProjectFixture):
    def runtime(self, data=None, **kwargs):
        project = self.write(data)
        records = []
        factory = mock.patch.object(type(project), "oracle",
                                    new=lambda project, route, **kw: RecordingOracle(route, records))
        factory.start()
        self.addCleanup(factory.stop)
        runtime = Interpreter(project=project, output=io.StringIO(), **kwargs)
        return runtime, records

    def test_parser_model_node_and_expression(self):
        node = Parser('with model("care" + "ful") { print(hi()) }').program()[0]
        self.assertEqual((node.kind, node.line), ("model", 1))
        self.assertEqual(node.args[0].kind, "binary")

    def test_route_selection_is_call_free(self):
        runtime, records = self.runtime()
        runtime.run('with model("careful") { val x = 1 }\nwith model("audit") { print(hi()) }')
        self.assertEqual(records, [])
        self.assertEqual(runtime.calls, 0)
        self.assertEqual((runtime.generative_route, runtime.decision_route), ("writing", "triage"))

    def test_defaults_and_constructor_route_overrides(self):
        runtime, records = self.runtime(generative_route="careful", decision_route="audit")
        runtime.run('maybe x: String = "hi"\ndefinitely y = DecisionInput("which", [1, 2], null)\nprint(x, y)')
        self.assertEqual([route for route, _ in records], ["careful", "audit"])
        self.assertEqual(runtime.calls, 2)
        self.assertEqual(runtime.decision_route, "audit")

    def test_lanes_nested_dynamic_forcing_and_restoration(self):
        runtime, records = self.runtime()
        runtime.run('''maybe x: String = "hi"
definitely y: Int = DecisionInput("which", [1, 2], null)
print(x, y)
with model("careful") {
 print(x, y)
 with model("audit") { print(x, y) }
 print(x, y)
}
print(x, y)''')
        self.assertEqual([route for route, _ in records], ["writing", "triage", "careful", "triage", "careful", "audit",
                                                         "careful", "triage", "writing", "triage"])

    def test_returned_lazy_value_uses_reader_route_and_collapse_binds(self):
        runtime, records = self.runtime()
        runtime.run('''fun lazy() { with model("careful") { maybe x: String = "hi"; return x } }
fun bound() { with model("careful") { maybe x: String = "hi"; return x!! } }
val x = lazy()
print(x)
print(bound())
print(x)''')
        self.assertEqual([route for route, _ in records], ["writing", "careful", "writing"])
        self.assertEqual(runtime.generative_route, "writing")

    def test_perhaps_cache_is_per_reading_scope(self):
        runtime, records = self.runtime()
        runtime.run('''perhaps x: String = "hi"
print(x, x)
with model("careful") { print(x, x) }
with model("writing") { print(x, x) }
print(x, x)''')
        self.assertEqual([route for route, _ in records], ["writing", "careful", "writing"])

    def test_temperature_profile_and_explicit_depth_restore(self):
        data = manifest()
        data["models"]["writer"]["temperature"] = .4
        runtime, records = self.runtime(data)
        runtime.run('''maybe x: String = "hi"
print(x)
with model("careful") {
 print(x)
 with temperature(0) {
  print(x)
  with model("writing") { print(x) }
  with temperature(1.2) { print(x) }
  print(x)
 }
 print(x)
}
print(x)''')
        self.assertEqual([req["temperature"] for _, req in records], [.4, .2, 0, 0, 1.2, 0, .2, .4])
        self.assertEqual([req["temperature_explicit"] for _, req in records], [False, False, True, True, True, True, False, False])
        self.assertEqual(runtime.temperature, .7)
        self.assertEqual(runtime._temperature_depth, 0)

    def test_routes_and_temperature_restored_after_return_and_exception(self):
        runtime, _ = self.runtime()
        runtime.run('fun f() { with model("careful") { with model("audit") { with temperature(0) { return 1 } } } }\nf()')
        self.assertEqual((runtime.generative_route, runtime.decision_route, runtime.temperature, runtime._temperature_depth),
                         ("writing", "triage", .7, 0))
        with self.assertRaises(TypeError):
            runtime.run('with model("careful") { with model("audit") { with temperature(0) { val invalid: Int = true } } }')
        self.assertEqual((runtime.generative_route, runtime.decision_route, runtime.temperature, runtime._temperature_depth),
                         ("writing", "triage", .7, 0))

    def test_model_expression_must_be_definite_string(self):
        for source in ('with model(1) {}', 'maybe x = "careful"\nwith model(x) {}'):
            runtime, records = self.runtime()
            with self.assertRaises(TypeError):
                runtime.run(source)
            self.assertEqual(records, [])
        runtime, records = self.runtime()
        runtime.run('val route = "care" + "ful"\nwith model(route) { maybe x = 1; print(x) }')
        self.assertEqual(records[0][0], "careful")

    def test_missing_route_and_project_fail_without_demo_fallback(self):
        runtime, records = self.runtime()
        with self.assertRaises(ConfigurationError):
            runtime.run('with model("PRIVATE_MARKER") {}')
        self.assertEqual(records, [])
        for kwargs in ({}, {"generative_route": "writing"}, {"decision_route": "triage"}):
            with self.assertRaises(ConfigurationError):
                Interpreter(**kwargs).run('with model("writing") {}')

    def test_shared_budget_manifest_and_explicit_override(self):
        data = manifest()
        data["runtime"] = {"max_calls": 1}
        runtime, records = self.runtime(data)
        with self.assertRaisesRegex(OracleError, "budget exhausted"):
            runtime.run('maybe x = 1\nprint(x)\nwith model("audit") { print(1 ?!= 2) }')
        self.assertEqual((runtime.calls, len(records)), (1, 1))
        runtime, records = self.runtime(data, max_calls=2)
        runtime.run('maybe x = 1\nprint(x)\nwith model("audit") { print(1 ?!= 2) }')
        self.assertEqual((runtime.calls, len(records)), (2, 2))

    def test_external_default_override_preserved_named_blocks_select_config(self):
        external_records = []
        external = RecordingOracle("external", external_records)
        data = manifest()
        data["models"]["writer"]["temperature"] = .3
        runtime, records = self.runtime(data, oracle=external)
        runtime.run('maybe x = 1\nprint(x)\nwith model("writing") { print(x) }\nprint(x)')
        self.assertEqual([route for route, _ in external_records], ["external", "external"])
        self.assertEqual([route for route, _ in records], ["writing"])
        self.assertEqual([request["temperature"] for _, request in external_records], [.7, .7])
        self.assertEqual(records[0][1]["temperature"], .3)
        self.assertIs(runtime.oracle, external)

    def test_explicit_temperature_unsupported_transports_fail_before_call(self):
        for transport in ("codex", "claude", "ollama-decision"):
            with self.subTest(transport=transport):
                data = manifest()
                provider = {"transport": transport}
                if transport == "ollama-decision":
                    provider["base_url"] = "http://example.invalid"
                data["providers"]["unsupported"] = provider
                model = "judge" if transport == "ollama-decision" else "writer"
                data["models"][model].update(provider="unsupported", model="exact/model")
                project = self.write(data)
                runtime = Interpreter(project=project, output=io.StringIO())
                with mock.patch("purrhaps.providers._run", side_effect=AssertionError("transport invoked")), \
                        mock.patch("purrhaps.providers.ConfiguredOracle._post", side_effect=AssertionError("HTTP invoked")):
                    with self.assertRaisesRegex(OracleError, "temperature"):
                        runtime.run('with temperature(0) { print(1 ?!= 2) }' if model == "judge" else
                                    'maybe x: String = "hi"\nwith temperature(0) { print(x) }')
                self.assertEqual(runtime._temperature_depth, 0)

    def test_unconfigured_requests_keep_original_exact_protocol(self):
        records = []
        runtime = Interpreter(RecordingOracle("legacy", records), output=io.StringIO())
        runtime.run('maybe x = 1\nprint(x)')
        self.assertEqual(set(records[0][1]), {"protocol", "kind", "input", "type", "line", "source", "temperature"})


class ProjectCliTests(ProjectFixture):
    def test_discovery_entry_explicit_source_and_no_project(self):
        self.write()
        nested = self.root / "nested"
        nested.mkdir()
        (nested / "other.prr").write_text('print("other")')
        for args, cwd, expected in (((), nested, "hi!"), (("other.prr",), nested, "other"),
                                    ((str(nested / "other.prr"),), Path("/tmp"), "other")):
            status, out, err = self.cli(*args, cwd=cwd)
            self.assertEqual(status, 0, err)
            self.assertIn(expected, out)
        status, out, err = self.cli("--no-project", cwd=nested)
        self.assertEqual(status, 1)
        self.assertIn("source file", err)
        self.assertNotIn("Traceback", err)

    def test_explicit_manifest_file_or_directory(self):
        self.write(source='print("selected")')
        for selected in (self.path, self.root):
            status, out, err = self.cli("--project", str(selected), cwd=Path("/tmp"))
            self.assertEqual(status, 0, err)
            self.assertEqual(out, "selected\n")

    def test_explicit_source_uses_its_parent_manifest(self):
        self.write(source='with model("careful") { print(hi()) }')
        status, out, err = self.cli(str(self.root / "main.prr"), cwd=Path("/tmp"))
        self.assertEqual(status, 0, err)
        self.assertIn("hi!", out)
        status, out, err = self.cli(str(self.root / "main.prr"), "--no-project", cwd=Path("/tmp"))
        self.assertEqual(status, 1)
        self.assertIn("project manifest", err)

    def test_no_file_or_entry_diagnosed(self):
        data = manifest()
        del data["project"]["entry"]
        self.write(data)
        for args in (("--no-project",), ("--project", str(self.path))):
            status, out, err = self.cli(*args, cwd=self.root)
            self.assertEqual(status, 1)
            self.assertIn("source file", err)

    def test_check_validates_manifest_and_routes_without_environment_or_provider_construction(self):
        data = manifest()
        data["providers"]["http"] = {"transport": "openai-chat", "base_url": "https://example.invalid", "api_key_env": "DUMMY_MISSING_ENV"}
        data["models"]["writer"].update(provider="http", model="vendor/model")
        self.write(data, 'maybe x: String = "hi"\nprint(x)')
        with mock.patch("os._Environ.__getitem__", side_effect=AssertionError("environment read")), \
                mock.patch("gettext.find", return_value=[]), \
                mock.patch("shutil.get_terminal_size", return_value=os.terminal_size((80, 24))), \
                mock.patch("purrhaps.providers.ConfiguredOracle", side_effect=AssertionError("transport constructed")):
            status, out, err = self.cli("--project", str(self.path), "--check", "--model", "writing")
        self.assertEqual(status, 0, err)
        self.assertIn("syntax is purrfect", out)
        data["models"]["writer"]["options"] = {"headers": {"Authorization": "PRIVATE_MARKER"}}
        self.path.write_text("\n".join(k + " = " + toml_value(v) for k, v in data.items()))
        status, out, err = self.cli("--project", str(self.path), "--check")
        self.assertEqual(status, 1)
        self.assertIn("ConfigurationError", err)
        self.assertNotIn("PRIVATE_MARKER", err)

    def test_bad_syntax_check_and_unknown_route_override(self):
        self.write(source="maybe x =")
        status, out, err = self.cli("--project", str(self.path), "--check")
        self.assertEqual(status, 1)
        self.assertIn("SyntaxError", err)
        for flags in (("--model", "PRIVATE_MARKER"), ("--decision-model", "writing"), ("--model", "triage")):
            status, out, err = self.cli("--project", str(self.path), "--check", *flags)
            self.assertEqual(status, 1)
            self.assertIn("ConfigurationError", err)
            self.assertNotIn("PRIVATE_MARKER", err)

    def test_emit_rust_copies_manifest_without_constructing_provider(self):
        data = manifest()
        data["project"]["no_humans_allowed"] = True
        data["providers"]["http"] = {
            "transport": "openai-chat", "base_url": "https://example.invalid/v1", "api_key_env": "DUMMY_BUILD_ENV",
        }
        data["models"]["writer"].update(provider="http", model="vendor/exact-model")
        self.write(data, 'maybe x: String = "hi"\nwith model("careful") { print(x) }')
        destination = self.root / "emitted"
        with mock.patch("os._Environ.__getitem__", side_effect=AssertionError("environment read")), \
                mock.patch("gettext.find", return_value=[]), \
                mock.patch("shutil.get_terminal_size", return_value=os.terminal_size((80, 24))), \
                mock.patch("purrhaps.providers.ConfiguredOracle", side_effect=AssertionError("transport constructed")):
            status, out, err = self.cli("--project", str(self.path), "--emit-rust", str(destination))
        self.assertEqual(status, 0, err)
        self.assertEqual(load_project(destination).data, data)
        generated = (destination / "src/main.rs").read_text()
        self.assertNotIn("example.invalid", generated)
        self.assertNotIn("DUMMY_BUILD_ENV", generated)
        self.assertNotIn(str(self.root), generated)

    def test_selection_conflicts_and_cli_numeric_validation(self):
        for flags in (("--project", str(self.path), "--no-project"),
                      ("--model", "writing", "--oracle-command", "adapter"),
                      ("--decision-model", "triage", "--decision-oracle-command", "adapter"),
                      ("--timeout", "0"), ("--timeout", "nan"), ("--timeout", "86401"), ("--max-calls", "0")):
            status, out, err = self.cli("-", *flags, source="print(hi())")
            self.assertEqual(status, 2)
        status, out, err = self.cli("-", "--no-project", "--model", "writing", source="print(hi())")
        self.assertEqual(status, 1)
        self.assertIn("project manifest", err)

    def test_cli_timeout_route_seed_and_budget_precedence(self):
        data = manifest()
        data["runtime"] = {"max_calls": 1, "timeout": 17}
        data["models"]["careful"]["timeout"] = 12
        self.write(data, 'maybe x = 1\nprint(x, x)')
        oracles = []
        def create(provider, model, root, route, seed=0, timeout=30):
            oracles.append((route, seed, timeout))
            return RecordingOracle(route, [])
        with mock.patch("purrhaps.providers.ConfiguredOracle", side_effect=create):
            status, out, err = self.cli("--project", str(self.path), "--model", "careful", "--decision-model", "audit", "--seed", "19")
        self.assertEqual(status, 1)
        self.assertIn("budget exhausted", err)
        self.assertEqual(oracles, [("careful", 19, 12.0), ("audit", 19, 17.0)])
        oracles.clear()
        with mock.patch("purrhaps.providers.ConfiguredOracle", side_effect=create):
            status, out, err = self.cli("--project", str(self.path), "--model", "careful", "--seed", "19",
                                        "--timeout", "4", "--max-calls", "2", "--stats")
        self.assertEqual(status, 0, err)
        self.assertEqual(oracles, [("careful", 19, 4.0), ("triage", 19, 4.0)])
        self.assertIn("calls=2", err)

    def test_actual_command_request_routes_options_and_manifest_cwd(self):
        data = manifest()
        script = self.root / "adapter.py"
        script.write_text('''import json, pathlib, sys
r = json.load(sys.stdin)
with pathlib.Path("requests.jsonl").open("a") as out: out.write(json.dumps(r) + "\\n")
value = r["options"][0] if r["kind"] in ("decision", "difference") else r["input"]
print(json.dumps({"value": value, "confidence": .99, "because": "Echo."}))
''')
        data["providers"]["cmd"] = {"transport": "command", "command": [sys.executable, "adapter.py"]}
        for model in data["models"].values():
            model.update(provider="cmd", model="exact/model", options={"reasoning_effort": "max"})
        self.write(data, '''maybe x: String = "hi"
print(x)
with model("careful") { print(x) }
with model("audit") { print(1 ?!= 2) }
print(x)''')
        status, out, err = self.cli("--project", str(self.path), cwd=Path("/tmp"))
        self.assertEqual(status, 0, err)
        records = [json.loads(line) for line in (self.root / "requests.jsonl").read_text().splitlines()]
        self.assertEqual([r["route"] for r in records], ["writing", "careful", "audit", "writing"])
        self.assertTrue(all(r["model"] == "exact/model" and r["model_options"] == {"reasoning_effort": "max"} for r in records))
        self.assertTrue(all("temperature_explicit" not in r for r in records))
        self.assertEqual([r["temperature"] for r in records], [.7, .2, .7, .7])

    def test_cli_explicit_command_default_override_still_allows_named_configured_block(self):
        self.write(source='maybe x = 1\nprint(x)\nwith model("writing") { print(x) }\nprint(x)')
        code = 'import sys,json; r=json.load(sys.stdin); print(json.dumps({"value":7,"confidence":.99,"because":"Override."}))'
        status, out, err = self.cli("--project", str(self.path), "--oracle-command", shlex.join([sys.executable, "-c", code]))
        self.assertEqual(status, 0, err)
        self.assertEqual(out, "7\n1\n7\n")


if __name__ == "__main__":
    unittest.main()
