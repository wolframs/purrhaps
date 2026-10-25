"""python -m purrhaps path/to/cat.prr"""
import argparse
import math
from pathlib import Path
import shlex
import sys
from . import __version__
from .oracle import CommandOracle
from .project import ConfigurationError, discover_project, load_project
from .runtime import Interpreter
from .syntax import Parser, PurrhapsError
from .transpile import transpile


def main(argv=None):
    parser = argparse.ArgumentParser(description="Purrhaps: an experimental language of uncertain cats.")
    parser.add_argument("file", type=Path, nargs="?", help=".prr source file (use - for stdin; defaults to project entry)")
    parser.add_argument("--version", action="version", version=f"Purrhaps {__version__}")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="parse only; never call the oracle")
    mode.add_argument("--emit-rust", type=Path, metavar="DIRECTORY", help="transpile into a new native Rust Cargo project")
    parser.add_argument("--seed", type=int, default=0, help="seed for the offline demo and random escalation terminal")
    parser.add_argument("--oracle-command", help="JSON oracle command, split into arguments without a shell")
    parser.add_argument("--decision-oracle-command", help="bounded-decision JSON oracle command, split without a shell")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--project", type=Path, metavar="FILE_OR_DIR", help="select a purrhaps.toml manifest")
    selection.add_argument("--no-project", action="store_true", help="disable project discovery")
    parser.add_argument("--model", metavar="ROUTE", help="default generative model route")
    parser.add_argument("--decision-model", metavar="ROUTE", help="default decision model route")
    parser.add_argument("--timeout", type=float, default=None, help="seconds per oracle invocation (overrides project timeouts)")
    parser.add_argument("--max-calls", type=int, default=None, help="maximum oracle calls (overrides project budget)")
    parser.add_argument("--stats", action="store_true", help="report oracle call count on stderr")
    options = parser.parse_args(argv)
    if options.timeout is not None and (not math.isfinite(options.timeout) or not 0 < options.timeout <= 86400):
        parser.error("timeout must be finite, positive and at most 86400 seconds")
    if options.max_calls is not None and options.max_calls < 1:
        parser.error("max-calls must be positive")
    if options.model is not None and options.oracle_command is not None:
        parser.error("--model conflicts with --oracle-command")
    if options.decision_model is not None and options.decision_oracle_command is not None:
        parser.error("--decision-model conflicts with --decision-oracle-command")
    try:
        project_path = options.project
        if project_path is None and not options.no_project:
            start = (options.file.resolve().parent if options.file is not None and str(options.file) != "-"
                     else Path.cwd())
            project_path = discover_project(start)
        project = load_project(project_path) if project_path is not None else None
        if options.model is not None:
            if project is None:
                raise ConfigurationError("--model requires a project manifest")
            project.route(options.model, "generative")
        if options.decision_model is not None:
            if project is None:
                raise ConfigurationError("--decision-model requires a project manifest")
            project.route(options.decision_model, "decision")
        if options.file is None:
            if project is None or "entry" not in project.data["project"]:
                raise ConfigurationError("supply a source file or configure a project entry")
            options.file = project.root / project.data["project"]["entry"]
        source = sys.stdin.read() if str(options.file) == "-" else options.file.read_text(encoding="utf-8")
        if options.emit_rust:
            destination = (transpile(source, options.emit_rust, project=project) if project is not None
                           else transpile(source, options.emit_rust))
            print(f"Rust project written to {destination}")
            return 0
        if options.check:
            Parser(source).program()
            print(f"{options.file}: syntax is purrfect")
            return 0
        timeout = (options.timeout if options.timeout is not None else
                   project.data.get("runtime", {}).get("timeout", 30) if project is not None else 30)
        oracle = CommandOracle(shlex.split(options.oracle_command), timeout) if options.oracle_command is not None else None
        decision_oracle = (CommandOracle(shlex.split(options.decision_oracle_command), timeout)
                           if options.decision_oracle_command is not None else None)
        runtime = Interpreter(oracle, decision_oracle=decision_oracle, max_calls=options.max_calls,
                              project=project, generative_route=options.model, decision_route=options.decision_model,
                              seed=options.seed, timeout_override=options.timeout)
        try:
            runtime.run(source)
        finally:
            if options.stats:
                print(f"oracle={type(runtime.oracle).__name__} decision_oracle={type(runtime.decision_oracle).__name__} calls={runtime.calls}", file=sys.stderr)
        return 0
    except (PurrhapsError, OSError, UnicodeError, ValueError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
