"""Lower the Purrhaps syntax tree into Rust functions and a native Cargo project."""
import json
import math
from pathlib import Path
from .syntax import Parser, PurrhapsError


def string(value):
    # Rust raw strings preserve Unicode and JSON/Purrhaps escapes verbatim.
    marks = '#'
    while '"' + marks in value:
        marks += '#'
    return f'r{marks}"{value}"{marks}'


class RustEmitter:
    def __init__(self):
        self.functions = []
        self.serial = 0

    def function(self, prefix, result, code):
        self.serial += 1
        name = f'{prefix}_{self.serial}'
        self.functions.append(f'fn {name}(rt: &mut Rt, scope: usize) -> Res<{result}> {{\n{code}\n}}')
        return name

    def expression(self, node):
        kind, args = node.kind, node.args
        code = []
        def sub(expr, variable):
            name = self.expression(expr)
            code.append(f'    let {variable} = rt.eval({name}, scope)?;')
        if kind == 'literal':
            value = args[0]
            if type(value) is float and not math.isfinite(value):
                raise PurrhapsError(f'line {node.line}: Rust backend requires finite Float literals')
            if type(value) is int and not -(2**63) <= value < 2**63:
                raise PurrhapsError(f'line {node.line}: Rust backend Int literals must fit signed 64-bit')
            # JSON decoding occurs only for literal data, never code or syntax trees.
            result = f'literal(serde_json::from_str({string(json.dumps(value, ensure_ascii=False))}).expect("compiler literal"), {node.line})'
            code.append(f'    Ok({result})')
        elif kind == 'name':
            code.append(f'    rt.name(scope, {string(args[0])})')
        elif kind == 'collapse':
            sub(args[0], 'value')
            code.append('    rt.collapse(&value, scope)')
        elif kind == 'attribute':
            sub(args[0], 'value')
            code.append(f'    rt.attribute(scope, value, {string(args[1])})')
        elif kind == 'unary':
            if args[0] == '-' and args[1].kind == 'literal' and type(args[1].args[0]) is int and args[1].args[0] == 2**63:
                # The parser represents the signed minimum as unary minus plus
                # a positive literal that is otherwise outside the native range.
                code.append(f'    Ok(literal(serde_json::json!(i64::MIN), {node.line}))')
                return self.function('expr', 'V', '\n'.join(code))
            sub(args[1], 'value')
            code.append(f'    rt.unary({string(args[0])}, value)')
        elif kind == 'binary':
            sub(args[1], 'left')
            sub(args[2], 'right')
            code.append(f'    rt.binary({string(args[0])}, left, right, {node.line})')
        elif kind == 'list':
            expressions = args[0]
            for i, expr in enumerate(expressions):
                sub(expr, f'item_{i}')
            code.append(f'    rt.list(vec![{", ".join(f"item_{i}" for i in range(len(expressions)))}])')
        elif kind == 'index':
            sub(args[0], 'receiver')
            sub(args[1], 'index')
            code.append('    rt.index(receiver, index)')
        elif kind == 'call':
            target, expressions = args
            if target.kind not in ('name', 'attribute'):
                raise PurrhapsError(f'line {node.line}: only named functions and member methods can be called')
            if target.kind == 'name' and target.args[0] == 'sample' and len(expressions) != 1:
                code.append('    rt.call(scope, "sample", vec![])')
                return self.function('expr', 'V', '\n'.join(code))
            if target.kind == 'attribute':
                sub(target.args[0], 'receiver')
            for i, expr in enumerate(expressions):
                sub(expr, f'arg_{i}')
            arguments = ", ".join(f"arg_{i}" for i in range(len(expressions)))
            if target.kind == 'attribute':
                code.append(f'    rt.method(receiver, {string(target.args[1])}, vec![{arguments}])')
            else:
                code.append(f'    rt.call(scope, {string(target.args[0])}, vec![{arguments}])')
        else:
            raise PurrhapsError(f'line {node.line}: unsupported Rust expression {kind}')
        return self.function('expr', 'V', '\n'.join(code))

    def statement(self, node):
        kind, args = node.kind, node.args
        if kind == 'declare':
            mode, name, typ, expr = args
            code = f'rt.declare(scope, {string(mode)}, {string(name)}, {string(typ or "")}, {self.expression(expr)}, {node.line})?;\nOk(None)'
        elif kind == 'escalate':
            name, typ, expr = args
            code = f'rt.escalate(scope, {string(name)}, {string(typ or "")}, {self.expression(expr)}, {node.line})?;\nOk(None)'
        elif kind == 'function':
            name, params, typ, body = args
            parameters = ', '.join(f'({string(n)}, {string(t or "")})' for n, t in params)
            code = f'rt.function(scope, {string(name)}, vec![{parameters}], {string(typ or "")}, {self.block(body)})?;\nOk(None)'
        elif kind in ('return', 'expression'):
            code = f'let value = rt.eval({self.expression(args[0])}, scope)?;\n'
            code += 'Ok(Some(value))' if kind == 'return' else 'let _ = value; Ok(None)'
        elif kind == 'if':
            cond, yes, no = args
            code = f'let value = rt.eval({self.expression(cond)}, scope)?;\nrt.branch(scope, value, {self.block(yes)}, {self.block(no)})'
        elif kind in ('temperature', 'model'):
            code = f'let value = rt.eval({self.expression(args[0])}, scope)?;\nrt.{kind}(scope, value, {self.block(args[1])})'
            if kind == 'model':
                code = 'rt.require_project()?;\n' + code
        else:
            raise PurrhapsError(f'line {node.line}: unsupported Rust statement {kind}')
        code = f'    // Purrhaps source line {node.line}\n    let result = (|| -> Res<Option<V>> {{\n{code}\n    }})();\n    located(result, {node.line})'
        return self.function('stmt', 'Option<V>', code)

    def block(self, nodes):
        calls = [f'    if let Some(value) = {self.statement(n)}(rt, scope)? {{ return Ok(Some(value)); }}' for n in nodes]
        return self.function('block', 'Option<V>', '\n'.join(calls + ['    Ok(None)']))

    def emit(self, source):
        entry = self.block(Parser(source).program())
        return ('// Generated by Purrhaps. Native Rust; no Python runtime required.\n'
                '#![allow(unused_variables)]\nmod runtime;\nuse runtime::*;\n\n'
                + '\n\n'.join(self.functions)
                + f'\n\nfn main() {{ launch({string(source)}, {entry}); }}\n')



def _manifest_value(value):
    """Emit validated JSON-shaped TOML without rereading the source manifest."""
    if isinstance(value, dict):
        return '{' + ', '.join(_manifest_value(key) + ' = ' + _manifest_value(item)
                               for key, item in value.items()) + '}'
    if isinstance(value, list):
        return '[' + ', '.join(map(_manifest_value, value)) + ']'
    if type(value) in (str, bool, int, float):
        return json.dumps(value, ensure_ascii=False, allow_nan=False).replace(chr(127), r'\u007f')
    raise PurrhapsError('project manifest cannot be emitted as TOML')


def transpile(source, destination, project=None):
    """Create a new Cargo project; never overwrite an existing destination."""
    generated = RustEmitter().emit(source)
    manifest_source = ('\n'.join(_manifest_value(key) + ' = ' + _manifest_value(value)
                                 for key, value in project.data.items()) + '\n'
                       if project is not None else None)
    destination = Path(destination)
    if destination.exists():
        raise PurrhapsError(f'Rust output directory already exists: {destination}; choose a new path')
    runtime_dir = Path(__file__).with_name('rust')
    destination.mkdir(parents=True)
    (destination / 'src').mkdir()
    (destination / 'Cargo.toml').write_text(
        '[package]\nname = "purrhaps-program"\nversion = "0.1.0"\nedition = "2021"\n\n'
        '[dependencies]\nserde_json = "1.0"\ntoml = "0.8"\nurl = "2"\n\n[workspace]\n', encoding='utf-8')
    (destination / 'src/main.rs').write_text(generated, encoding='utf-8')
    for runtime_file in runtime_dir.glob('*.rs'):
        (destination / 'src' / runtime_file.name).write_text(runtime_file.read_text(encoding='utf-8'), encoding='utf-8')
    if manifest_source is not None:
        (destination / 'purrhaps.toml').write_text(manifest_source, encoding='utf-8')
    (destination / '.gitignore').write_text('/target/\n', encoding='utf-8')
    return destination
