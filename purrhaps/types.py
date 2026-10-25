"""Structural JSON types shared by the interpreter and model adapters.

Public helpers raise ValueError for malformed types/data; the runtime translates
these into its diagnosed language TypeError. Schemas describe provider output,
while validate_data also checks invariants JSON Schema cannot express here.
"""
from functools import lru_cache
import math
import re


ALIASES = {"Boolean": "Bool", "Double": "Float", "Long": "Int"}
RECORDS = {
    "Human": {"lastMessage": "String"},
    "Message": {"role": "String", "content": "String"},
    "Prompt": {"messages": "List<Message>"},
    "DecisionInput": {"question": "String", "options": "List<Json>", "context": "Json"},
    "TokenUsage": {"inputTokens": "Int", "outputTokens": "Int"},
    "ModelResponse": {"text": "String", "finishReason": "String", "usage": "TokenUsage"},
}
PRIMITIVES = {"Any", "Json", "Int", "Float", "String", "Bool", "Unit"}


@lru_cache(maxsize=256)
def parse_type(annotation):
    """Return (name, generic arguments, nullable), validating every nested type."""
    if not isinstance(annotation, str):
        raise ValueError("type annotation must be a string")
    tokens = re.findall(r"[A-Za-z_][A-Za-z_0-9]*|[<>,?]|\S", annotation)
    position = 0

    def parse():
        nonlocal position
        if position >= len(tokens) or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", tokens[position]):
            raise ValueError(f"invalid type annotation {annotation!r}")
        name = ALIASES.get(tokens[position], tokens[position])
        position += 1
        args = []
        if position < len(tokens) and tokens[position] == "<":
            position += 1
            args.append(parse())
            while position < len(tokens) and tokens[position] == ",":
                position += 1
                args.append(parse())
            if position >= len(tokens) or tokens[position] != ">":
                raise ValueError(f"invalid type annotation {annotation!r}")
            position += 1
        arity = {"List": 1, "Map": 2, "Pair": 2}.get(name)
        if name not in PRIMITIVES and name not in RECORDS and arity is None:
            raise ValueError(f"unknown type {name!r}")
        if args and (arity is None or len(args) != arity):
            raise ValueError(f"invalid type arguments for {name}")
        if name == "Map" and args and args[0] != ("String", (), False):
            raise ValueError("Map keys must be String")
        nullable = position < len(tokens) and tokens[position] == "?"
        if nullable:
            position += 1
        return name, tuple(args), nullable

    result = parse()
    if position != len(tokens):
        raise ValueError(f"invalid type annotation {annotation!r}")
    return result


def _canonical(spec):
    name, args, nullable = spec
    return name + ("<" + ",".join(map(_canonical, args)) + ">" if args else "") + ("?" if nullable else "")


def canonical_type(annotation):
    return _canonical(parse_type(annotation))


def is_json(data):
    """Finite JSON, with string object keys and no Python-only containers."""
    if data is None or type(data) in (str, bool, int):
        return True
    if type(data) is float:
        return math.isfinite(data)
    if type(data) is list:
        return all(is_json(item) for item in data)
    if type(data) is dict:
        return all(type(k) is str and is_json(v) for k, v in data.items())
    return False


def strict_equal(a, b):
    """JSON equality without Python's True == 1 or 1 == 1.0 coercion."""
    if type(a) is not type(b):
        return False
    if type(a) is list:
        return len(a) == len(b) and all(strict_equal(x, y) for x, y in zip(a, b))
    if type(a) is dict:
        return a.keys() == b.keys() and all(strict_equal(a[k], b[k]) for k in a)
    return a == b


def validate_data(data, annotation):
    """Validate recursive structural shape and constructor invariants, returning data."""
    spec = parse_type(annotation or "Any")

    def check(value, typ, path):
        name, args, nullable = typ
        if nullable and value is None:
            return
        expected = _canonical(typ)

        def fail(detail=None):
            raise ValueError(f"{path}: expected {expected}" + (f" ({detail})" if detail else f", got {type(value).__name__}"))

        if name in ("Any", "Json"):
            if not is_json(value):
                fail("finite JSON data required")
        elif name in ("Int", "Float", "String", "Bool", "Unit"):
            host = {"Int": int, "Float": float, "String": str, "Bool": bool, "Unit": type(None)}[name]
            if type(value) is not host or (name == "Float" and not math.isfinite(value)):
                fail()
        elif name == "List":
            if type(value) is not list:
                fail()
            for i, item in enumerate(value):
                check(item, args[0] if args else parse_type("Json"), f"{path}[{i}]")
        elif name == "Map":
            if type(value) is not dict or any(type(k) is not str for k in value):
                fail("String keys required")
            for key, item in value.items():
                check(item, args[1] if args else parse_type("Json"), f"{path}.{key}")
        else:
            fields = {"first": _canonical(args[0]) if args else "Json", "second": _canonical(args[1]) if args else "Json"} if name == "Pair" else RECORDS[name]
            if type(value) is not dict or value.keys() != fields.keys():
                fail("fields must be " + ", ".join(fields))
            for key, typ_name in fields.items():
                check(value[key], parse_type(typ_name), f"{path}.{key}")
            if name == "Message" and value["role"] not in ("system", "user", "assistant", "tool"):
                fail("role must be system, user, assistant or tool")
            if name == "Prompt" and not value["messages"]:
                fail("messages must be nonempty")
            if name == "TokenUsage" and any(v < 0 for v in value.values()):
                fail("token counts must be nonnegative")
            if name == "DecisionInput":
                if not value["question"].strip():
                    fail("question must be nonempty")
                options = value["options"]
                if len(options) < 2:
                    fail("at least two options required")
                if any(strict_equal(item, earlier) for i, item in enumerate(options) for earlier in options[:i]):
                    fail("options must be unique under strict equality")
    check(data, spec, "value")
    return data


def type_schema(annotation):
    """JSON Schema for a canonical/aliased annotation (runtime checks still apply)."""
    name, args, nullable = parse_type(annotation or "Any")
    if name in ("Any", "Json"):
        result = {}
    elif name in ("Int", "Float", "String", "Bool", "Unit"):
        result = {"type": {"Int": "integer", "Float": "number", "String": "string", "Bool": "boolean", "Unit": "null"}[name]}
    elif name == "List":
        result = {"type": "array", "items": type_schema(_canonical(args[0])) if args else {}}
    elif name == "Map":
        result = {"type": "object", "additionalProperties": type_schema(_canonical(args[1])) if args else {}}
    else:
        fields = {"first": _canonical(args[0]) if args else "Json", "second": _canonical(args[1]) if args else "Json"} if name == "Pair" else RECORDS[name]
        result = {"type": "object", "properties": {k: type_schema(v) for k, v in fields.items()}, "required": list(fields), "additionalProperties": False}
        if name == "Message":
            result["properties"]["role"]["enum"] = ["system", "user", "assistant", "tool"]
        elif name == "Prompt":
            result["properties"]["messages"]["minItems"] = 1
        elif name == "DecisionInput":
            result["properties"]["question"]["minLength"] = 1
            result["properties"]["options"].update(minItems=2)
        elif name == "TokenUsage":
            for schema in result["properties"].values():
                schema["minimum"] = 0
    return {"anyOf": [result, {"type": "null"}]} if nullable else result
