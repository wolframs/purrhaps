"""Deterministic Kotlin-shaped operations; no oracle access or mutable values.

Strings use Unicode codepoint indices and Python Unicode case conversion.
All operations work on concrete JSON; the interpreter lifts uncertainty around
one invocation and translates ValueError/TypeError into language diagnostics.
"""
import json
import math
import re
from .types import RECORDS, strict_equal, validate_data


def display(data):
    if data is None:
        return "null"
    if type(data) is bool:
        return "true" if data else "false"
    if type(data) in (dict, list):
        return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return str(data)


def require(test, message):
    if not test:
        raise ValueError(message)


def arity(name, args, minimum, maximum=None):
    maximum = minimum if maximum is None else maximum
    require(minimum <= len(args) <= maximum, f"{name} expects {minimum}" + (f"..{maximum}" if minimum != maximum else "") + " arguments")


def number(value):
    require(type(value) in (int, float), "a number is required")
    return value


def integer(value):
    require(type(value) is int, "an Int is required")
    return value


def string(value):
    require(type(value) is str, "a String is required")
    return value


def index(data, key, nullable=False):
    if type(data) is dict:
        string(key)
        return data.get(key)
    require(type(data) in (list, str), "indexing requires a List, String or Map")
    integer(key)
    if nullable and (key < 0 or key >= len(data)):
        return None
    require(0 <= key < len(data), "index out of bounds (indices must be nonnegative)")
    return data[key]


def attribute(data, name):
    if name == "size" and type(data) in (dict, list):
        return len(data)
    if name == "length" and type(data) is str:
        return len(data)
    if name in ("keys", "values") and type(data) is dict:
        keys = sorted(data)
        return keys if name == "keys" else [data[k] for k in keys]
    if type(data) is dict and name in data:
        return data[name]
    raise ValueError(f"no field {name!r}")


INTRINSICS = {"listOf", "emptyList", "mapOf", "emptyMap", "Pair", "abs", "minOf", "maxOf", *RECORDS}


def intrinsic(name, args):
    if name == "listOf":
        return list(args)
    if name in ("emptyList", "emptyMap"):
        arity(name, args, 0)
        return [] if name == "emptyList" else {}
    if name == "Pair":
        arity(name, args, 2)
        return {"first": args[0], "second": args[1]}
    if name == "mapOf":
        result = {}
        for pair in args:
            validate_data(pair, "Pair<String,Json>")
            result[pair["first"]] = pair["second"]
        return result
    if name in RECORDS:
        fields = RECORDS[name]
        arity(name, args, len(fields))
        return validate_data(dict(zip(fields, args)), name)
    if name == "abs":
        arity(name, args, 1)
        return abs(number(args[0]))
    if name in ("minOf", "maxOf"):
        require(len(args) >= 2, f"{name} expects at least two arguments")
        for arg in args:
            number(arg)
        return (min if name == "minOf" else max)(args)
    raise ValueError(f"unknown intrinsic {name!r}")


def method(data, name, args):
    if name == "toString":
        arity(name, args, 0)
        return display(data)
    if name in ("isEmpty", "isNotEmpty", "contains"):
        require(type(data) in (str, list, dict), f"{name} requires List, Map or String")
        arity(name, args, 1 if name == "contains" else 0)
        if name != "contains":
            return (len(data) == 0) if name == "isEmpty" else (len(data) != 0)
        if type(data) in (dict, str):
            return string(args[0]) in data
        return any(strict_equal(args[0], item) for item in data)
    if name in ("first", "last", "get", "getOrNull"):
        require(type(data) in (list, str), f"{name} requires List or String")
        arity(name, args, 1 if name in ("get", "getOrNull") else 0)
        key = args[0] if args else (0 if name == "first" else len(data) - 1)
        return index(data, key, nullable=name == "getOrNull")
    if type(data) is list:
        if name == "joinToString":
            arity(name, args, 0, 1)
            sep = string(args[0]) if args else ", "
            return sep.join(display(item) for item in data)
        if name in ("distinct", "reversed", "sorted", "sum"):
            arity(name, args, 0)
            if name == "distinct":
                result = []
                for item in data:
                    if not any(strict_equal(item, existing) for existing in result):
                        result.append(item)
                return result
            if name == "reversed":
                return list(reversed(data))
            if name == "sum":
                for item in data:
                    number(item)
                return sum(data)
            require(all(type(v) in (int, float) for v in data) or all(type(v) is str for v in data), "sorted requires numbers only or Strings only")
            return sorted(data)
        if name in ("take", "drop"):
            arity(name, args, 1)
            count = integer(args[0])
            require(count >= 0, "count must be nonnegative")
            return data[:count] if name == "take" else data[count:]
    if type(data) is str:
        if name in ("trim", "lowercase", "uppercase", "toIntOrNull", "toDoubleOrNull"):
            arity(name, args, 0)
            if name == "trim":
                return data.strip()
            if name == "lowercase":
                return data.lower()
            if name == "uppercase":
                return data.upper()
            text = data.strip()
            if name == "toIntOrNull":
                if not re.fullmatch(r"[+-]?[0-9]+", text):
                    return None
                # Avoid Python's integer parsing digit limit for enormous input.
                digits = text.lstrip("+-").lstrip("0") or "0"
                if len(digits) > 19:
                    return None
                result = int(("-" if text.startswith("-") else "") + digits)
                return result if -(2 ** 63) <= result <= 2 ** 63 - 1 else None
            if not re.fullmatch(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?", text):
                return None
            result = float(text)
            return result if math.isfinite(result) else None
        if name in ("startsWith", "endsWith", "split"):
            arity(name, args, 1)
            text = string(args[0])
            if name == "startsWith":
                return data.startswith(text)
            if name == "endsWith":
                return data.endswith(text)
            require(bool(text), "split separator must be nonempty")
            return data.split(text)
        if name == "replace":
            arity(name, args, 2)
            return data.replace(string(args[0]), string(args[1]))
        if name == "substring":
            arity(name, args, 1, 2)
            start = integer(args[0])
            end = integer(args[1]) if len(args) == 2 else len(data)
            require(0 <= start <= end <= len(data), "substring bounds must satisfy 0 <= start <= end <= length")
            return data[start:end]
    if type(data) in (int, float):
        if name == "abs":
            arity(name, args, 0)
            return abs(data)
        if name == "coerceIn":
            arity(name, args, 2)
            low, high = map(number, args)
            require(low <= high, "coerceIn minimum must not exceed maximum")
            return min(max(data, low), high)
    raise ValueError(f"no method {name!r} for {type(data).__name__}")
