"""A small lexer and Pratt parser. No Python eval, no translated host code."""
from dataclasses import dataclass
import json
import re


class PurrhapsError(Exception):
    """A diagnosed language/runtime failure."""


class SyntaxError(PurrhapsError):
    pass


@dataclass(frozen=True)
class Token:
    kind: str
    text: str
    line: int
    column: int


@dataclass(frozen=True)
class Node:
    kind: str
    args: tuple
    line: int


_PATTERN = re.compile(
    r'(?P<SPACE>[^\S\n]+)|(?P<COMMENT>//[^\n]*)|(?P<NL>\n)|'
    r'(?P<STRING>"(?:[^"\\\n]|\\.)*")|(?P<NUMBER>\d+(?:\.\d+)?)|'
    r'(?P<ID>[A-Za-z_][A-Za-z_0-9]*)|'
    r'(?P<OP>\?!=|\?=|!!|~=|==|!=|<=|>=|&&|\|\||[\[\]{}(),.:;=+*/%<>!?\-])'
)


def lex(source):
    tokens, pos, line, column = [], 0, 1, 1
    while pos < len(source):
        match = _PATTERN.match(source, pos)
        if not match:
            raise SyntaxError(f"line {line}:{column}: unexpected character {source[pos]!r}")
        kind, text = match.lastgroup, match.group()
        if kind not in ("SPACE", "COMMENT"):
            tokens.append(Token(kind, text, line, column))
        pos = match.end()
        if kind == "NL":
            line, column = line + 1, 1
        else:
            column += len(text)
    tokens.append(Token("EOF", "<eof>", line, column))
    return tokens


class Parser:
    PRECEDENCE = {"||": 1, "&&": 2, "==": 3, "!=": 3, "~=": 3, "?!=": 3,
                  "<": 4, ">": 4, "<=": 4, ">=": 4, "to": 4,
                  "+": 5, "-": 5, "*": 6, "/": 6, "%": 6}

    def __init__(self, source):
        self.tokens, self.i = lex(source), 0

    @property
    def token(self):
        return self.tokens[self.i]

    def take(self, text=None):
        token = self.token
        if token.kind == "EOF":
            self.fail("unexpected end of file")
        if text is not None and token.text != text:
            self.fail(f"expected {text!r}, got {token.text!r}")
        self.i += 1
        return token

    def fail(self, message):
        t = self.token
        raise SyntaxError(f"line {t.line}:{t.column}: {message}")

    def skip(self):
        while self.token.kind == "NL" or self.token.text == ";":
            self.take()

    def program(self, end="<eof>"):
        statements = []
        self.skip()
        while self.token.text != end:
            if self.token.kind == "EOF":
                self.fail(f"expected {end!r}")
            statements.append(self.statement())
            if self.token.kind not in ("NL", "EOF") and self.token.text not in (";", end):
                self.fail("expected a newline or ';' between statements")
            self.skip()
        return statements

    def block(self):
        self.take("{")
        body = self.program("}")
        self.take("}")
        return body

    def identifier(self):
        if self.token.kind != "ID":
            self.fail("expected an identifier")
        return self.take().text

    def annotation(self):
        if self.token.text == ":":
            self.take()
            return self.type_annotation()
        return None

    def type_annotation(self):
        aliases = {"Boolean": "Bool", "Double": "Float", "Long": "Int"}
        name = self.identifier()
        result = aliases.get(name, name)
        if self.token.text == "<":
            self.take()
            params = [self.type_annotation()]
            while self.token.text == ",":
                self.take()
                params.append(self.type_annotation())
            # The lexer uses maximal munch for comparisons. In a type argument
            # list, however, >= may close the annotation and start assignment.
            if self.token.text == ">=":
                token = self.token
                self.tokens[self.i:self.i + 1] = [
                    Token("OP", ">", token.line, token.column),
                    Token("OP", "=", token.line, token.column + 1),
                ]
            self.take(">")
            result += "<" + ",".join(params) + ">"
        # Preserve the original adjacent nullable assignment `x: Int?=null`.
        # Escalation after an annotation separates its operator with whitespace:
        # `x: Int ?= ...`, or `x: Int? ?= ...` for a nullable escalation result.
        if self.token.text == "?=":
            token, previous = self.token, self.tokens[self.i - 1]
            if token.line == previous.line and token.column == previous.column + len(previous.text):
                self.tokens[self.i:self.i + 1] = [
                    Token("OP", "?", token.line, token.column),
                    Token("OP", "=", token.line, token.column + 1),
                ]
        if self.token.text == "?":
            self.take()
            result += "?"
        return result

    def statement(self):
        t = self.token
        if t.text in ("maybe", "perhaps", "definitely", "val"):
            mode = self.take().text
            name = self.identifier()
            typ = self.annotation()
            if self.token.text == "?=":
                if mode != "maybe":
                    self.fail("only maybe declarations accept ?=")
                self.take()
                return Node("escalate", (name, typ, self.expr()), t.line)
            self.take("=")
            return Node("declare", (mode, name, typ, self.expr()), t.line)
        if t.text == "fun":
            self.take()
            name = self.identifier()
            self.take("(")
            params = []
            while self.token.text != ")":
                param = self.identifier()
                params.append((param, self.annotation()))
                if self.token.text != ",":
                    break
                self.take()
            self.take(")")
            typ = self.annotation()
            return Node("function", (name, params, typ, self.block()), t.line)
        if t.text == "if":
            self.take()
            self.take("(")
            cond = self.expr()
            self.take(")")
            yes = self.block() if self.token.text == "{" else [self.statement()]
            saved = self.i
            self.skip()
            no = []
            if self.token.text == "else":
                self.take()
                no = self.block() if self.token.text == "{" else [self.statement()]
            else:
                self.i = saved
            return Node("if", (cond, yes, no), t.line)
        if t.text == "with":
            self.take()
            if self.token.text not in ("temperature", "model"):
                self.fail("expected 'temperature' or 'model' after 'with'")
            setting = self.take().text
            self.take("(")
            value = self.expr()
            self.take(")")
            return Node(setting, (value, self.block()), t.line)
        if t.text == "return":
            self.take()
            return Node("return", (self.expr(),), t.line)
        return Node("expression", (self.expr(),), t.line)

    def expr(self, minimum=0):
        t = self.take()
        if t.kind == "NUMBER":
            left = Node("literal", (float(t.text) if "." in t.text else int(t.text),), t.line)
        elif t.kind == "STRING":
            try:
                value = json.loads(t.text)
            except ValueError as exc:
                raise SyntaxError(f"line {t.line}: invalid string escape") from exc
            left = Node("literal", (value,), t.line)
        elif t.text in ("true", "false", "null"):
            left = Node("literal", ({"true": True, "false": False, "null": None}[t.text],), t.line)
        elif t.text in ("-", "!"):
            left = Node("unary", (t.text, self.expr(7)), t.line)
        elif t.text == "(":
            left = self.expr()
            self.take(")")
        elif t.text == "[":
            items = []
            while self.token.text != "]":
                items.append(self.expr())
                if self.token.text != ",":
                    break
                self.take()
            self.take("]")
            left = Node("list", (items,), t.line)
        elif t.kind == "ID":
            left = Node("name", (t.text,), t.line)
        else:
            self.fail(f"expected expression, got {t.text!r}")
        while True:
            op = self.token.text
            if op == "!!":
                self.take()
                left = Node("collapse", (left,), t.line)
            elif op == ".":
                self.take()
                left = Node("attribute", (left, self.identifier()), t.line)
            elif op == "[":
                self.take()
                index = self.expr()
                self.take("]")
                left = Node("index", (left, index), t.line)
            elif op == "(":
                self.take()
                args = []
                while self.token.text != ")":
                    args.append(self.expr())
                    if self.token.text != ",":
                        break
                    self.take()
                self.take(")")
                left = Node("call", (left, args), t.line)
            elif op in self.PRECEDENCE and self.PRECEDENCE[op] >= minimum:
                self.take()
                right = self.expr(self.PRECEDENCE[op] + 1)
                left = Node("binary", (op, left, right), t.line)
            else:
                break
        return left
