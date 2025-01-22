from dataclasses import dataclass
from collections.abc import Iterator
import re

# Lexer
class Token:
    pass

@dataclass
class KeywordToken(Token):
    k: str

@dataclass
class OperatorToken(Token):
    o: str

@dataclass
class ConstantToken(Token):
    c: str

@dataclass
class SeparatorToken(Token):
    s: str

@dataclass
class IdentifierToken(Token):
    i: str

@dataclass
class NumberToken(Token):
    v: str

@dataclass
class StringToken(Token):
    s: str

KEYWORDS = {
    "if", "else", "while", "for", "break", "continue", 
    "function", "return", "let", "const", "print", "input", "int", "main"
}

OPERATORS = {
    "+", "-", "*", "/", "%", "**", "=", "==", "!=",
    "<", ">", "<=", ">=", "&&", "||", "!", "++", "--"
}

SEPARATORS = {
    "{", "}", "(", ")", "[", "]", ";", ":", ",", ".", "->", 
    "'", '"', "//", "/*", "*/", " ", "\t", "\n"
}

IDENTIFIERS = {
    "a", "b"
}

CONSTANTS = {
    "true", "false"
}

def lex(s: str) -> Iterator[Token]:
    i = 0
    while True:
        while i < len(s) and s[i].isspace():
            i += 1

        if i >= len(s):
            return

        if s[i].isdigit():
            t = s[i]
            i += 1
            while i < len(s) and s[i].isdigit():
                t += s[i]
                i += 1
            yield NumberToken(t)

        elif s[i] in "\"'":
            quote = s[i]
            i += 1
            t = ''
            while i < len(s) and s[i] != quote:
                t += s[i]
                i += 1
            i += 1  # Skip the closing quote
            yield StringToken(t)

        elif s[i].isalpha() or s[i] == '_':
            t = s[i]
            i += 1
            while i < len(s) and (s[i].isalnum() or s[i] == '_'):
                t += s[i]
                i += 1
            if t in KEYWORDS:
                yield KeywordToken(t)
            else:
                yield IdentifierToken(t)

        elif s[i] in OPERATORS:
            t = s[i]
            i += 1
            yield OperatorToken(t)

        elif s[i] in SEPARATORS:
            t = s[i]
            i += 1
            yield SeparatorToken(t)

        else:
            i += 1  # Skip unrecognized characters

for t in lex("int main() { /* find max of a and b */ int a=20, b=30; if(a<b) return(b); else return(a);}"):
    print(t)
