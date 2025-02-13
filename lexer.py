from dataclasses import dataclass
from collections.abc import Iterator

# Base Token class
class Token:
    pass

# Token subclasses
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

# Defined categories for syntax elements
KEYWORDS = {
    "if", "else", "while", "for", "break", "continue",
    "func", "return", "let", "const", "print", "input", "int", "main", "float", "bool", "string"
}

OPERATORS = {"+",
    "-", "*", "/", "%", "**", "=", "==", "!=",
    "<", ">", "<=", ">=", "&&", "||", "!", "++", "--"
}

SEPARATORS = {
    "{", "}", "(", ")", "[", "]", ";", ":", ",", ".", "->",
    "'", '"', "//", "/*", "*/", " ", "\t", "\n"
}

CONSTANTS = {
    "true", "false"
}

# Lexer function
def lex(s: str) -> Iterator[Token]:
    i = 0
    while True:
        # Skip whitespace
        while i < len(s) and s[i].isspace():
            i += 1

        if i >= len(s):
            return

        # Skip block comments (/* ... */)
        if s[i:i+2] == "/*":
            i += 2  # Skip the opening /*
            while i < len(s) and s[i:i+2] != "*/":
                i += 1
            i += 2  # Skip the closing */
            continue

        if s[i].isdigit():  # Numbers
            t = s[i]
            i += 1
            while i < len(s) and s[i].isdigit():
                t += s[i]
                i += 1
            yield NumberToken(t)

        elif s[i] in "\"'":  # Strings
            quote = s[i]
            i += 1
            t = ''
            while i < len(s) and s[i] != quote:
                t += s[i]
                i += 1
            i += 1  # Skip the closing quote
            yield StringToken(t)

        elif s[i].isalpha() or s[i] == '_':  # Identifiers or Keywords
            t = s[i]
            i += 1
            while i < len(s) and (s[i].isalnum() or s[i] == '_'):
                t += s[i]
                i += 1
            if t in KEYWORDS:
                yield KeywordToken(t)
            elif t in CONSTANTS:
                yield ConstantToken(t)
            else:
                yield IdentifierToken(t)  # Allow any valid identifier

        elif s[i] in OPERATORS:  # Operators
            t = s[i]
            i += 1
            yield OperatorToken(t)

        elif s[i] in SEPARATORS:  # Separators
            t = s[i]
            i += 1
            yield SeparatorToken(t)

        else:  # Unrecognized characters
            raise ValueError(f"Unrecognized character: {s[i]}")

# Example usage
code = """
/* Arithmetic Operations */
int a = 5
int b = 10
int c = a + b
printk {c}

/* Printing Statements */
printk {"Hello, World!"}
printk {a}

/* Loops */
repeat 5:
    printk {"*"}

/* While Loop */
int counter = 1
while (counter <= 5):
    printk {"Counter: " + counter}
    counter = counter + 1

/* Conditionals */
if (a > b):
    printk {"a is greater"}
else:
    printk {"b is greater"}

/* Design Formation */
pattern triangle 3

/* Comments */
/* This is a comment */
a = 5

/8 Variables */
int x = 10
int y = x * 2
printk {y}

/* Basic Functions */
func sqr(n):
    return n * n
printk {sqr(4)}

/* Modulo Operation */
int m = 10
int n = 3
float mod_result = m % n
printk {mod_result}

/* String Operations */
str1 = "Hello"
str2 = "World"
result = str1 + " " + str2
printk {result}

/* Input from User */
name = input "Enter your name: "
printk {"Hello, " + name}

/* Array/List Operations */
arr = [1, 2, 3, 4, 5]
printk {arr[2]}

/* Math Functions */
float sqrt_result = sqrt(16)
printk {sqrt_result}

"""

for t in lex(code):
    print(t)
