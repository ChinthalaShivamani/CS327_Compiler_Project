# Klang Programming Language

Klang is a modern, interpreted programming language with clean syntax and strong typing features. This README provides an overview of the language and its implementation.

## Table of Contents
- Introduction
- Features
- Getting Started
- Language Overview
- Examples
- REPL Environment
- Implementation

## Introduction

Klang is a high-level programming language designed to be expressive, easy to read, and well-structured. It features a block-based syntax with clear beginning and ending markers, strong typing, and support for both procedural and functional programming paradigms.

## Features

- **Strong Type System**
  - Basic types: `int`, `float`, `string`
  - Explicit variable declarations with types
  - Optional mutability modifiers (`const`, `var`)
  - Type checking and enforcement at runtime

- **Control Flow**
  - `if`/`elif`/`else` conditional statements
  - `while` loops for condition-based iteration
  - `for` loops for iterating over collections
  - `repeat` loops for executing code a specific number of times

- **Functions**
  - Named functions with parameter types
  - Return values
  - Support for array parameters

- **Data Structures**
  - Arrays with literal notation `[1, 2, 3]`
  - Array methods (`add`, `remove`)
  - Array indexing

- **String Handling**
  - Double-quoted string literals
  - Multiline string syntax with pipe (`|`) delimiters
  - String concatenation
  - Escape sequences (`\n`, etc.)

- **Operators**
  - Arithmetic: `+`, `-`, `*`, `/`, `%`
  - Comparison: `==`, `<`, `>`, `<=`, `>=`
  - Compound assignment: `+=`, `-=`, `*=`, `/=`, `%=`

- **Block Structure**
  - Clear `begin` and `end` block delimiters
  - Enforced block structure

- **Input/Output**
  - Built-in `printk` function for output

## Getting Started

### Running a Klang Program

```bash
python main.py example.klang
```

### Using the REPL

```bash
python repl.py
```

## Language Overview

### Variable Declaration and Assignment

```
int x = 10
float pi = 3.14
string greeting = "Hello, World!"
const int MAX_VALUE = 100  # Constant variable
```

### Arrays

```
int numbers[5] = [1, 2, 3, 4, 5]
string names[] = ["Alice", "Bob", "Charlie"]

# Array methods
numbers.add(6)         # Add element at end
numbers.add(0, 42)     # Insert at index 0
numbers.remove(1)      # Remove element at index 1
```

### Control Flow

#### If-Elif-Else Statement

```
if (condition) begin
    # statements
elif (another_condition) begin
    # statements
else begin
    # statements
end
```

#### While Loop

```
while (condition) begin
    # statements
end
```

#### For Loop

```
for item in collection begin
    # statements
end
```

#### Repeat Loop

```
repeat 5 begin
    # statements to repeat 5 times
end
```

### Functions

```
func add(int a, int b) begin
    return a + b
end

int result = add(5, 10)
```

### Multiline Strings

```
string poem = |
    Roses are red
    Violets are blue
    This is a multiline string
    In Klang for you
|
```

### Printing

```
printk{"Hello, World!"}
printk{variable}
printk{expression + 5}
```

## Examples

See the included `example.klang` file for a comprehensive demonstration of language features.

## REPL Environment

The REPL environment supports:

- Interactive code execution
- Special commands:
  - `help` - Show help information
  - `vars` - Display defined variables
  - `exit` or `quit` - Exit the REPL
  - `clear` - Clear the screen
  - `load <filename>` - Load and execute a Klang file

## Implementation

The Klang interpreter consists of several components:

1. **lexer.py**: Tokenizes source code into meaningful tokens
2. **parser.py**: Converts tokens into an abstract syntax tree (AST)
3. **ast_nodes.py**: Defines classes for AST representation
4. **evaluator.py**: Executes the AST and manages program state
5. **main.py**: Entry point for running Klang programs
6. **repl.py**: Interactive read-eval-print loop for Klang

The interpreter follows a standard compilation pipeline:
Source code → Lexical analysis → Parsing → Evaluation
