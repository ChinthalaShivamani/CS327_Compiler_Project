import re
import math
import sys

# Define AST nodes
class ASTNode:
    pass

class Program(ASTNode):
    def __init__(self, statements):
        self.statements = statements

    def __repr__(self):
        return f"Program({self.statements})"

class VariableDeclaration(ASTNode):
    def __init__(self, var_type, name, value, is_array=False, array_size=None):
        self.var_type = var_type
        self.name = name
        self.value = value
        self.is_array = is_array
        self.array_size = array_size

    def __repr__(self):
        return f"VariableDeclaration({self.var_type}, {self.name}, {self.value}, {self.is_array}, {self.array_size})"

class PrintStatement(ASTNode):
    def __init__(self, expression):
        self.expression = expression

    def __repr__(self):
        return f"PrintStatement({self.expression})"

class RepeatLoop(ASTNode):
    def __init__(self, count, body):
        self.count = count
        self.body = body

    def __repr__(self):
        return f"RepeatLoop({self.count}, {self.body})"

class BinaryOperation(ASTNode):
    def __init__(self, left, operator, right):
        self.left = left
        self.operator = operator
        self.right = right

    def __repr__(self):
        return f"BinaryOperation({self.left}, {self.operator}, {self.right})"

class Expression(ASTNode):
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"Expression({self.value})"

class IfElse(ASTNode):
    def __init__(self, condition, true_branch, elif_branches, false_branch):
        self.condition = condition
        self.true_branch = true_branch
        self.elif_branches = elif_branches
        self.false_branch = false_branch

    def __repr__(self):
        return f"IfElse({self.condition}, {self.true_branch}, {self.elif_branches}, {self.false_branch})"

class WhileLoop(ASTNode):
    def __init__(self, condition, body):
        self.condition = condition
        self.body = body

    def __repr__(self):
        return f"WhileLoop({self.condition}, {self.body})"

class ForLoop(ASTNode):
    def __init__(self, loop_var, iterable, body):
        self.loop_var = loop_var
        self.iterable = iterable
        self.body = body

    def __repr__(self):
        return f"ForLoop({self.loop_var}, {self.iterable}, {self.body})"

class FunctionDeclaration(ASTNode):
    def __init__(self, name, params, body):
        self.name = name
        self.params = params
        self.body = body

    def __repr__(self):
        return f"FunctionDeclaration({self.name}, {self.params}, {self.body})"

class FunctionCall(ASTNode):
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments

    def __repr__(self):
        return f"FunctionCall({self.name}, {self.arguments})"

class ReturnStatement(ASTNode):
    def __init__(self, expression):
        self.expression = expression

    def __repr__(self):
        return f"ReturnStatement({self.expression})"

class ArrayLiteral(ASTNode):
    def __init__(self, elements):
        self.elements = elements

    def __repr__(self):
        return f"ArrayLiteral({self.elements})"

class ArrayAccess(ASTNode):
    def __init__(self, array, index):
        self.array = array
        self.index = index

    def __repr__(self):
        return f"ArrayAccess({self.array}, {self.index})"

class MethodCall(ASTNode):
    def __init__(self, object_name, method_name, arguments):
        self.object_name = object_name
        self.method_name = method_name
        self.arguments = arguments

    def __repr__(self):
        return f"MethodCall({self.object_name}, {self.method_name}, {self.arguments})"

# Tokenizer
def tokenize(code):
    token_specification = [
        ('INT', r'int\b'),
        ('FLOAT', r'float\b'),
        ('STRING_TYPE', r'string\b'),
        ('CONST', r'const\b'),
        ('VAR', r'var\b'),
        ('IF', r'if\b'),
        ('ELIF', r'elif\b'),
        ('ELSE', r'else\b'),
        ('WHILE', r'while\b'),
        ('FOR', r'for\b'),
        ('FUNC', r'func\b'),
        ('RETURN', r'return\b'),
        ('IN', r'in\b'),
        ('BEGIN', r'begin\b'),
        ('END', r'end\b'),
        ('REPEAT', r'repeat\b'),
        # Multi-character operators
        ('EQ', r'=='),
        ('GE', r'>='),
        ('LE', r'<='),
        # Single-character operators
        ('ASSIGN', r'='),
        ('GT', r'>'),
        ('LT', r'<'),
        ('NUMBER', r'\d+(\.\d+)?'),
        ('STRING', r'".*?"'),
        ('IDENT', r'[a-zA-Z_][a-zA-Z_0-9]*'),
        ('OP', r'[+\-*/%]'),
        ('LPAREN', r'\('),
        ('RPAREN', r'\)'),
        ('LBRACE', r'\{'),
        ('RBRACE', r'\}'),
        ('COLON', r':'),
        ('COMMA', r','),
        ('LBRACKET', r'\['),
        ('RBRACKET', r'\]'),
        ('DOT', r'\.'),
        ('NEWLINE', r'\n'),
        ('SKIP', r'[ \t]+'),
        ('COMMENT', r'#.*'),
        ('MISMATCH', r'.')
    ]

    # Fix the token regex construction
    tok_regex = '|'.join(f"(?P<{pair[0]}>{pair[1]})" for pair in token_specification)
    
    # Track bracket balance
    bracket_stack = []
    
    i = 0
    while i < len(code):
        # Check for multiline string
        if i < len(code) and code[i] == '|':
            # Find the closing vertical bar
            end = code.find('\n|', i + 1)
            if end == -1:
                raise SyntaxError("Unterminated multiline string, missing closing '|'")
            
            # Extract the string content (excluding delimiters)
            content = code[i+1:end]
            
            # Handle indentation - remove common leading whitespace
            lines = content.split('\n')
            
            # Find minimum indentation (ignoring empty lines)
            non_empty_lines = [line for line in lines if line.strip()]
            if non_empty_lines:
                min_indent = min(len(line) - len(line.lstrip()) for line in non_empty_lines)
                # Remove the common indentation
                processed_lines = []
                for line in lines:
                    if line.strip():  # If line is not empty
                        processed_lines.append(line[min_indent:])
                    else:
                        processed_lines.append(line)  # Keep empty lines as is
                content = '\n'.join(processed_lines)
            
            yield 'STRING', content
            i = end + 2  # Skip past the ending | and newline
            continue
        
        # Regular token matching
        match = re.match(tok_regex, code[i:])
        if match:
            kind = match.lastgroup
            value = match.group(kind)
            
            if kind == 'LBRACKET':
                bracket_stack.append('[')
            elif kind == 'RBRACKET':
                if bracket_stack and bracket_stack[-1] == '[':
                    bracket_stack.pop()
                else:
                    raise SyntaxError("Unexpected closing bracket ']' without matching opening bracket")
            
            if kind == 'NUMBER':
                value = float(value) if '.' in value else int(value)
            elif kind == 'STRING':
                # Process escape sequences in string literals
                value = value.strip('"')
                # Convert escape sequences like \n to actual newlines
                try:
                    value = bytes(value, "utf-8").decode("unicode_escape")
                except:
                    # If there's an error in escape sequence processing, keep original
                    pass
            elif kind in ['SKIP', 'COMMENT']:
                i += match.end()
                continue
            elif kind == 'MISMATCH':
                raise SyntaxError(f"Unexpected token: {value}")
            
            yield kind, value
            
            i += match.end()
        else:
            i += 1
    
    # Check for unclosed brackets at the end
    if bracket_stack:
        raise SyntaxError(f"Array is not closed. Missing closing bracket ']'")

class Parser:
    def __init__(self, tokens):
        self.tokens = list(tokens)
        self.pos = 0
        self.functions = {}  # Store user-defined functions
        self.declared_variables = set()  # Initialize declared_variables

    def current_token(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else (None, None)

    def eat(self, token_type):
        current = self.current_token()
        if current[0] == token_type:
            self.pos += 1
            return current[1]
        else:
            raise SyntaxError(f"Expected {token_type} but got {current[0]}")

    def parse(self):
        statements = []
        while self.current_token()[0] is not None:
            statement = self.parse_statement()
            if statement:
                statements.append(statement)
        return Program(statements)

    def parse_return_statement(self):
        self.eat('RETURN')
        expression = self.parse_expression()
        return ReturnStatement(expression)

    def parse_statement(self):
        token_type, value = self.current_token()
        
        if token_type == 'IDENT':
            next_token_type = self.tokens[self.pos + 1][0] if self.pos + 1 < len(self.tokens) else None
            if next_token_type == 'ASSIGN':
                return self.parse_assignment()
            elif next_token_type == 'LPAREN':
                func_name = self.eat('IDENT')
                self.eat('LPAREN')
                arguments = self.parse_argument_list()
                self.eat('RPAREN')
                
                if self.pos < len(self.tokens) and self.tokens[self.pos][0] == 'NEWLINE':
                    self.eat('NEWLINE')
                    
                return FunctionCall(func_name, arguments)
            elif value == 'printk':
                return self.parse_print_statement()
            elif value == 'repeat':
                return self.parse_repeat_loop()
            elif value == 'if':
                return self.parse_if_else()
            elif value == 'while':
                return self.parse_while_loop()
            elif value == 'for':
                return self.parse_for_loop()
            elif value == 'func':
                return self.parse_function_declaration()
            else:
                return self.parse_expression()
        elif token_type in ('INT', 'FLOAT', 'STRING_TYPE', 'CONST', 'VAR'):
            return self.parse_variable_declaration()
        elif token_type in ('IF', 'WHILE', 'FOR', 'FUNC', 'REPEAT', 'ELSE', 'ELIF'):
            if token_type == 'IF':
                return self.parse_if_else()
            elif token_type == 'WHILE':
                return self.parse_while_loop()
            elif token_type == 'FOR':
                return self.parse_for_loop()
            elif token_type == 'FUNC':
                return self.parse_function_declaration()
            elif token_type == 'REPEAT':
                return self.parse_repeat_loop()
            elif token_type == 'ELSE' or token_type == 'ELIF':
                raise SyntaxError(f"Unexpected '{value}' statement without matching 'if'")
        elif token_type == 'NEWLINE':
            self.eat('NEWLINE')
            return None
        elif token_type == 'RETURN':
            return self.parse_return_statement()
        elif token_type == 'END':
            self.eat('END')
            return None
        elif token_type == 'BEGIN':
            self.eat('BEGIN')
            return None
        else:
            current = self.current_token()
            if current[0] is not None:
                raise SyntaxError(f"Unknown statement or syntax: {current[1]}")
            else:
                return None

    def _parse_block(self):
        body = []
        while self.current_token()[0] not in ('END', 'ELSE', 'ELIF', None):
            statement = self.parse_statement()
            if statement:
                body.append(statement)

        # Allow optional NEWLINE before END
        while self.current_token()[0] == 'NEWLINE':
            self.eat('NEWLINE')
        
        # Consume the END token
        if self.current_token()[0] == 'END':
            self.eat('END')
        else:
            raise SyntaxError("Expected 'end' at the end of block")
        
        return body

    def parse_function_declaration(self):
        self.eat('FUNC')
        name = self.eat('IDENT')
        self.declared_variables.add(name)
        self.eat('LPAREN')
        params = []
        
        # Parse parameters
        while self.current_token()[0] != 'RPAREN':
            token_type, value = self.current_token()
            
            if token_type == 'IDENT' and value == 'string':
                param_type = 'string'
                self.eat('IDENT')
            else:
                param_type = self.eat(token_type)
                if param_type not in ('int', 'float', 'const', 'var'):
                    raise SyntaxError(f"Unexpected parameter type: {param_type}")
            
            param_name = self.eat('IDENT')
            
            # Check for array parameter
            is_array = False
            if self.current_token()[0] == 'LBRACKET':
                self.eat('LBRACKET')
                self.eat('RBRACKET')
                is_array = True
                
            params.append((param_type, param_name, is_array))
            
            if self.current_token()[0] == 'COMMA':
                self.eat('COMMA')
                
        self.eat('RPAREN')
        
        # Add parameters to declared variables
        for param_type, param_name, _ in params:
            self.declared_variables.add(param_name)
            
        self.eat('BEGIN')
        
        if self.current_token()[0] != 'NEWLINE':
            raise SyntaxError("Expected newline after 'begin'")
        self.eat('NEWLINE')
        
        body = self._parse_block()
        
        if self.current_token()[0] != 'NEWLINE':
            raise SyntaxError("Expected newline after 'end'")
        self.eat('NEWLINE')
        
        func_decl = FunctionDeclaration(name, params, body)
        self.functions[name] = func_decl
        return func_decl

    def parse_function_call(self):
        func_name = self.eat('IDENT')
        self.eat('LPAREN')
        arguments = self.parse_argument_list()
        self.eat('RPAREN')
        return FunctionCall(func_name, arguments)

    def parse_argument_list(self):
        arguments = []
        while self.current_token()[0] != 'RPAREN':
            arguments.append(self.parse_expression())
            if self.current_token()[0] == 'COMMA':
                self.eat('COMMA')
        return arguments

    def parse_variable_declaration(self):
        # Check for optional mutability specifier
        mutability = None
        if self.current_token()[0] in ('CONST', 'VAR'):
            mutability = self.current_token()[1]
            self.eat(self.current_token()[0])
        
        # Require type specifier
        token_type = self.current_token()[0]
        if token_type not in ('INT', 'FLOAT', 'STRING_TYPE'):
            raise SyntaxError(f"Type specifier (int, float, string) is required for variable declarations")
        
        # Get the type
        var_type = self.current_token()[1]
        if token_type == 'STRING_TYPE':
            var_type = 'string'
        self.eat(token_type)
        
        # Combine mutability and type if specified
        if mutability:
            var_type = f"{mutability}_{var_type}"
        
        # Parse variable name
        name = self.eat('IDENT')
        
        # Check for array declaration
        is_array = False
        array_size = None
        if self.current_token()[0] == 'LBRACKET':
            self.eat('LBRACKET')
            
            if self.current_token()[0] == 'NUMBER':
                array_size = int(self.eat('NUMBER'))
                
            self.eat('RBRACKET')
            is_array = True
        
        self.declared_variables.add(name)

        # Check for assignment
        if self.current_token()[0] == 'ASSIGN':
            self.eat('ASSIGN')
            value = self.parse_expression()
            self.eat('NEWLINE')
            return VariableDeclaration(var_type, name, value, is_array, array_size)
        else:
            self.eat('NEWLINE')
            return VariableDeclaration(var_type, name, None, is_array, array_size)

    def parse_print_statement(self):
        self.eat('IDENT')  # Expecting 'printk'
        self.eat('LBRACE')
        expression = self.parse_expression()
        self.eat('RBRACE')
        return PrintStatement(expression)

    def parse_repeat_loop(self):
        self.eat('REPEAT')
        count = self.parse_expression()
        self.eat('BEGIN')
        
        if self.current_token()[0] != 'NEWLINE':
            raise SyntaxError("Expected newline after 'begin'")
        self.eat('NEWLINE')
        
        body = self._parse_block()
        
        if self.current_token()[0] != 'NEWLINE':
            raise SyntaxError("Expected newline after 'end'")
        self.eat('NEWLINE')
        
        return RepeatLoop(count, body)

    def parse_if_else(self):
        self.eat('IF')
        self.eat('LPAREN')
        condition = self.parse_expression()
        self.eat('RPAREN')
        self.eat('BEGIN')
        
        if self.current_token()[0] != 'NEWLINE':
            raise SyntaxError("Expected newline after 'begin'")
        self.eat('NEWLINE')
        
        true_branch = []
        while self.current_token()[0] not in ('ELSE', 'ELIF', 'END', None):
            statement = self.parse_statement()
            if statement:
                true_branch.append(statement)
        
        # Skip newlines
        while self.current_token()[0] == 'NEWLINE':
            self.eat('NEWLINE')
        
        elif_branches = []
        false_branch = []
        
        # Handle 'elif' branches
        while self.current_token()[0] == 'ELIF':
            self.eat('ELIF')
            self.eat('LPAREN')
            elif_condition = self.parse_expression()
            self.eat('RPAREN')
            self.eat('BEGIN')
            
            if self.current_token()[0] != 'NEWLINE':
                raise SyntaxError("Expected newline after 'begin'")
            self.eat('NEWLINE')
            
            elif_body = []
            while self.current_token()[0] not in ('ELSE', 'ELIF', 'END', None):
                statement = self.parse_statement()
                if statement:
                    elif_body.append(statement)
            
            # Skip newlines
            while self.current_token()[0] == 'NEWLINE':
                self.eat('NEWLINE')
            
            elif_branches.append((elif_condition, elif_body))
        
        # Require 'else' branch
        if self.current_token()[0] != 'ELSE':
            raise SyntaxError("Expected 'else' branch for 'if' statement")
        
        # Handle 'else' branch
        self.eat('ELSE')
        self.eat('BEGIN')
        
        if self.current_token()[0] != 'NEWLINE':
            raise SyntaxError("Expected newline after 'begin'")
        self.eat('NEWLINE')
        
        # Parse the 'else' block
        while self.current_token()[0] not in ('END', None):
            statement = self.parse_statement()
            if statement:
                false_branch.append(statement)
        
        # Skip newlines
        while self.current_token()[0] == 'NEWLINE':
            self.eat('NEWLINE')
        
        # Consume 'end'
        if self.current_token()[0] == 'END':
            self.eat('END')
            
            if self.current_token()[0] != 'NEWLINE':
                raise SyntaxError("Expected newline after 'end'")
            self.eat('NEWLINE')
        else:
            raise SyntaxError("Expected 'end' at the end of if-else block")
        
        return IfElse(condition, true_branch, elif_branches, false_branch)

    def parse_while_loop(self):
        self.eat('WHILE')
        self.eat('LPAREN')
        condition = self.parse_expression()
        self.eat('RPAREN')
        self.eat('BEGIN')
        
        if self.current_token()[0] != 'NEWLINE':
            raise SyntaxError("Expected newline after 'begin'")
        self.eat('NEWLINE')
        
        body = self._parse_block()
        
        if self.current_token()[0] != 'NEWLINE':
            raise SyntaxError("Expected newline after 'end'")
        self.eat('NEWLINE')
        
        return WhileLoop(condition, body)

    def parse_for_loop(self):
        self.eat('FOR')
        loop_var = self.eat('IDENT')
        self.eat('IN')
        iterable = self.parse_expression()
        self.eat('BEGIN')
        self.eat('NEWLINE')
        body = self._parse_block()
        return ForLoop(loop_var, iterable, body)

    def parse_expression(self):
        token_type, value = self.current_token()
        
        # Handle identifiers (variables)
        if token_type == 'IDENT':
            ident = self.eat('IDENT')
            
            # Check if the variable is declared
            if ident not in self.declared_variables and ident not in self.functions:
                raise SyntaxError(f"Variable '{ident}' is not declared. Please declare it before use.")
            
            # Check for function call
            if self.current_token()[0] == 'LPAREN':
                self.eat('LPAREN')
                arguments = self.parse_argument_list()
                self.eat('RPAREN')
                return FunctionCall(ident, arguments)
            
            # Check for method call
            elif self.current_token()[0] == 'DOT':
                self.eat('DOT')
                method_name = self.eat('IDENT')
                self.eat('LPAREN')
                arguments = self.parse_argument_list()
                self.eat('RPAREN')
                return MethodCall(ident, method_name, arguments)
            
            # Check for array access
            elif self.current_token()[0] == 'LBRACKET':
                self.eat('LBRACKET')
                index = self.parse_expression()
                self.eat('RBRACKET')
                left = ArrayAccess(ident, index)
            else:
                left = Expression(ident)
        
        # Handle numbers
        elif token_type == 'NUMBER':
            self.eat('NUMBER')
            left = Expression(value)
        
        # Handle string literals
        elif token_type == 'STRING':
            self.eat('STRING')
            left = Expression(value)
        
        # Handle parenthesized expressions
        elif token_type == 'LPAREN':
            self.eat('LPAREN')
            left = self.parse_expression()
            self.eat('RPAREN')
        
        # Handle array literals
        elif token_type == 'LBRACKET':
            try:
                self.eat('LBRACKET')
                elements = []
                
                # Check for empty array
                if self.current_token()[0] == 'RBRACKET':
                    self.eat('RBRACKET')
                    return ArrayLiteral(elements)
                
                # Parse array elements
                while self.current_token()[0] != 'RBRACKET':
                    if self.current_token()[0] is None:
                        raise SyntaxError(f"Array is not closed. Missing closing bracket ']'")
                    
                    elements.append(self.parse_expression())
                    
                    if self.current_token()[0] == 'COMMA':
                        self.eat('COMMA')
                    elif self.current_token()[0] == 'RBRACKET':
                        break
                    elif self.current_token()[0] == 'NEWLINE' or self.current_token()[0] is None:
                        raise SyntaxError(f"Array is not closed. Missing closing bracket ']'")
                    else:
                        raise SyntaxError(f"Expected ',' or ']' in array but got '{self.current_token()[1]}'")
                
                self.eat('RBRACKET')
                left = ArrayLiteral(elements)
            except SyntaxError as e:
                if "Array is not closed" in str(e):
                    raise
                else:
                    raise SyntaxError(f"Invalid array syntax: {e}")
        
        # Handle unary minus
        elif token_type == 'OP' and value == '-':
            self.eat('OP')
            operand = self.parse_expression()
            left = BinaryOperation(Expression(0), '-', operand)
        
        else:
            raise SyntaxError(f"Unexpected expression: {value}")
        
        # Handle binary operations
        while self.current_token()[0] in ['OP', 'LT', 'GT', 'EQ', 'GE', 'LE']:
            operator_type, operator = self.current_token()
            
            # Check for assignment vs equality
            if operator_type == 'ASSIGN':
                raise SyntaxError("Cannot use assignment operator (=) in conditions. Use equality operator (==) instead.")
            
            self.eat(operator_type)
            right = self.parse_primary()
            left = BinaryOperation(left, operator, right)
        
        return left

    def parse_primary(self):
        token_type, value = self.current_token()

        if token_type == 'OP' and value == '-':
            self.eat('OP')
            operand = self.parse_primary()
            return BinaryOperation(Expression(0), '-', operand)
        elif token_type == 'NUMBER':
            self.eat('NUMBER')
            return Expression(value)
        elif token_type == 'STRING':
            self.eat('STRING')
            return Expression(value)
        elif token_type == 'IDENT':
            ident = self.eat('IDENT')
            if ident not in self.declared_variables:
                raise SyntaxError(f"Undeclared variable: {ident}")

            if self.current_token()[0] == 'LBRACKET':
                self.eat('LBRACKET')
                index = self.parse_expression()
                self.eat('RBRACKET')
                return ArrayAccess(ident, index)
            else:
                return Expression(ident)
        elif token_type == 'LPAREN':
            self.eat('LPAREN')
            expr = self.parse_expression()
            self.eat('RPAREN')
            return expr
        elif token_type == 'LBRACKET':
            self.eat('LBRACKET')
            elements = []
            while self.current_token()[0] != 'RBRACKET':
                elements.append(self.parse_expression())
                if self.current_token()[0] == 'COMMA':
                    self.eat('COMMA')
            self.eat('RBRACKET')
            return ArrayLiteral(elements)
        else:
            raise SyntaxError(f"Unexpected primary expression: {value}")
    
    def parse_array_literal(self):
        self.eat('LBRACKET')
        elements = []
        
        if self.current_token()[0] == 'RBRACKET':
            self.eat('RBRACKET')
            return ArrayLiteral(elements)
        
        while True:
            if self.current_token()[0] is None:
                raise SyntaxError(f"Array is not closed. Missing closing bracket ']'")
                
            elements.append(self.parse_expression())
            
            if self.current_token()[0] == 'COMMA':
                self.eat('COMMA')
            elif self.current_token()[0] == 'RBRACKET':
                self.eat('RBRACKET')
                break
            elif self.current_token()[0] == 'NEWLINE' or self.current_token()[0] is None:
                raise SyntaxError(f"Array is not closed. Missing closing bracket ']'")
            else:
                raise SyntaxError(f"Expected ',' or ']' in array but got '{self.current_token()[1]}'")
        
        return ArrayLiteral(elements)

    def parse_assignment(self):
        name = self.eat('IDENT')
        if name not in self.declared_variables:
            raise SyntaxError(f"Variable '{name}' is not declared. Please declare it before assignment.")
        
        self.eat('ASSIGN')
        
        if self.current_token()[0] == 'IDENT' and self.tokens[self.pos + 1][0] == 'LPAREN':
            func_name = self.eat('IDENT')
            self.eat('LPAREN')
            arguments = self.parse_argument_list()
            self.eat('RPAREN')
            value = FunctionCall(func_name, arguments)
        else:
            value = self.parse_expression()
        
        if self.current_token()[0] == 'NEWLINE':
            self.eat('NEWLINE')
        
        return VariableDeclaration(None, name, value)

class Evaluator:
    def __init__(self, functions):
        self.variables = {}
        self.functions = functions

    def evaluate(self, node):
        if isinstance(node, Program):
            for statement in node.statements:
                self.evaluate(statement)
        elif isinstance(node, VariableDeclaration):
            self.evaluate_variable_declaration(node)
        elif isinstance(node, PrintStatement):
            self.evaluate_print_statement(node)
        elif isinstance(node, RepeatLoop):
            self.evaluate_repeat_loop(node)
        elif isinstance(node, BinaryOperation):
            return self.evaluate_binary_operation(node)
        elif isinstance(node, Expression):
            return self.evaluate_expression(node)
        elif isinstance(node, IfElse):
            self.evaluate_if_else(node)
        elif isinstance(node, WhileLoop):
            self.evaluate_while_loop(node)
        elif isinstance(node, ForLoop):
            self.evaluate_for_loop(node)
        elif isinstance(node, FunctionDeclaration):
            self.evaluate_function_declaration(node)
        elif isinstance(node, FunctionCall):
            return self.evaluate_function_call(node)
        elif isinstance(node, ReturnStatement):
            return self.evaluate_return_statement(node)
        elif isinstance(node, ArrayLiteral):
            return self.evaluate_array_literal(node)
        elif isinstance(node, ArrayAccess):
            return self.evaluate_array_access(node)
        elif isinstance(node, MethodCall):
            return self.evaluate_method_call(node)
        else:
            raise ValueError(f"Unknown node type: {type(node)}")

    def evaluate_variable_declaration(self, node):
        value = self.evaluate(node.value) if node.value else None
        
        # Check if this is a const variable
        is_const = node.var_type == 'const' or (node.var_type and node.var_type.startswith('const_'))
        is_reassignment = node.name in self.variables and node.var_type is None
        
        # Extract the actual type if it's a combined form
        actual_type = None
        if node.var_type and '_' in node.var_type:
            _, actual_type = node.var_type.split('_', 1)
        else:
            actual_type = node.var_type
        
        # Prevent reassigning constants
        if is_reassignment and self.variables.get(f"__const_{node.name}"):
            raise SyntaxError(f"Cannot reassign constant: {node.name}")
        
        # Type checking for reassignment
        if is_reassignment:
            original_type = self.variables.get(f"__type_{node.name}")
            if original_type:
                # Check type compatibility based on original declaration
                if original_type == 'int' and not isinstance(value, int):
                    raise TypeError(f"Type error: Cannot assign {type(value).__name__} to variable '{node.name}' of type {original_type}")
                elif original_type == 'float' and not isinstance(value, (int, float)):
                    raise TypeError(f"Type error: Cannot assign {type(value).__name__} to variable '{node.name}' of type {original_type}")
                elif original_type == 'string' and not isinstance(value, str):
                    raise TypeError(f"Type error: Cannot assign {type(value).__name__} to variable '{node.name}' of type {original_type}")
        
        # Store the variable and mark as const if needed
        self.variables[node.name] = value
        if is_const:
            self.variables[f"__const_{node.name}"] = True
        
        # Store the type information for future checks
        if actual_type:
            self.variables[f"__type_{node.name}"] = actual_type

    def evaluate_print_statement(self, node):
        value = self.evaluate(node.expression)
        print(value, end='')  # Use end='' to prevent automatic newline

    def evaluate_repeat_loop(self, node):
        count = self.evaluate(node.count)
        for _ in range(int(count)):
            for statement in node.body:
                self.evaluate(statement)

    def evaluate_binary_operation(self, node):
        left = self.evaluate(node.left)
        right = self.evaluate(node.right)
        
        if node.operator == '+':
            return left + right
        elif node.operator == '-':
            return left - right
        elif node.operator == '*':
            return left * right
        elif node.operator == '/':
            return left / right
        elif node.operator == '%':
            return left % right
        elif node.operator == '<':
            return left < right
        elif node.operator == '>':
            return left > right
        elif node.operator == '==':
            return left == right
        elif node.operator == '>=':
            return left >= right
        elif node.operator == '<=':
            return left <= right
        elif node.operator == '=':
            raise ValueError("Assignment operator (=) cannot be used in conditions. Use equality operator (==) instead.")
        else:
            raise ValueError(f"Unknown operator: {node.operator}")

    def evaluate_expression(self, node):
        if isinstance(node.value, str) and node.value in self.variables:
            return self.variables[node.value]
        return node.value

    def evaluate_if_else(self, node):
        condition = self.evaluate(node.condition)
        if condition:
            for statement in node.true_branch:
                self.evaluate(statement)
        else:
            # Evaluate elif branches
            for elif_condition, elif_body in node.elif_branches:
                if self.evaluate(elif_condition):
                    for statement in elif_body:
                        self.evaluate(statement)
                    return
                    
            # Evaluate else branch if no elif conditions are true
            if node.false_branch:
                for statement in node.false_branch:
                    self.evaluate(statement)

    def evaluate_while_loop(self, node):
        while self.evaluate(node.condition):
            for statement in node.body:
                self.evaluate(statement)

    def evaluate_for_loop(self, node):
        iterable = self.evaluate(node.iterable)
        for value in iterable:
            self.variables[node.loop_var] = value
            for statement in node.body:
                self.evaluate(statement)

    def evaluate_function_declaration(self, node):
        self.functions[node.name] = node

    def evaluate_function_call(self, node):
        func = self.functions.get(node.name)
        if not func:
            raise ValueError(f"Function '{node.name}' is not defined")

        old_variables = self.variables.copy()

        # Bind arguments to parameters
        if len(func.params) != len(node.arguments):
            raise ValueError(f"Function '{node.name}' expected {len(func.params)} arguments but got {len(node.arguments)}")

        for (param_type, param_name, is_array), arg in zip(func.params, node.arguments):
            value = self.evaluate(arg)
            self.variables[param_name] = value

        result = None
        # Execute function body
        for statement in func.body:
            stmt_result = self.evaluate(statement)
            if isinstance(statement, ReturnStatement):
                result = stmt_result
                break

        self.variables = old_variables
        return result

    def evaluate_return_statement(self, node):
        return self.evaluate(node.expression)

    def evaluate_array_literal(self, node):
        return [self.evaluate(element) for element in node.elements]

    def evaluate_array_access(self, node):
        if isinstance(node.array, str):
            array = self.variables.get(node.array)
            if array is None:
                raise ValueError(f"Array '{node.array}' not defined")
        else:
            array = self.evaluate(node.array)

        index = self.evaluate(node.index)

        if not isinstance(array, (list, tuple)):
            raise ValueError(f"Expected array but got {type(array)}")
        if not isinstance(index, int):
            raise ValueError(f"Array index must be an integer, got {type(index)}")
        if index < 0 or index >= len(array):
            raise ValueError(f"Index {index} out of bounds for array of length {len(array)}")

        return array[index]

    def evaluate_method_call(self, node):
        # Get the object (array)
        object_value = self.variables.get(node.object_name)
        if object_value is None:
            raise ValueError(f"Object '{node.object_name}' not defined")
        
        if not isinstance(object_value, list):
            raise ValueError(f"Method call only supported on arrays, but '{node.object_name}' is {type(object_value)}")
        
        # Evaluate method arguments
        args = [self.evaluate(arg) for arg in node.arguments]
        
        # Handle different methods
        if node.method_name == 'remove':
            return self._evaluate_array_remove(object_value, args, node.object_name)
        elif node.method_name == 'add':
            return self._evaluate_array_add(object_value, args, node.object_name)
        else:
            raise ValueError(f"Unknown method '{node.method_name}' for array")

    def _evaluate_array_remove(self, array, args, array_name):
        # No arguments means remove last element
        if not args:
            if not array:
                raise IndexError(f"IndexError: remove index out of range (array is empty)")
            removed = array.pop()
            return removed
        
        # One argument specifies the index to remove
        if len(args) == 1:
            index = args[0]
            if not isinstance(index, int):
                raise ValueError(f"Array index must be an integer, got {type(index)}")
            
            # Handle negative indices
            if index < 0:
                index = len(array) + index
            
            # Check bounds
            if index < 0 or index >= len(array):
                raise IndexError(f"IndexError: remove index out of range")
            
            # Remove and return the element
            removed = array.pop(index)
            return removed
        
        raise ValueError(f"remove() takes at most 1 argument ({len(args)} given)")

    def _evaluate_array_add(self, array, args, array_name):
        # One argument means add at the end
        if len(args) == 1:
            value = args[0]
            array.append(value)
            return value
        
        # Two arguments: first is index, second is value
        elif len(args) == 2:
            index, value = args[0], args[1]
            if not isinstance(index, int):
                raise ValueError(f"Array index must be an integer, got {type(index)}")
            
            # Handle negative indices (insert before the element at that position)
            if index < 0:
                index = len(array) + index
            
            # For add, we allow index == len(array) to append at the end
            if index < 0 or index > len(array):
                raise IndexError(f"IndexError: add index out of range")
            
            # Insert the element
            array.insert(index, value)
            return value
        
        raise ValueError(f"add() takes 1 or 2 arguments ({len(args)} given)")

# Function to run code
def run_code(code):
    try:
        tokens = list(tokenize(code))
        parser = Parser(tokens)
        ast = parser.parse()
        evaluator = Evaluator(parser.functions)
        evaluator.evaluate(ast)
        return True
    except SyntaxError as e:
        print(f"SyntaxError: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

# Main function to run the interpreter
if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Run code from file
        file_path = sys.argv[1]
        try:
            with open(file_path, 'r') as file:
                code = file.read()
            run_code(code)
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found")
    else:
        print("Usage: python evaluator.py <filename>")
