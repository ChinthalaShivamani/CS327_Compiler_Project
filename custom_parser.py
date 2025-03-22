import re
import math

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

class ReturnStatement(ASTNode): # Add this AST node
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
        ('STRING_TYPE', r'string\b'),  # Add this token for string type keyword
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
    
    for match in re.finditer(tok_regex, code):
        kind = match.lastgroup
        value = match.group(kind)
        
        # Track brackets
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
            value = value.strip('"')
        elif kind in ['SKIP', 'COMMENT']:
            continue
        elif kind == 'MISMATCH':
            raise SyntaxError(f"Unexpected token: {value}")
        
        yield kind, value
    
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
        # Do not require a NEWLINE after the return statement
        return ReturnStatement(expression)

    def parse_statement(self):
        token_type, value = self.current_token()
        
        if token_type == 'IDENT':
            next_token_type = self.tokens[self.pos + 1][0] if self.pos + 1 < len(self.tokens) else None
            if next_token_type == 'ASSIGN':
                return self.parse_assignment()
            elif next_token_type == 'LPAREN':  # This is a function call
                func_name = self.eat('IDENT')
                self.eat('LPAREN')
                arguments = self.parse_argument_list()
                self.eat('RPAREN')
                
                # Make the newline optional
                if self.current_token()[0] == 'NEWLINE':
                    self.eat('NEWLINE')
                    
                return FunctionCall(func_name, arguments)
            elif value == 'printk':
                return self.parse_print_statement()
            elif value == 'repeat':  # Handle 'repeat' keyword
                return self.parse_repeat_loop()
            elif value == 'if':  # Handle 'if' directly here
                return self.parse_if_else()
            elif value == 'while':  # Handle 'while' directly here
                return self.parse_while_loop()
            elif value == 'for':  # Handle 'for' directly here
                return self.parse_for_loop()
            elif value == 'func':  # Handle 'func' directly here
                return self.parse_function_declaration()
            else:  # It's an expression or something else
                return self.parse_expression()  # or handle other IDENT cases
        elif token_type in ('INT', 'FLOAT', 'STRING_TYPE', 'CONST', 'VAR'):  # Added STRING_TYPE here
            return self.parse_variable_declaration()
        elif token_type in ('IF', 'WHILE', 'FOR', 'FUNC', 'REPEAT', 'ELSE', 'ELIF'):  # Added 'ELSE' and 'ELIF' here
            if token_type == 'IF':
                return self.parse_if_else()
            elif token_type == 'WHILE':
                return self.parse_while_loop()
            elif token_type == 'FOR':
                return self.parse_for_loop()
            elif token_type == 'FUNC':
                return self.parse_function_declaration()
            elif token_type == 'REPEAT':  # Handle 'repeat' keyword
                return self.parse_repeat_loop()
            elif token_type == 'ELSE' or token_type == 'ELIF':
                raise SyntaxError(f"Unexpected '{value}' statement without matching 'if'")
        elif token_type == 'NEWLINE':
            self.eat('NEWLINE')
            return None  # Important: Return None for empty lines!
        elif token_type == 'RETURN':  # Handle return statements
            self.eat('RETURN')
            expression = self.parse_expression()
            self.eat('NEWLINE')
            return ReturnStatement(expression)  # Assuming you have a ReturnStatement AST node
        elif token_type == 'END':  # Handle 'end' token
            self.eat('END')
            return None  # Return None to indicate no statement was parsed
        elif token_type == 'BEGIN':  # Handle 'begin' token similarly
            self.eat('BEGIN')
            return None  # Return None to indicate no statement was parsed
        else:
            current = self.current_token()
            if current[0] is not None:
                raise SyntaxError(f"Unknown statement or syntax: {current[1]}")
            else:
                return None  # End of file
        
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
        self.eat('FUNC')  # Consume 'func'
        name = self.eat('IDENT')  # Function name
        self.declared_variables.add(name)
        self.eat('LPAREN')  # Consume '('
        params = []
        
        # Parse parameters
        while self.current_token()[0] != 'RPAREN':
            token_type, value = self.current_token()
            
            # Handle string type as an identifier
            if token_type == 'IDENT' and value == 'string':
                param_type = 'string'
                self.eat('IDENT')
            else:
                param_type = self.eat(token_type)
                if param_type not in ('int', 'float', 'const', 'var'):
                    raise SyntaxError(f"Unexpected parameter type: {param_type}")
            
            param_name = self.eat('IDENT')
            
            # Check for array parameter - now after the parameter name
            is_array = False
            if self.current_token()[0] == 'LBRACKET':
                self.eat('LBRACKET')
                self.eat('RBRACKET')
                is_array = True
                
            params.append((param_type, param_name, is_array))
            
            if self.current_token()[0] == 'COMMA':
                self.eat('COMMA')
                
        self.eat('RPAREN')  # Consume ')'
        
        # Add parameters to declared variables
        for param_type, param_name, _ in params:
            self.declared_variables.add(param_name)
            
        self.eat('BEGIN')  # Consume 'begin'
        
        # Make newline mandatory after 'begin'
        if self.current_token()[0] != 'NEWLINE':
            raise SyntaxError("Expected newline after 'begin'")
        self.eat('NEWLINE')
        
        body = self._parse_block()  # Parse function body
        
        # Make newline mandatory after 'end'
        if self.current_token()[0] != 'NEWLINE':
            raise SyntaxError("Expected newline after 'end'")
        self.eat('NEWLINE')
        
        func_decl = FunctionDeclaration(name, params, body)
        self.functions[name] = func_decl
        return func_decl

    def parse_function_call(self):
        func_name = self.eat('IDENT')  # Function name
        self.eat('LPAREN')  # Left parenthesis
        arguments = self.parse_argument_list()  # Parse arguments
        self.eat('RPAREN')  # Right parenthesis
        # Make newline optional after function call
        if self.current_token()[0] == 'NEWLINE':
            self.eat('NEWLINE')
        return FunctionCall(func_name, arguments)

    def parse_argument_list(self):
        arguments = []
        while self.current_token()[0] != 'RPAREN':
            arguments.append(self.parse_expression())
            if self.current_token()[0] == 'COMMA':
                self.eat('COMMA')
        return arguments

    def parse_variable_declaration(self):
        # Handle int, float, string, const, and var declarations
        token_type = self.current_token()[0]
        var_type = self.current_token()[1]
        
        if token_type in ('INT', 'FLOAT', 'STRING_TYPE', 'CONST', 'VAR'):
            # Convert STRING_TYPE to 'string' for consistency with other types
            if token_type == 'STRING_TYPE':
                var_type = 'string'
            self.eat(token_type)
        else:
            raise SyntaxError(f"Expected type declaration but got {token_type}")
        
        # Rest of the method remains unchanged
        name = self.eat('IDENT')  # Variable name
        
        # Check if this is an array declaration (e.g., float numbers[])
        is_array = False
        array_size = None
        if self.current_token()[0] == 'LBRACKET':
            self.eat('LBRACKET')  # Consume '['
            
            # Check if there's a size specified
            if self.current_token()[0] == 'NUMBER':
                array_size = int(self.eat('NUMBER'))
                
            self.eat('RBRACKET')  # Consume ']'
            is_array = True
        
        self.declared_variables.add(name)

        # Check if there's an assignment following the variable declaration
        if self.current_token()[0] == 'ASSIGN':
            self.eat('ASSIGN')  # Consume '='
            value = self.parse_expression()  # Parse the right-hand side expression
            self.eat('NEWLINE')  # Consume the newline after declaration
            return VariableDeclaration(var_type, name, value, is_array, array_size)
        else:
            self.eat('NEWLINE')  # Consume the newline after declaration
            return VariableDeclaration(var_type, name, None, is_array, array_size)

    def parse_print_statement(self):
        self.eat('IDENT')  # Expecting 'printk'
        self.eat('LBRACE')  # Expecting '{'
        expression = self.parse_expression()  # Parse the expression inside the braces
        self.eat('RBRACE')  # Expecting '}'
        return PrintStatement(expression)

    def parse_repeat_loop(self):
        self.eat('REPEAT')  # Consume 'repeat'
        count = self.parse_expression()  # Parse repetition count
        self.eat('BEGIN')  # Consume 'begin'
        
        # Make newline mandatory after 'begin'
        if self.current_token()[0] != 'NEWLINE':
            raise SyntaxError("Expected newline after 'begin'")
        self.eat('NEWLINE')
        
        body = self._parse_block()  # Parse loop body
        
        # Make newline mandatory after 'end'
        if self.current_token()[0] != 'NEWLINE':
            raise SyntaxError("Expected newline after 'end'")
        self.eat('NEWLINE')
        
        return RepeatLoop(count, body)

    def parse_if_else(self):
        self.eat('IF')  # Consume 'if'
        self.eat('LPAREN')  # Consume '('
        condition = self.parse_expression()  # Parse condition
        self.eat('RPAREN')  # Consume ')'
        self.eat('BEGIN')  # Consume 'begin'
        
        # Make newline mandatory after 'begin'
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
            self.eat('ELIF')  # Consume 'elif'
            self.eat('LPAREN')  # Consume '('
            elif_condition = self.parse_expression()  # Parse elif condition
            self.eat('RPAREN')  # Consume ')'
            self.eat('BEGIN')  # Consume 'begin'
            
            # Make newline mandatory after 'begin'
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
        
        # MODIFICATION: Require 'else' branch for every 'if' statement
        if self.current_token()[0] == 'ELSE':
            # Handle 'else' branch(optional)
            self.eat('ELSE')
            self.eat('BEGIN')
        
            # Make newline mandatory after 'begin'
            if self.current_token()[0] != 'NEWLINE':
                raise SyntaxError("Expected newline after 'begin'")
            self.eat('NEWLINE')
        
            # Parse the 'else' block until 'END'
            while self.current_token()[0] not in ('END', None):
                statement = self.parse_statement()
                if statement:
                    false_branch.append(statement)
        
            # Skip newlines
            while self.current_token()[0] == 'NEWLINE':
                self.eat('NEWLINE')
        
        # Consume the 'end' token at the end of the if-else structure
        if self.current_token()[0] == 'END':
            self.eat('END')
            
            # Make newline mandatory after 'end'
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
        
        # Make newline mandatory after 'begin'
        if self.current_token()[0] != 'NEWLINE':
            raise SyntaxError("Expected newline after 'begin'")
        self.eat('NEWLINE')
        
        body = self._parse_block()
        
        # Make newline mandatory after 'end'
        if self.current_token()[0] != 'NEWLINE':
            raise SyntaxError("Expected newline after 'end'")
        self.eat('NEWLINE')
        
        return WhileLoop(condition, body)

    def parse_for_loop(self):
        self.eat('FOR')
        loop_var = self.eat('IDENT')
        self.eat('IN')
        iterable = self.parse_expression()
        self.eat('BEGIN')  # Expect colon after iterable
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
            
            # Check for function call (e.g., sum_array(numbers))
            if self.current_token()[0] == 'LPAREN':
                self.eat('LPAREN')
                arguments = self.parse_argument_list()
                self.eat('RPAREN')
                return FunctionCall(ident, arguments)
            
            # Check for method call (e.g., array.method())
            elif self.current_token()[0] == 'DOT':
                self.eat('DOT')
                method_name = self.eat('IDENT')
                self.eat('LPAREN')
                arguments = self.parse_argument_list()
                self.eat('RPAREN')
                return MethodCall(ident, method_name, arguments)
            
            # Check for array access - existing code continues
            elif self.current_token()[0] == 'LBRACKET':
                self.eat('LBRACKET')
                index = self.parse_expression()
                self.eat('RBRACKET')
                left = ArrayAccess(ident, index)
            else:
                left = Expression(ident)
        
        # Handle numbers (integer or float)
        elif token_type == 'NUMBER':
            self.eat('NUMBER')
            left = Expression(value)
        
        # Handle string literals
        elif token_type == 'STRING':
            self.eat('STRING')
            left = Expression(value)
        
        # Handle parenthesized expressions
        elif token_type == 'LPAREN':  # '('
            self.eat('LPAREN')  # Consume '('
            left = self.parse_expression()  # Parse the inner expression
            self.eat('RPAREN')  # Ensure closing ')'
        
        # Handle array literals
        elif token_type == 'LBRACKET':  # '['
            try:
                self.eat('LBRACKET')  # Consume '['
                elements = []
                
                # Check for empty array
                if self.current_token()[0] == 'RBRACKET':
                    self.eat('RBRACKET')
                    return ArrayLiteral(elements)
                
                # Parse array elements
                while self.current_token()[0] != 'RBRACKET':  # Parse elements until ']'
                    if self.current_token()[0] is None:
                        raise SyntaxError(f"Array is not closed. Missing closing bracket ']'")
                    
                    elements.append(self.parse_expression())
                    
                    if self.current_token()[0] == 'COMMA':  # Handle comma-separated elements
                        self.eat('COMMA')
                    elif self.current_token()[0] == 'RBRACKET':
                        break
                    elif self.current_token()[0] == 'NEWLINE' or self.current_token()[0] is None:
                        raise SyntaxError(f"Array is not closed. Missing closing bracket ']'")
                    else:
                        raise SyntaxError(f"Expected ',' or ']' in array but got '{self.current_token()[1]}'")
                
                self.eat('RBRACKET')  # Ensure closing ']'
                left = ArrayLiteral(elements)
            except SyntaxError as e:
                if "Array is not closed" in str(e):
                    raise  # Re-raise the specific array error
                else:
                    raise SyntaxError(f"Invalid array syntax: {e}")
        
        # Handle unary minus (e.g., -x)
        elif token_type == 'OP' and value == '-':
            self.eat('OP')  # Consume '-'
            operand = self.parse_expression()
            left = BinaryOperation(Expression(0), '-', operand)  # Convert -x to 0 - x
        
        else:
            raise SyntaxError(f"Unexpected expression: {value}")
        
        # Handle binary operations (like a > b)
        while self.current_token()[0] in ['OP', 'LT', 'GT', 'EQ', 'GE', 'LE']:
            operator_type, operator = self.current_token()
            
            # If we're in an if condition and trying to use ASSIGN (=) instead of EQ (==)
            if operator_type == 'ASSIGN':
                raise SyntaxError("Cannot use assignment operator (=) in conditions. Use equality operator (==) instead.")
            
            self.eat(operator_type)  # Consume operator
            right = self.parse_primary()  # Parse right side
            left = BinaryOperation(left, operator, right)
        
        return left

    def parse_primary(self):
        token_type, value = self.current_token()

        if token_type == 'OP' and value == '-':  # Handle unary minus
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

            if self.current_token()[0] == 'LBRACKET':  # Array access
                self.eat('LBRACKET')
                index = self.parse_expression()
                self.eat('RBRACKET')
                return ArrayAccess(ident, index)
            else:
                return Expression(ident)
        elif token_type == 'LPAREN':  # Explicitly handle parentheses here!
            self.eat('LPAREN')
            expr = self.parse_expression()
            self.eat('RPAREN')  # Expect closing parenthesis here explicitly
            return expr
        elif token_type == 'LBRACKET':  # Array literal
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
        self.eat('LBRACKET')  # Consume '['
        elements = []
        
        # Check for empty array
        if self.current_token()[0] == 'RBRACKET':
            self.eat('RBRACKET')
            return ArrayLiteral(elements)
        
        # Parse array elements
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
        name = self.eat('IDENT')  # Get the variable name
        if name not in self.declared_variables:
            raise SyntaxError(f"Variable '{name}' is not declared. Please declare it before assignment.")
        
        self.eat('ASSIGN')  # Consume '='
        
        # Check if the next token is an identifier followed by a left parenthesis (function call)
        if self.current_token()[0] == 'IDENT' and self.tokens[self.pos + 1][0] == 'LPAREN':
            func_name = self.eat('IDENT')
            self.eat('LPAREN')
            arguments = self.parse_argument_list()
            self.eat('RPAREN')
            value = FunctionCall(func_name, arguments)
        else:
            value = self.parse_expression()  # Parse the right-hand side expression
        
        # Consume the NEWLINE after the assignment
        if self.current_token()[0] == 'NEWLINE':
            self.eat('NEWLINE')
        
        return VariableDeclaration(None, name, value)  # No explicit type, it's reassignment

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
        
        if node.var_type == 'const' and node.name in self.variables:
            raise SyntaxError(f"Cannot reassign constant: {node.name}")
        
        # Type checking for int, float, and string declarations
        if node.is_array:
            # Check array size constraints if specified
            if node.array_size is not None and isinstance(value, list):
                if len(value) > node.array_size:
                    raise ValueError(f"List index is out of range: array '{node.name}' declared with size {node.array_size} but initialized with {len(value)} elements")
            
            # For arrays, check that all elements match the declared type
            if node.var_type == 'int':
                if not all(isinstance(item, int) for item in value):
                    raise SyntaxError(f"Type Error: Array '{node.name}' declared as int[] but contains non-integer values")
            elif node.var_type == 'float':
                if not all(isinstance(item, (int, float)) for item in value):
                    raise SyntaxError(f"Type Error: Array '{node.name}' declared as float[] but contains non-float values")
                # Convert integers to floats
                value = [float(item) if isinstance(item, int) else item for item in value]
            elif node.var_type == 'string':
                if not all(isinstance(item, str) for item in value):
                    raise SyntaxError(f"Type Error: Array '{node.name}' declared as string[] but contains non-string values")
        else:
            # For non-arrays, check the value directly
            if node.var_type == 'int':
                if not isinstance(value, int):
                    raise SyntaxError(f"Type Error: Variable '{node.name}' declared as int but assigned non-integer value '{value}'")
            elif node.var_type == 'float':
                if not isinstance(value, (int, float)):
                    raise SyntaxError(f"Type Error: Variable '{node.name}' declared as float but assigned non-float value '{value}'")
                # Convert integer to float if needed
                if isinstance(value, int):
                    value = float(value)
            elif node.var_type == 'string':
                if not isinstance(value, str):
                    raise SyntaxError(f"Type Error: Variable '{node.name}' declared as string but assigned non-string value '{value}'")
        
        self.variables[node.name] = value

    def evaluate_print_statement(self, node):
        value = self.evaluate(node.expression)
        print(value)

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
                    return  # Stop evaluating once an elif condition is true
                    
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

        # Save the current scope before function call
        old_variables = self.variables.copy()

        # Evaluate arguments
        arg_values = [self.evaluate(arg) for arg in node.arguments]

        # Check argument count
        if len(func.params) != len(arg_values):
            raise ValueError(f"Function '{node.name}' expected {len(func.params)} arguments but got {len(node.arguments)}")

        # Create a new local scope for this function call
        local_scope = {}

        # Bind parameters
        for (param_type, param_name, is_array), value in zip(func.params, arg_values):
            local_scope[param_name] = value

        # Set the function scope
        self.variables = local_scope

        # Execute function body
        result = None
        for statement in func.body:
            stmt_result = self.evaluate(statement)
            if isinstance(statement, ReturnStatement):
                result = stmt_result
                break  # Stop execution upon return

        # Restore the previous scope
        self.variables = old_variables

        if result is None:
            raise ValueError(f"Function '{node.name}' did not return a value")

        return result

    def evaluate_return_statement(self, node):
        # If the return statement contains a function call, evaluate it
        if isinstance(node.expression, FunctionCall):
            return self.evaluate_function_call(node.expression)
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
        obj = self.variables.get(node.object_name)
        if obj is None:
            raise ValueError(f"Object '{node.object_name}' is not defined")

        method = getattr(obj, node.method_name, None)
        if method is None:
            raise ValueError(f"Method '{node.method_name}' not found in object '{node.object_name}'")

        arguments = [self.evaluate(arg) for arg in node.arguments]
        return method(*arguments)

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

# Example usage
code = """
int a = 5
int b = 10
int b = 20
printk{b}
float pi = 3.14
const MAX_VALUE = 100
var temp = 20

if (a == b) begin 
    printk {"a equals b"}
elif (a > b) begin
    printk {"a is greater than b"}
else begin
    printk {"a not equals b"} 
    printk{a}
end

while (a < b) begin
    printk {"Incrementing a"}
    a = a + 2
end

repeat 2 begin 
    printk {"This is a repeat loop"}
end

int i = 0
for i in [1, 2] begin
    printk {"For loop iteration:"}
    printk{i}
end

func mul(int p,  int q) begin 
    int total = p * q
    return total
end 

int result = 0
result = mul(50, 10)
printk {result}

func sum_array(float arr[]) begin
    float sum = 0
    int num = 0
    for num in arr begin
        sum = sum + num
    end
    return sum
end

float numbers[4] = [1.5, 2, 3.5, 4.5]
float total_sum = sum_array(numbers)
printk {total_sum}
printk {numbers[1]}  

# Example usage - add these examples to your code section
string greeting = "Hello, World!"
printk {greeting}

string names[3] = ["Alice", "Bob", "Charlie"]
printk {names[1]}  # Should print "Bob"

# Concatenation example
string first = "Hello"
string last = "World"
string full = first + " " + last + "!"
printk {full}

printk {numbers}  # [1.5, 2.5, 3.5, 4.5]

numbers.remove(2)  # Remove element at index 2
printk {numbers}   # [1.5, 2.5, 4.5]

numbers.add(99.9)  # Add at the end
printk {numbers}   # [1.5, 2.5, 4.5, 99.9]

numbers.add(1, 50.5)  # Insert at index 1
printk {numbers}   # [1.5, 50.5, 2.5, 4.5, 99.9]

# Will produce an error:
# numbers.remove(10)  # IndexError: remove index out of range
"""
try:
    tokens = list(tokenize(code))  # Force tokenization to complete here
    parser = Parser(tokens)
    ast = parser.parse()
    evaluator = Evaluator(parser.functions)
    evaluator.evaluate(ast)
except SyntaxError as e:
    print(f"SyntaxError: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
