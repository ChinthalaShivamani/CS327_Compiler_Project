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
    def __init__(self, var_type, name, value):
        self.var_type = var_type
        self.name = name
        self.value = value

    def __repr__(self):
        return f"VariableDeclaration({self.var_type}, {self.name}, {self.value})"

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

# Tokenizer
def tokenize(code):
    token_specification = [
        ('INT', r'int\b'),              # int keyword
        ('FLOAT', r'float\b'),          # float keyword
        ('CONST', r'const\b'),          # const keyword
        ('VAR', r'var\b'),              # var keyword
        ('IF', r'if\b'),                # if keyword
        ('ELIF', r'elif\b'),            # elif keyword
        ('ELSE', r'else\b'),            # else keyword
        ('WHILE', r'while\b'),          # while keyword
        ('FOR', r'for\b'),              # for keyword
        ('FUNC', r'func\b'),            # func keyword
        ('RETURN', r'return\b'),        # return keyword
        ('IN', r'in\b'),                # in keyword
        ('BEGIN', r'begin\b'),          # begin keyword
        ('END', r'end\b'),              # end keyword

        # ❗ Multi-character operators FIRST:
        ('EQ', r'=='),                  # Equality operator (must be BEFORE ASSIGN)
        ('GE', r'>='),                  # Greater than or equal to
        ('LE', r'<='),                  # Less than or equal to

        # Single-character operators AFTER:
        ('ASSIGN', r'='),               # Assignment operator
        ('GT', r'>'),                   # Greater than
        ('LT', r'<'),                   # Less than

        ('NUMBER', r'\d+(\.\d+)?'),     # Numbers (including floats)
        ('STRING', r'".*?"'),           # Strings
        ('IDENT', r'[a-zA-Z_][a-zA-Z_0-9]*'),  # Identifiers
        ('OP', r'[+\-*/%]'),            # Arithmetic operators
        ('LPAREN', r'\('),              # Left parenthesis
        ('RPAREN', r'\)'),              # Right parenthesis
        ('LBRACE', r'\{'),              # Left brace
        ('RBRACE', r'\}'),              # Right brace
        ('COLON', r':'),                # Colon symbol ':'
        ('COMMA', r','),                # Comma ','
        ('LBRACKET', r'\['),            # Left bracket '[' for arrays
        ('RBRACKET', r'\]'),            # Right bracket ']' for arrays
        ('DOT', r'\.'),                 # Dot '.'
        
        ('NEWLINE', r'\n'),             # Newline character '\n'
        
        ('SKIP', r'[ \t]+'),            # Skip spaces and tabs
        ('COMMENT', r'#.*'),            # Comments starting with '#'
        
        ('MISMATCH', r'.')              # Any other unexpected character (catch-all)
    ]

    tok_regex = '|'.join(f"(?P<{pair[0]}>{pair[1]})" for pair in token_specification)
    for match in re.finditer(tok_regex, code):
        kind = match.lastgroup
        value = match.group(kind)
        
        if kind == 'NUMBER':
            value = float(value) if '.' in value else int(value)
            
        elif kind == 'STRING':
            value = value.strip('"')
            
        elif kind in ['SKIP', 'COMMENT']:
            continue
            
        elif kind == 'MISMATCH':
            raise SyntaxError(f"Unexpected token: {value}")
            
        yield kind, value


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
                self.eat('NEWLINE')  # Expect a newline after the function call
                return FunctionCall(func_name, arguments)
            elif value == 'printk':
                return self.parse_print_statement()
            elif value == 'repeat':
                return self.parse_repeat_loop()
            elif value == 'if':  # Handle 'if' directly here
                return self.parse_if_else()
            elif value == 'while':  # Handle 'while' directly here
                return self.parse_while_loop()
            elif value == 'for':  # Handle 'for' directly here
                return self.parse_for_loop()
            elif value == 'func':  # Handle 'func' directly here
                return self.parse_function_declaration()
            elif next_token_type == 'LPAREN':  # Function Call
                return self.parse_function_call()
            else:  # It's an expression or something else
                return self.parse_expression()  # or handle other IDENT cases
        elif token_type in ('INT', 'FLOAT', 'CONST', 'VAR'):  # Handle int, float, const, and var declarations
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
            elif token_type == 'REPEAT':
                return self.parse_repeat_loop()
            elif token_type == 'ELSE' or token_type == 'ELIF':
                # 'else' and 'elif' should only appear as part of an if-else structure
                # If we encounter them on their own, it's a syntax error
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
            # 'end' should be consumed by the block parsing methods like _parse_block
            # If we encounter it here, it's likely part of a block structure
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

        return body

    def parse_function_declaration(self):
        self.eat('FUNC')  # Consume 'func'
        name = self.eat('IDENT')  # Function name
        self.declared_variables.add(name)
        self.eat('LPAREN')  # Consume '('
        params = []
        while self.current_token()[0] != 'RPAREN':  # Parse parameters until ')'
            param_type = self.eat(self.current_token()[0])  # Expect type (e.g., 'int', 'float')
            if param_type not in ('int', 'float', 'const', 'var'):  # Validate parameter type
                raise SyntaxError(f"Unexpected parameter type: {param_type}")
            param_name = self.eat('IDENT')  # Expect parameter name
            params.append((param_type, param_name))  # Store as a tuple (type, name)
            if self.current_token()[0] == 'COMMA':  # Handle comma-separated parameters
                self.eat('COMMA')
        self.eat('RPAREN')  # Consume ')'

        # Add parameters to declared variables (local scope for this function)
        for param_type, param_name in params:
            self.declared_variables.add(param_name)

        self.eat('BEGIN')  # Consume 'begin'
        body = self._parse_block()  # Parse the function body
        #self.eat('END')  # Consume 'end'

        func_decl = FunctionDeclaration(name, params, body)
        self.functions[name] = func_decl
        return func_decl

    def parse_function_call(self):
        func_name = self.eat('IDENT')  # Function name
        self.eat('LPAREN')  # Left parenthesis
        arguments = self.parse_argument_list()  # Parse arguments
        self.eat('RPAREN')  # Right parenthesis
        return FunctionCall(func_name, arguments)

    def parse_argument_list(self):
        arguments = []
        while self.current_token()[0] != 'RPAREN':
            arguments.append(self.parse_expression())
            if self.current_token()[0] == 'COMMA':
                self.eat('COMMA')
        return arguments

    def parse_variable_declaration(self):
        # Handle int, float, const, and var declarations
        var_type = self.current_token()[1]  # Get the type (int, float, const, var)
        if var_type not in ('int', 'float', 'const', 'var'):
            raise SyntaxError(f"Unexpected type: {var_type}")
        self.eat(self.current_token()[0])  # Eat the type token (INT, FLOAT, CONST, VAR)
        name = self.eat('IDENT')  # Variable name
        self.declared_variables.add(name)

        # Check if there's an assignment following the variable declaration (e.g., 'int a = 5')
        if self.current_token()[0] == 'ASSIGN':
            self.eat('ASSIGN')  # Consume '='
            value = self.parse_expression()  # Parse the right-hand side expression
            self.eat('NEWLINE')  # Consume the newline after declaration
            return VariableDeclaration(var_type, name, value)
        else:
            self.eat('NEWLINE')  # Consume the newline after declaration
            return VariableDeclaration(var_type, name, None)

    def parse_print_statement(self):
        self.eat('IDENT')  # Expecting 'printk'
        self.eat('LBRACE')  # Expecting '{'
        expression = self.parse_expression()  # Parse the expression inside the braces
        self.eat('RBRACE')  # Expecting '}'
        return PrintStatement(expression)

    def parse_repeat_loop(self):
        self.eat('REPEAT')
        count = self.parse_expression()
        self.eat('COLON')
        self.eat('NEWLINE')
        body = self._parse_block()
        return RepeatLoop(count, body)

    def parse_if_else(self):
        self.eat('IF')  # Consume 'if'
        self.eat('LPAREN')  # Consume '('
        condition = self.parse_expression()  # Parse condition
        self.eat('RPAREN')  # Consume ')'
        self.eat('BEGIN')  # Consume 'begin'
        
        true_branch = []
        while self.current_token()[0] not in ('ELSE', 'END', None):
            statement = self.parse_statement()
            if statement:
                true_branch.append(statement)
            # Skip newlines
            while self.current_token()[0] == 'NEWLINE':
                self.eat('NEWLINE')
        
        false_branch = []
        elif_branches = []
        
        # Handle 'else' branch if present
        if self.current_token()[0] == 'ELSE':
            self.eat('ELSE')
            self.eat('BEGIN')
            
            # Parse the 'else' block until 'END'
            while self.current_token()[0] not in ('END', None):
                statement = self.parse_statement()
                if statement:
                    false_branch.append(statement)
                # Skip newlines
                while self.current_token()[0] == 'NEWLINE':
                    self.eat('NEWLINE')
        
        # Consume the single 'end' token at the end of the if-else structure
        if self.current_token()[0] == 'END':
            self.eat('END')
        else:
            raise SyntaxError("Expected 'end' at the end of if-else block")
        
        return IfElse(condition, true_branch, elif_branches, false_branch)

    def parse_while_loop(self):
        self.eat('WHILE')
        self.eat('LPAREN')
        condition = self.parse_expression()
        self.eat('RPAREN')
        self.eat('BEGIN')  # changed from COLON to BEGIN
        body = self._parse_block()
        return WhileLoop(condition, body)

    def parse_for_loop(self):
        self.eat('FOR')
        loop_var = self.eat('IDENT')
        self.eat('IN')
        iterable = self.parse_expression()
        self.eat('COLON')  # Expect colon after iterable
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

            # Check for array access
            if self.current_token()[0] == 'LBRACKET':
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
            self.eat('LBRACKET')  # Consume '['
            elements = []
            while self.current_token()[0] != 'RBRACKET':  # Parse elements until ']'
                elements.append(self.parse_expression())
                if self.current_token()[0] == 'COMMA':  # Handle comma-separated elements
                    self.eat('COMMA')
            self.eat('RBRACKET')  # Ensure closing ']'
            left = ArrayLiteral(elements)

        # Handle unary minus (e.g., -x)
        elif token_type == 'OP' and value == '-':
            self.eat('OP')  # Consume '-'
            operand = self.parse_expression()
            left = BinaryOperation(Expression(0), '-', operand)  # Convert `-x` to `0 - x`

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
        else:
            raise ValueError(f"Unknown node type: {type(node)}")

    def evaluate_variable_declaration(self, node):
        value = self.evaluate(node.value) if node.value else None

        if node.var_type == 'const' and node.name in self.variables:
            raise SyntaxError(f"Cannot reassign constant: {node.name}")

        # Type checking for int and float declarations
        if node.var_type == 'int':
            if not isinstance(value, int):
                raise SyntaxError(f"Type Error: Variable '{node.name}' declared as int but assigned non-integer value '{value}'")
        elif node.var_type == 'float':
            if not isinstance(value, float):
                raise SyntaxError(f"Type Error: Variable '{node.name}' declared as float but assigned non-float value '{value}'")

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

        old_variables = self.variables.copy()  # Save current variable scope

        # Bind arguments to parameters
        if len(func.params) != len(node.arguments):
            raise ValueError(f"Function '{node.name}' expected {len(func.params)} arguments but got {len(node.arguments)}")

        for (param_type, param_name), arg in zip(func.params, node.arguments):
            value = self.evaluate(arg)

            # Type checking for parameters
            if param_type == 'int' and not isinstance(value, int):
                raise SyntaxError(f"Type Error: Parameter '{param_name}' expected an integer but got '{value}'")
            elif param_type == 'float' and not isinstance(value, float):
                raise SyntaxError(f"Type Error: Parameter '{param_name}' expected a float but got '{value}'")

            self.variables[param_name] = value

        result = None
        for statement in func.body:
            result = self.evaluate(statement)
            if isinstance(statement, ReturnStatement):  # Handle return statements
                break

        self.variables = old_variables  # Restore previous variable scope
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

# Example usage
code = """
int a = 5
int b = 10

if (a == b) begin
    printk {"a equals b"}
else begin
    printk {"a not equals b"}
end

while (a < b) begin
    printk {"Incrementing a"}
    a = a + 2
end
func mul(int p, int q) begin
    int total = p * q
    return total
end 
int result = 0
result = mul(50, 10)
printk {result}
"""

tokens = tokenize(code)
parser = Parser(tokens)
try:
    ast = parser.parse()
    evaluator = Evaluator(parser.functions)
    evaluator.evaluate(ast)
except SyntaxError as e:
    print(f"Error: {e}")
