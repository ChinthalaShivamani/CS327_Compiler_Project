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
    
class ReturnStatement(ASTNode):  # Add this AST node
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
        ('IF', r'if'),  # If statement
        ('ELIF', r'elif'),  # Elif statement
        ('ELSE', r'else'),  # Else statement
        ('WHILE', r'while'),  # While loop
        ('FOR', r'for'),  # For loop
        ('FUNC', r'func\b'),  # Function keyword
        ('RETURN', r'return'),  # Return keyword
        ('IN', r'in\b'),  # 'in' keyword for for loops
        ('END', r'end\b'),  # 'end' keyword for loops
        ('NUMBER', r'\d+(\.\d+)?'),  # Numbers
        ('STRING', r'".*?"'),  # Strings
        ('IDENT', r'[a-zA-Z_][a-zA-Z_0-9]*'),  # Identifiers
        ('OP', r'[+\-*/%]'),  # Arithmetic operators
        ('ASSIGN', r'='),  # Assignment operator
        ('EQ', r'=='),  # Equality operator
        ('GE', r'>='),  # Greater than or equal to
        ('LE', r'<='),  # Less than or equal to
        ('GT', r'>'),  # Greater than
        ('LT', r'<'),  # Less than
        ('LPAREN', r'\('),  # Left parenthesis
        ('RPAREN', r'\)'),  # Right parenthesis
        ('LBRACE', r'\{'),  # Left brace
        ('RBRACE', r'\}'),  # Right brace
        ('COLON', r':'),  # Colon
        ('NEWLINE', r'\n'),  # Newline
        ('SKIP', r'[ \t]+'),  # Skip spaces and tabs
        ('COMMA', r','),  # Added COMMA token
        ('LBRACKET', r'\['),  # Left bracket for arrays
        ('RBRACKET', r'\]'),  # Right bracket for arrays
        ('DOT', r'\.'),  # Handle dot separately
        ('COMMENT', r'#.*'),  # Comments starting with #
        ('MISMATCH', r'[^ \t\n\w\d\+\-\*/%=<>:{}(),\[\]]'), # Unexpected characters
    ]
    tok_regex = '|'.join(f"(?P<{pair[0]}>{pair[1]})" for pair in token_specification)
    for match in re.finditer(tok_regex, code):
        kind = match.lastgroup
        value = match.group(kind)
        if kind == 'NUMBER':
            value = float(value) if '.' in value else int(value)
        elif kind == 'STRING':
            value = value.strip('"')
        elif kind in ('SKIP', 'COMMENT'):  # Skip spaces, tabs, and comments
            continue
        elif kind == 'MISMATCH':
            raise SyntaxError(f"Unexpected token: {value}")
        yield kind, value

class Parser:
    def __init__(self, tokens):
        self.tokens = list(tokens)
        self.pos = 0
        self.functions = {}  # Store user-defined functions

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
            elif value == 'int':  # Or other type keywords
                return self.parse_variable_declaration()
            elif value == 'printk':
                return self.parse_print_statement()
            elif value == 'repeat':
                return self.parse_repeat_loop()
            elif value == 'if':  # Handle 'if' directly here
                return self.parse_if_else()
            elif value == 'while':  # Handle 'while' directly here
                return self.parse_while_loop()
            elif value == 'for': # Handle 'for' directly here
                return self.parse_for_loop()
            elif value == 'func': # Handle 'func' directly here
                return self.parse_function_declaration()
            elif next_token_type == 'LPAREN':  # Function Call
                return self.parse_function_call()
            else:  # It's an expression or something else
                return self.parse_expression() # or handle other IDENT cases

        elif token_type in ('IF', 'WHILE', 'FOR', 'FUNC', 'REPEAT'):  # Alternative for keywords
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

        elif token_type == 'NEWLINE':
            self.eat('NEWLINE')
            return None  # Important: Return None for empty lines!
        elif token_type == 'RETURN':  # Handle return statements
            self.eat('RETURN')
            expression = self.parse_expression()
            self.eat('NEWLINE')
            return ReturnStatement(expression) # Assuming you have a ReturnStatement AST node
        else:
            current = self.current_token()
            if current[0] is not None:
                raise SyntaxError(f"Unknown statement or syntax: {current[1]}")
            else:
                return None # End of file
        
    def _parse_block(self):
        body = []
        while self.current_token()[0] not in ('RBRACE', 'END', None):
            statement = self.parse_statement()
            if statement:  # Only add if not None (handles empty lines)
                body.append(statement)
        if self.current_token()[0] == 'RBRACE':
            self.eat('RBRACE')
            if self.current_token()[0] == 'NEWLINE':
                self.eat('NEWLINE')
        elif self.current_token()[0] == 'END':
            self.eat('END')
            if self.current_token()[0] == 'NEWLINE':
                self.eat('NEWLINE')
        return body

    def parse_function_declaration(self):
        self.eat('FUNC')  # Ensure 'func' is eaten
        name = self.eat('IDENT')  # Expect function name
        self.eat('LPAREN')  # Expect '('
        params = []
        
        while self.current_token()[0] == 'IDENT':  # Read function parameters
            params.append(self.eat('IDENT'))
            if self.current_token()[0] == 'COMMA':
                self.eat('COMMA')
        
        self.eat('RPAREN')  # Expect ')'
        self.eat('COLON')  # Expect ':'
        
        # Consume NEWLINE if present
        if self.current_token()[0] == 'NEWLINE':
            self.eat('NEWLINE')

        body = self._parse_block()
        
        # Store the function in the functions dictionary
        self.functions[name] = FunctionDeclaration(name, params, body)
        print(f"Stored function: {name}")  # Debug print
        return self.functions[name]  # Return the function declaration

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
        var_type = self.eat('IDENT')  # Expect 'int' or other types
        name = self.eat('IDENT')  # Variable name

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
        self.eat('IF')
        self.eat('LPAREN')
        condition = self.parse_expression()
        self.eat('RPAREN')
        self.eat('COLON')  # Expect colon after condition

        # Consume NEWLINE if present
        if self.current_token()[0] == 'NEWLINE':
            self.eat('NEWLINE')

        # Parse the true branch
        true_branch = []
        while self.current_token()[0] not in ['ELIF', 'ELSE', 'NEWLINE', None]:
            true_branch.append(self.parse_statement())

        # Consume NEWLINE if present before elif or else
        if self.current_token()[0] == 'NEWLINE':
            self.eat('NEWLINE')

        # Parse elif branches (if any)
        elif_branches = []
        while self.current_token()[0] == 'ELIF':
            self.eat('ELIF')
            self.eat('LPAREN')
            elif_condition = self.parse_expression()
            self.eat('RPAREN')
            self.eat('COLON')  # Expect colon after elif condition

            # Consume NEWLINE if present
            if self.current_token()[0] == 'NEWLINE':
                self.eat('NEWLINE')

            # Parse the elif branch
            elif_body = []
            while self.current_token()[0] not in ['ELIF', 'ELSE', 'NEWLINE', None]:
                elif_body.append(self.parse_statement())

            elif_branches.append((elif_condition, elif_body))

            # Consume NEWLINE if present before next elif or else
            if self.current_token()[0] == 'NEWLINE':
                self.eat('NEWLINE')

        # Parse the else branch (mandatory)
        if self.current_token()[0] != 'ELSE':
            raise SyntaxError("Expected 'else' block after 'if' and 'elif' conditions")
        
        self.eat('ELSE')
        self.eat('COLON')  # Expect colon after else

        # Consume NEWLINE if present
        if self.current_token()[0] == 'NEWLINE':
            self.eat('NEWLINE')

        false_branch = []
        while self.current_token()[0] not in ['NEWLINE', None]:
            false_branch.append(self.parse_statement())

        return IfElse(condition, true_branch, elif_branches, false_branch)

    def parse_while_loop(self):
        self.eat('WHILE')
        self.eat('LPAREN')
        condition = self.parse_expression()
        self.eat('RPAREN')
        self.eat('COLON')  # Expect colon after condition
        self.eat('NEWLINE')
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

        if token_type == 'IDENT' and self.tokens[self.pos + 1][0] == 'LPAREN':
            func_name = self.eat('IDENT')
            if func_name in self.functions:  # Check if it's a user-defined function
                self.eat('LPAREN')
                arguments = self.parse_argument_list()
                self.eat('RPAREN')
                return FunctionCall(func_name, arguments)
            else:
                raise SyntaxError(f"Undefined function: {func_name}")

        # Handle binary operations or primary expressions
        left = self.parse_primary()
        while self.current_token()[0] in ['OP', 'LT', 'GT', 'EQ', 'GE', 'LE']:
            operator = self.eat(self.current_token()[0])  # Eat OP, LT, GT, EQ, GE, or LE
            right = self.parse_primary()
            left = BinaryOperation(left, operator, right)

        return left

    def parse_primary(self):
        token_type, value = self.current_token()
        if token_type == 'OP' and value == '-':  # Handle unary minus
            self.eat('OP')
            operand = self.parse_primary()
            return BinaryOperation(Expression(0), '-', operand)  # Convert `-x` to `0 - x`
        if token_type == 'NUMBER':
            self.eat('NUMBER')
            return Expression(value)
        elif token_type == 'STRING':
            self.eat('STRING')
            return Expression(value)
        elif token_type == 'IDENT':
            ident = self.eat('IDENT')
            if self.current_token()[0] == 'LBRACKET':  # Array access
                self.eat('LBRACKET')
                index = self.parse_expression()
                self.eat('RBRACKET')
                return ArrayAccess(ident, index)
            else:
                return Expression(ident)
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
        name = self.eat('IDENT')  # Get the variable name (e.g., 'a')
        self.eat('ASSIGN')  # Consume '='
        value = self.parse_expression()  # Parse the right-hand side expression (e.g., 'a + 1')
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

            # Evaluate else branch
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
            raise ValueError(f"Function {node.name} not defined")
        old_variables = self.variables.copy()
        for param, arg in zip(func.params, node.arguments):
            self.variables[param] = self.evaluate(arg)
        result = None
        for statement in func.body:
            result = self.evaluate(statement)
            if isinstance(statement, ReturnStatement):
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

# Example usage
# Example usage
code = """
int a = 5
int b = 10

if (a > b):
    printk {"a is greater"}
else:
    printk {"b is greater"}


int x = 10
int y = x * 2
printk {y}

func div(p, q):
    total = p / q
    printk {total} 
end

div(5, 3) 

while (a < b):
    printk {"Incrementing a"}
    a = a + 2
end


if (a <= b):
    printk {"a is less than or equal to b"}
elif (a > b):
    printk {"a is greater than or equal to b"}
else:
    printk {"a equals b"}
"""

tokens = tokenize(code)
parser = Parser(tokens)
try:
    ast = parser.parse()
    evaluator = Evaluator(parser.functions)  # Pass the functions dictionary to the Evaluator
    evaluator.evaluate(ast)
except SyntaxError as e:
    print(f"Error: {e}")
