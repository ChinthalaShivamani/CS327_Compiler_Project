import re
# Define AST nodes
class ASTNode:
    pass

class Evaluator:
    def __init__(self,ast):
        self.ast = ast
        self.variables = {}
        self.functions = {}
        self.built_in_functions = {
            "sum": sum,
            "sqrt": lambda x: x ** 0.5,
            "pow": lambda x, y: x ** y,
            "abs": abs,
            "max": max,
            "min": min,
            "floor": lambda x: int(x // 1),
            "ceil": lambda x: int(x) + (1 if x % 1 > 0 else 0)
        }
    
    def evaluate(self, node):
        if isinstance(node, Program):
            for stmt in node.statements:
                self.evaluate(stmt)
        
        elif isinstance(node, VariableDeclaration):
            self.variables[node.name] = self.evaluate(node.value) if node.value else None
        
        elif isinstance(node, PrintStatement):
            print(self.evaluate(node.expression))
        
        elif isinstance(node, RepeatLoop):
            count = self.evaluate(node.count)
            for _ in range(count):
                for stmt in node.body:
                    self.evaluate(stmt)
        
        elif isinstance(node, BinaryOperation):
            left = self.evaluate(node.left)
            right = self.evaluate(node.right)
            if node.operator == '+': return left + right
            if node.operator == '-': return left - right
            if node.operator == '*': return left * right
            if node.operator == '/': return left / right
            if node.operator == '%': return left % right
            if node.operator == '<': return left < right
            if node.operator == '>': return left > right
        
        elif isinstance(node, Expression):
            if isinstance(node.value, str) and node.value in self.variables:
                return self.variables[node.value]
            return node.value
        
        elif isinstance(node, IfElse):
            condition = self.evaluate(node.condition)
            if condition:
                for stmt in node.true_branch:
                    self.evaluate(stmt)
            else:
                for stmt in node.false_branch:
                    self.evaluate(stmt)
        
        elif isinstance(node, WhileLoop):
            while self.evaluate(node.condition):
                for stmt in node.body:
                    self.evaluate(stmt)
        
        elif isinstance(node, FunctionDeclaration):
            self.functions[node.name] = (node.params, node.body)
        
        elif isinstance(node, FunctionCall):
            if node.name in self.built_in_functions:
                args = [self.evaluate(arg) for arg in node.arguments]
                return self.built_in_functions[node.name](*args)
            elif node.name in self.functions:
                params, body = self.functions[node.name]
                local_vars = dict(zip(params, [self.evaluate(arg) for arg in node.arguments]))
                prev_vars = self.variables.copy()
                self.variables.update(local_vars)
                for stmt in body:
                    self.evaluate(stmt)
                self.variables = prev_vars
            else:
                raise RuntimeError(f"Undefined function: {node.name}")
        
        elif isinstance(node, ArrayLiteral):
            return [self.evaluate(element) for element in node.elements]
        
        elif isinstance(node, ArrayAccess):
            array = self.variables.get(node.array)
            index = self.evaluate(node.index)
            if isinstance(array, list):
                return array[index]
            else:
                raise RuntimeError(f"'{node.array}' is not an array")
        
        else:
            raise RuntimeError(f"Unknown AST Node: {node}")
        
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
    def __init__(self, condition, true_branch, false_branch):
        self.condition = condition
        self.true_branch = true_branch
        self.false_branch = false_branch

    def __repr__(self):
        return f"IfElse({self.condition}, {self.true_branch}, {self.false_branch})"

class WhileLoop(ASTNode):
    def __init__(self, condition, body):
        self.condition = condition
        self.body = body

    def __repr__(self):
        return f"WhileLoop({self.condition}, {self.body})"

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
    ('ELSE', r'else'),  # Else statement
    ('WHILE', r'while'),  # While loop
    ('NUMBER', r'\d+(\.\d+)?'),  # Numbers
    ('STRING', r'".*?"'),  # Strings
    ('IDENT', r'[a-zA-Z_][a-zA-Z_0-9]*'),  # Identifiers
    ('OP', r'[+\-*/%]'),  # Arithmetic operators
    ('ASSIGN', r'='),  # Assignment operator
    ('LPAREN', r'\('),  # Left parenthesis
    ('RPAREN', r'\)'),  # Right parenthesis
    ('LBRACE', r'\{'),  # Left brace
    ('RBRACE', r'\}'),  # Right brace
    ('COLON', r':'),  # Colon
    ('NEWLINE', r'\n'),  # Newline
    ('SKIP', r'[ \t]+'),  # Skip spaces and tabs
    ('LT', r'<'),  # Less than
    ('GT', r'>'),  # Greater than
    ('FUNC', r'func'),  # Function keyword
    ('FUNC_NAME', r'[a-zA-Z_][a-zA-Z_0-9]*'),  # Adjust this regex if necessary
    ('RETURN', r'return'),  # Return keyword
    ('COMMA', r','),  # Added COMMA token
    ('LBRACKET', r'\['),  # Left bracket for arrays
    ('RBRACKET', r'\]'),  # Right bracket for arrays
    ('MISMATCH', r'.'),  # Any other character
]

    tok_regex = '|'.join(f"(?P<{pair[0]}>{pair[1]})" for pair in token_specification)
    for match in re.finditer(tok_regex, code):
        kind = match.lastgroup
        value = match.group(kind)
        if kind == 'NUMBER':
            value = float(value) if '.' in value else int(value)
        elif kind == 'STRING':
            value = value.strip('"')
        elif kind == 'SKIP':
            continue
        elif kind == 'MISMATCH':
            raise SyntaxError(f"Unexpected token: {value}")
        yield kind, value

# Parser
class Parser:
    def __init__(self, tokens):
        self.tokens = list(tokens)
        self.pos = 0
        self.built_in_functions = {  # Store built-in function names
            "sum": None,
            "sqrt": None,
            "pow": None,
            "abs": None,
            "max": None,
            "min": None,
            "floor": None,
            "ceil": None,
        }

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

    def parse_statement(self):  # Crucial change here!
        token_type, value = self.current_token()

        if token_type == 'IDENT':
            next_token_type = self.tokens[self.pos + 1][0] if self.pos + 1 < len(self.tokens) else None
            
            if next_token_type == 'ASSIGN':
                return self.parse_assignment()
            elif value == 'int':  
                return self.parse_variable_declaration()
            elif value == 'printk':
                return self.parse_print_statement()
            elif value == 'repeat':
                return self.parse_repeat_loop()
            elif next_token_type == 'LPAREN':  # Function Call
                return self.parse_expression() # this is the solution
            else:
                return self.parse_expression()

        elif token_type == 'IF':
            return self.parse_if_else()
        elif token_type == 'WHILE':
            return self.parse_while_loop()
        elif token_type == 'FUNC':
            return self.parse_function_declaration()
        elif token_type == 'RETURN':
            return self.parse_return_statement()
        elif token_type == 'NEWLINE':  
            self.eat('NEWLINE')
            return None
        else:
            raise SyntaxError(f"Unknown statement or syntax: {value}")  # This was triggering your error


    def parse_function_declaration(self):
        self.eat('FUNC')  # "func" keyword
        name = self.eat('IDENT')  # Function name (e.g., sum)
        self.eat('LPAREN')  # Left parenthesis after function name
        params = []
        while self.current_token()[0] == 'IDENT':  # Parameter names
            params.append(self.eat('IDENT'))
            if self.current_token()[0] == 'COMMA':  # Handle comma-separated parameters
                self.eat('COMMA')
        self.eat('RPAREN')  # Right parenthesis
        self.eat('COLON')  # Colon
        body = []
        while self.current_token()[0] != 'NEWLINE' and self.current_token()[0] is not None:
            body.append(self.parse_statement())  # Parse body statements
        self.eat('NEWLINE')  # Consume the newline after the function body
        return FunctionDeclaration(name, params, body)

    def parse_return_statement(self):
        self.eat('RETURN')  # "return" keyword
        expression = self.parse_expression()  # Parse the return value
        return Expression(expression)  # Return as a value expression (you can modify this if needed)
    
    def parse_variable_declaration(self):
        var_type = self.eat('IDENT')  # Expect 'int' or other types
        name = self.eat('IDENT')  # Variable name

        # Check if there's an assignment following the variable declaration (e.g., 'int a = 5')
        if self.current_token()[0] == 'ASSIGN':
            self.eat('ASSIGN')  # Consume '='
            value = self.parse_expression()  # Parse the right-hand side expression
            return VariableDeclaration(var_type, name, value)
        else:
            # Handle the case where the variable is declared but not assigned
            return VariableDeclaration(var_type, name, None)


    def parse_print_statement(self):
        self.eat('IDENT')  # Expecting 'printk'
        self.eat('LBRACE')  # Expecting '{'
        expression = self.parse_expression()  # Parse the expression inside the braces
        self.eat('RBRACE')  # Expecting '}'
        return PrintStatement(expression)


    def parse_repeat_loop(self):
        self.eat('IDENT')  # repeat
        count = self.parse_expression()
        self.eat('COLON')
        body = []
        while self.current_token()[0] != 'NEWLINE' and self.current_token()[0] is not None:
            body.append(self.parse_statement())
        self.eat('NEWLINE')  # Consume the newline after the loop body
        return RepeatLoop(count, body)

    def parse_if_else(self):
        self.eat('IF')
        self.eat('LPAREN')
        condition = self.parse_expression()
        self.eat('RPAREN')
        self.eat('COLON')

        # Consume NEWLINE if present
        if self.current_token()[0] == 'NEWLINE':
            self.eat('NEWLINE')

        # Parse the true branch
        true_branch = []
        while self.current_token()[0] not in ['ELSE', 'NEWLINE', None]:
            true_branch.append(self.parse_statement())

        # Consume NEWLINE if present before else
        if self.current_token()[0] == 'NEWLINE':
            self.eat('NEWLINE')

        # Parse the else branch (if present)
        false_branch = []
        if self.current_token()[0] == 'ELSE':
            self.eat('ELSE')
            self.eat('COLON')

            # Consume NEWLINE if present
            if self.current_token()[0] == 'NEWLINE':
                self.eat('NEWLINE')

            while self.current_token()[0] not in ['NEWLINE', None]:
                false_branch.append(self.parse_statement())

        return IfElse(condition, true_branch, false_branch)
    
    def parse_while_loop(self):
        self.eat('WHILE')  #while
        self.eat('LPAREN')
        condition = self.parse_expression()  # Parse the loop condition
        self.eat('RPAREN')
        self.eat('COLON')

        # Consume NEWLINE if present
        if self.current_token()[0] == 'NEWLINE':
            self.eat('NEWLINE')

        body = []
        while self.current_token()[0] not in ['NEWLINE', None]:
            body.append(self.parse_statement())

        return WhileLoop(condition, body)


    def parse_expression(self):
        token_type, value = self.current_token()

        if token_type == 'FUNC_NAME':  # Function Call
            func_name = self.eat('FUNC_NAME')
            if func_name in self.built_in_functions:  # Check if it's a built-in
                self.eat('LPAREN')
                arguments = self.parse_argument_list()
                self.eat('RPAREN')
                return FunctionCall(func_name, arguments)
            else:
                 raise SyntaxError(f"Undefined function: {func_name}")
        elif token_type == 'IDENT' and self.tokens[self.pos+1][0] == 'LPAREN':
            func_name = self.eat('IDENT')
            if func_name in self.built_in_functions:  # Check if it's a built-in
                self.eat('LPAREN')
                arguments = self.parse_argument_list()
                self.eat('RPAREN')
                return FunctionCall(func_name, arguments)
            else:
                 raise SyntaxError(f"Undefined function: {func_name}")

        # Handle binary operations or primary expressions (No changes needed here)
        left = self.parse_primary()
        while self.current_token()[0] in ['OP', 'LT', 'GT']:
            operator = self.eat(self.current_token()[0])  # Eat OP, LT, or GT
            right = self.parse_primary()
            left = BinaryOperation(left, operator, right)

        return left

    def parse_argument_list(self):
        arguments = []
        while self.current_token()[0] != 'RPAREN':
            arguments.append(self.parse_expression())
            if self.current_token()[0] == 'COMMA':
                self.eat('COMMA')
        return arguments

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
# Example usage
if __name__ == "__main__":
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

arr = [1, 2, 3, 4, 5]
printk {arr[2]}

printk {sum(arr)}
printk {sqrt(16)}
printk {pow(2, 3)}
printk {abs(-10)}
printk {max(1, 2, 3, 4, 5)}
printk {min(1, 2, 3, 4, 5)}
printk {floor(3.7)}
printk {ceil(3.2)}
"""
    tokens = tokenize(code)
    parser = Parser(tokens)
    try:
        ast = parser.parse()
        evaluator = Evaluator(ast)
        evaluator.evaluate(ast)
    except SyntaxError as e:
        print(f"Error: {e}")