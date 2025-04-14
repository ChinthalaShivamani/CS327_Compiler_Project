from ast_nodes import *

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
            if next_token_type in ('ASSIGN', 'PLUS_ASSIGN', 'MINUS_ASSIGN', 'MULT_ASSIGN', 'DIV_ASSIGN', 'MOD_ASSIGN'):
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
        
        # Determine which assignment operator we have
        token_type, _ = self.current_token()
        if token_type == 'ASSIGN':
            operator = '='
            self.eat('ASSIGN')
        elif token_type == 'PLUS_ASSIGN':
            operator = '+='
            self.eat('PLUS_ASSIGN')
        elif token_type == 'MINUS_ASSIGN':
            operator = '-='
            self.eat('MINUS_ASSIGN')
        elif token_type == 'MULT_ASSIGN':
            operator = '*='
            self.eat('MULT_ASSIGN')
        elif token_type == 'DIV_ASSIGN':
            operator = '/='
            self.eat('DIV_ASSIGN')
        elif token_type == 'MOD_ASSIGN':
            operator = '%='
            self.eat('MOD_ASSIGN')
        else:
            raise SyntaxError(f"Expected assignment operator but got {token_type}")
        
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
        
        # Return appropriate AST node based on operator
        if operator == '=':
            return VariableDeclaration(None, name, value)
        else:
            return CompoundAssignment(name, operator, value)
