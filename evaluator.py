from ast_nodes import *

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
        elif isinstance(node, CompoundAssignment):
            self.evaluate_compound_assignment(node)
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

    def evaluate_compound_assignment(self, node):
        # Get the current value of the variable
        if node.name not in self.variables:
            raise ValueError(f"Variable '{node.name}' is not defined")
        
        current_value = self.variables[node.name]
        
        # Check if this is a const variable (can't reassign)
        if self.variables.get(f"__const_{node.name}"):
            raise SyntaxError(f"Cannot reassign constant: {node.name}")
        
        # Get the value to operate with
        value = self.evaluate(node.value)
        
        # Perform the operation based on the operator
        if node.operator == '+=':
            result = current_value + value
        elif node.operator == '-=':
            result = current_value - value
        elif node.operator == '*=':
            result = current_value * value
        elif node.operator == '/=':
            result = current_value / value
        elif node.operator == '%=':
            result = current_value % value
        else:
            raise ValueError(f"Unknown compound operator: {node.operator}")
        
        # Type checking 
        original_type = self.variables.get(f"__type_{node.name}")
        if original_type:
            # Check type compatibility based on original declaration
            if original_type == 'int' and not isinstance(result, int):
                # Convert result to int if it's a float without decimal part
                if isinstance(result, float) and result.is_integer():
                    result = int(result)
                else:
                    raise TypeError(f"Type error: Cannot assign {type(result).__name__} to variable '{node.name}' of type {original_type}")
            elif original_type == 'float' and not isinstance(result, (int, float)):
                raise TypeError(f"Type error: Cannot assign {type(result).__name__} to variable '{node.name}' of type {original_type}")
            elif original_type == 'string' and not isinstance(result, str):
                raise TypeError(f"Type error: Cannot assign {type(result).__name__} to variable '{node.name}' of type {original_type}")
        
        # Store the updated value
        self.variables[node.name] = result

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
