import re
import math
from example import Parser
from example import tokenize
from example import (
    Program,
    VariableDeclaration,
    PrintStatement,
    RepeatLoop,
    BinaryOperation,
    Expression,
    IfElse,
    WhileLoop,
    ForLoop,
    FunctionDeclaration,
    FunctionCall,
    ReturnStatement,
    ArrayLiteral,
    ArrayAccess,
    MethodCall
)
class IntermediateCodeGenerator:
    def __init__(self):
        self.code = []
        self.temp_counter = 0
        self.label_counter = 0
        self.current_function = None
        self.function_table = {}
        
    def new_temp(self):
        temp = f"t{self.temp_counter}"
        self.temp_counter += 1
        return temp
        
    def new_label(self):
        label = f"L{self.label_counter}"
        self.label_counter += 1
        return label
        
    def generate(self, node):
        if isinstance(node, Program):
            for statement in node.statements:
                self.generate(statement)
        elif isinstance(node, VariableDeclaration):
            self.generate_variable_declaration(node)
        elif isinstance(node, PrintStatement):
            self.generate_print_statement(node)
        elif isinstance(node, RepeatLoop):
            self.generate_repeat_loop(node)
        elif isinstance(node, BinaryOperation):
            return self.generate_binary_operation(node)
        elif isinstance(node, Expression):
            return self.generate_expression(node)
        elif isinstance(node, IfElse):
            self.generate_if_else(node)
        elif isinstance(node, WhileLoop):
            self.generate_while_loop(node)
        elif isinstance(node, ForLoop):
            self.generate_for_loop(node)
        elif isinstance(node, FunctionDeclaration):
            self.generate_function_declaration(node)
        elif isinstance(node, FunctionCall):
            return self.generate_function_call(node)
        elif isinstance(node, ReturnStatement):
            return self.generate_return_statement(node)
        elif isinstance(node, ArrayLiteral):
            return self.generate_array_literal(node)
        elif isinstance(node, ArrayAccess):
            return self.generate_array_access(node)
        elif isinstance(node, MethodCall):
            return self.generate_method_call(node)
        else:
            raise ValueError(f"Unknown node type: {type(node).__name__}")
        
        return None
    
    def generate_variable_declaration(self, node):
        if node.value is not None:
            value = self.generate(node.value)
            if value is not None:
                self.code.append(f"{node.name} = {value}")
    
    def generate_print_statement(self, node):
        value = self.generate(node.expression)
        self.code.append(f"print {value}")
    
    def generate_repeat_loop(self, node):
        count = self.generate(node.count)
        loop_var = self.new_temp()
        end_label = self.new_label()
        
        self.code.append(f"{loop_var} = 0")
        loop_start = self.new_label()
        self.code.append(f"{loop_start}:")
        
        # Loop condition
        self.code.append(f"if {loop_var} >= {count} goto {end_label}")
        
        # Loop body
        for statement in node.body:
            self.generate(statement)
            
        # Increment and loop back
        self.code.append(f"{loop_var} = {loop_var} + 1")
        self.code.append(f"goto {loop_start}")
        
        # End label
        self.code.append(f"{end_label}:")
    
    def generate_binary_operation(self, node):
        left = self.generate(node.left)
        right = self.generate(node.right)
        temp = self.new_temp()
        
        if node.operator == '+':
            self.code.append(f"{temp} = {left} + {right}")
        elif node.operator == '-':
            self.code.append(f"{temp} = {left} - {right}")
        elif node.operator == '*':
            self.code.append(f"{temp} = {left} * {right}")
        elif node.operator == '/':
            self.code.append(f"{temp} = {left} / {right}")
        elif node.operator == '%':
            self.code.append(f"{temp} = {left} % {right}")
        elif node.operator == '<':
            self.code.append(f"{temp} = {left} < {right}")
        elif node.operator == '>':
            self.code.append(f"{temp} = {left} > {right}")
        elif node.operator == '==':
            self.code.append(f"{temp} = {left} == {right}")
        elif node.operator == '>=':
            self.code.append(f"{temp} = {left} >= {right}")
        elif node.operator == '<=':
            self.code.append(f"{temp} = {left} <= {right}")
        else:
            raise ValueError(f"Unknown operator: {node.operator}")
            
        return temp
    
    def generate_expression(self, node):
        if isinstance(node.value, (int, float)):
            return str(node.value)
        elif isinstance(node.value, str):
            # Handle string literals by escaping quotes if needed
            escaped = node.value.replace('"', '\\"')
            return f'"{escaped}"'
        elif isinstance(node.value, Expression):
            # Handle nested expressions
            return self.generate(node.value)
        else:
            # For variables and other cases
            return str(node.value)
        
    def generate_if_else(self, node):
        condition = self.generate(node.condition)
        false_label = self.new_label()
        end_label = self.new_label()
        
        # If condition
        self.code.append(f"if not {condition} goto {false_label}")
        
        # True branch
        for statement in node.true_branch:
            self.generate(statement)
        self.code.append(f"goto {end_label}")
        
        # False/elif branches
        self.code.append(f"{false_label}:")
        for elif_cond, elif_body in node.elif_branches:
            elif_condition = self.generate(elif_cond)
            elif_false_label = self.new_label()
            
            self.code.append(f"if not {elif_condition} goto {elif_false_label}")
            for statement in elif_body:
                self.generate(statement)
            self.code.append(f"goto {end_label}")
            
            self.code.append(f"{elif_false_label}:")
        
        # Else branch
        if node.false_branch:
            for statement in node.false_branch:
                self.generate(statement)
        
        self.code.append(f"{end_label}:")
    
    def generate_while_loop(self, node):
        start_label = self.new_label()
        condition_label = self.new_label()
        end_label = self.new_label()
        
        self.code.append(f"goto {condition_label}")
        self.code.append(f"{start_label}:")
        
        # Loop body
        for statement in node.body:
            self.generate(statement)
        
        # Condition check
        self.code.append(f"{condition_label}:")
        condition = self.generate(node.condition)
        self.code.append(f"if {condition} goto {start_label}")
        
        self.code.append(f"{end_label}:")
    
    def generate_for_loop(self, node):
        iterable = self.generate(node.iterable)
        index_var = self.new_temp()
        end_label = self.new_label()
        
        # Initialize index
        self.code.append(f"{index_var} = 0")
        
        # Loop condition
        loop_start = self.new_label()
        self.code.append(f"{loop_start}:")
        self.code.append(f"if {index_var} >= len({iterable}) goto {end_label}")
        
        # Get current element
        element = self.new_temp()
        self.code.append(f"{element} = {iterable}[{index_var}]")
        self.code.append(f"{node.loop_var} = {element}")
        
        # Loop body
        for statement in node.body:
            self.generate(statement)
            
        # Increment and loop back
        self.code.append(f"{index_var} = {index_var} + 1")
        self.code.append(f"goto {loop_start}")
        
        # End label
        self.code.append(f"{end_label}:")
    
    def generate_function_declaration(self, node):
        # Save current function context
        old_function = self.current_function
        self.current_function = node.name
        
        # Create function entry
        self.code.append(f"func {node.name} begin")
        
        # Parameters
        for param_type, param_name, is_array in node.params:
            if is_array:
                self.code.append(f"param {param_name}[]")
            else:
                self.code.append(f"param {param_name}")
        
        # Function body
        for statement in node.body:
            self.generate(statement)
        
        # End function
        self.code.append(f"end {node.name}")
        
        # Restore previous function context
        self.current_function = old_function
    
    def generate_function_call(self, node):
        # Evaluate arguments
        args = [self.generate(arg) for arg in node.arguments]
        
        # If the function returns a value, create a temp for it
        if self.current_function != node.name:  # Avoid recursion issues
            temp = self.new_temp()
            self.code.append(f"{temp} = call {node.name}({', '.join(args)})")
            return temp
        else:
            self.code.append(f"call {node.name}({', '.join(args)})")
            return None
    
    def generate_return_statement(self, node):
        if node.expression:
            value = self.generate(node.expression)
            self.code.append(f"return {value}")
        else:
            self.code.append("return")
    
    def generate_array_literal(self, node):
        temp = self.new_temp()
        elements = [self.generate(elem) for elem in node.elements]
        self.code.append(f"{temp} = array[{', '.join(elements)}]")
        return temp
    
    def generate_array_access(self, node):
        if isinstance(node.array, str):
            array = node.array  # Directly use the string as array name
        else:
            array = self.generate(node.array)
        
        index = self.generate(node.index)
        temp = self.new_temp()
        self.code.append(f"{temp} = {array}[{index}]")
        return temp
    
    def generate_method_call(self, node):
        obj = self.generate(Expression(node.object_name))
        method = node.method_name
        args = [self.generate(arg) for arg in node.arguments]
        
        temp = self.new_temp()
        if method == 'add':
            if len(args) == 1:
                self.code.append(f"{obj}.add({args[0]})")
            else:
                self.code.append(f"{obj}.add({args[0]}, {args[1]})")
        elif method == 'remove':
            if len(args) == 0:
                self.code.append(f"{temp} = {obj}.remove()")
            else:
                self.code.append(f"{temp} = {obj}.remove({args[0]})")
        else:
            raise ValueError(f"Unknown method {method}")
        
        return temp

    def get_code(self):
        return self.code

    def print_code(self):
        for line in self.code:
            print(line)

code = """
int x = 5 + 3
"""
parser = Parser(tokenize(code))
ast = parser.parse()

codegen = IntermediateCodeGenerator()
codegen.generate(ast)

# Print the generated code
print("Generated Intermediate Code:")
codegen.print_code()
