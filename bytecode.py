from example import (
    Program,
    VariableDeclaration,
    PrintStatement,
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
    RepeatLoop,
    MethodCall,
    AssignmentStatement,  # Ensure this is imported
    Expression)
from example import Parser, tokenize
class BytecodeGenerator:
    def __init__(self):
        self.bytecode = []
        self.variables = set()
        self.functions = {}
        self.label_counter = 0
        self.loop_stack = []
    
    def generate(self, node):
        if isinstance(node, Program):
            for statement in node.statements:
                self.generate(statement)
        elif isinstance(node, VariableDeclaration):
            self.generate_variable_declaration(node)
        elif isinstance(node, AssignmentStatement):  # Ensure this check exists
            self.generate_assignment(node)
        elif isinstance(node, PrintStatement):
            self.generate_print_statement(node)
        elif isinstance(node, BinaryOperation):
            self.generate_binary_operation(node)
        elif isinstance(node, Expression):
            self.generate_expression(node)
        elif isinstance(node, IfElse):
            self.generate_if_else(node)
        elif isinstance(node, WhileLoop):
            self.generate_while_loop(node)
        elif isinstance(node, ForLoop):
            self.generate_for_loop(node)
        elif isinstance(node, FunctionDeclaration):
            self.generate_function_declaration(node)
        elif isinstance(node, FunctionCall):
            self.generate_function_call(node)
        elif isinstance(node, ReturnStatement):
            self.generate_return_statement(node)
        elif isinstance(node, ArrayLiteral):
            self.generate_array_literal(node)
        elif isinstance(node, ArrayAccess):
            self.generate_array_access(node)
        elif isinstance(node, RepeatLoop):
            self.generate_repeat_loop(node)
        elif isinstance(node, MethodCall):  # Add this case
            self.generate_method_call(node)
        else:
            raise ValueError(f"Unknown node type: {type(node)}")
        return self.bytecode
    
    def generate_variable_declaration(self, node):
        # Handle variable type and const/var
        if node.var_type:
            if node.var_type.startswith('const_'):
                var_type = node.var_type.split('_')[1]
                is_const = True
            elif node.var_type.startswith('var_'): # Handle explicit 'var'
                 var_type = node.var_type.split('_')[1]
                 is_const = False
            else:
                var_type = node.var_type
                is_const = False
        else:
            # Should not happen if parser enforces type declaration
            raise SyntaxError(f"Variable '{node.name}' must have a type (int, float, string).")

        # Handle array declaration part (ARRAY_NEW)
        if node.is_array:
            self.bytecode.append(('ARRAY_NEW', var_type, node.name))
            
            # Handle initialization
            if node.value is not None:
                if isinstance(node.value, ArrayLiteral):
                    # Handle array literal initialization
                    for element_expr in reversed(node.value.elements):
                        self.generate(element_expr)
                    self.bytecode.append(('ARRAY_INIT', node.name, len(node.value.elements)))
                else:
                    # Handle initialization from function calls or other expressions
                    self.generate(node.value)  # Generate code for the value expression
                    self.bytecode.append(('ASSIGN', node.name))  # Assign the result to array
                    
                    # If the value is a function call that returns an array,
                    # the VM will handle the array assignment automatically
        else:
            # Non-array variable handling
            if node.value is not None:
                self.generate(node.value)
                if node.name in self.variables and not is_const:
                    self.bytecode.append(('ASSIGN', node.name))
                elif is_const:
                    self.bytecode.append(('DECLARE_CONST', var_type, node.name))
                    self.bytecode.append(('ASSIGN', node.name))
                else:
                    self.bytecode.append(('DECLARE_VAR', var_type, node.name))
                    self.bytecode.append(('ASSIGN', node.name))
            else:
                self.bytecode.append(('DECLARE_VAR', var_type, node.name))

        # Add variable name to the set
        self.variables.add(node.name)

    def generate_assignment(self, node):
        # Generate code for the value expression (e.g., a + 2)
        self.generate(node.value)
        # Assign the result to the variable
        self.bytecode.append(('ASSIGN', node.variable))
    
    def generate_print_statement(self, node):
        # Let generate_expression handle LOAD vs PUSH logic
        self.generate(node.expression)
        self.bytecode.append(('CALL', 'printk', 1))
    
    def generate_binary_operation(self, node):
        # Special case for assignment operation (a = b)
        if node.operator == '=':
            if isinstance(node.left, Expression):
                # Check if right side is another binary operation (like a + 2)
                if isinstance(node.right, BinaryOperation):
                    # Handle compound operation (a = a + 2)
                    if isinstance(node.right.left, Expression) and node.right.left.value == node.left.value:
                        # Load current variable value
                        self.bytecode.append(('LOAD', node.left.value))
                        
                        # Generate right operand
                        self.generate(node.right.right)
                        
                        # Apply operation
                        op_map = {
                            '+': 'ADD',
                            '-': 'SUB',
                            '*': 'MUL',
                            '/': 'DIV',
                            '%': 'MOD'
                        }
                        self.bytecode.append((op_map[node.right.operator],))
                        
                        # Assign back to variable
                        self.bytecode.append(('ASSIGN', node.left.value))
                        return
                
                # Normal assignment
                self.generate(node.right)
                self.bytecode.append(('ASSIGN', node.left.value))
                return
        
        # Normal binary operations
        self.generate(node.left)
        self.generate(node.right)
        
        # Map operators to bytecode operations
        op_map = {
            '+': 'ADD',
            '-': 'SUB',
            '*': 'MUL',
            '/': 'DIV',
            '%': 'MOD',
            '==': 'EQ',
            '!=': 'NEQ',
            '<': 'LT',
            '>': 'GT',
            '<=': 'LTE',
            '>=': 'GTE',
            'and': 'AND',
            'or': 'OR'
        }
        self.bytecode.append((op_map[node.operator],))
    
    def generate_expression(self, node):
        # Use the is_identifier flag to decide between LOAD and PUSH
        if node.is_identifier:
            # It's an identifier, generate LOAD
            if node.value in self.variables: # Check if known during generation
                self.bytecode.append(('LOAD', node.value))
            else:
                # This should ideally be caught by the parser earlier
                raise NameError(f"Bytecode Generation: Undeclared variable referenced: {node.value}")
        # Check for literal types (int, float, string)
        elif isinstance(node.value, (int, float, str)):
            # It's a literal value, generate PUSH
            self.bytecode.append(('PUSH', node.value))
        else:
            # Handle other cases or raise error
            raise TypeError(f"Cannot generate bytecode for Expression with value: {node.value} (type: {type(node.value)})")
    
    def generate_if_else(self, node):
        # Generate condition
        self.generate(node.condition)

        # Create labels
        else_label = f"L{self.label_counter}" # Label for start of next block (elif/else/end)
        end_label = f"L{self.label_counter + 1}" # Label for the very end of the if/elif/else structure
        self.label_counter += 2

        # Jump if false to the start of the next block
        self.bytecode.append(('JUMP_IF_FALSE', else_label))

        # Generate true branch
        for statement in node.true_branch:
            self.generate(statement)

        # Check if a jump to the end is needed after the true branch.
        # It's needed if there are subsequent elif/else blocks, OR
        # if the true branch doesn't end in a RETURN (to avoid falling through).
        needs_jump_after_true = (node.elif_branches or node.false_branch) or \
                                (node.true_branch and (not self.bytecode or not self.bytecode[-1][0].startswith('RETURN')))
        if needs_jump_after_true:
             self.bytecode.append(('JUMP', end_label))

        # --- Handle Elif/Else ---
        current_else_label = else_label # Start with the label for the first potential next block

        # Generate elif branches
        for i, (elif_condition, elif_body) in enumerate(node.elif_branches):
            # Define the label for this elif block
            self.bytecode.append((current_else_label,))

            # Create the label for the *next* block (next elif or else or end)
            next_else_label = f"L{self.label_counter}"
            self.label_counter += 1

            # Generate condition and jump
            self.generate(elif_condition)
            self.bytecode.append(('JUMP_IF_FALSE', next_else_label))

            # Generate body
            for statement in elif_body:
                self.generate(statement)

            # Jump to the end after the body executes, unless it ends with RETURN
            needs_jump_after_elif = (not self.bytecode or not self.bytecode[-1][0].startswith('RETURN'))
            if needs_jump_after_elif:
                 self.bytecode.append(('JUMP', end_label))

            # Update the current_else_label for the next iteration or the final else/end
            current_else_label = next_else_label

        # Generate else branch (if it exists)
        if node.false_branch:
            # Define the label for the else block (using the last generated else_label)
            self.bytecode.append((current_else_label,))
            for statement in node.false_branch:
                self.generate(statement)
            # No JUMP needed here, execution flows naturally to end_label
        else:
            # If there was no else branch, the last current_else_label
            # still needs to be defined as the target for the initial JUMP_IF_FALSE
            # (or the last JUMP_IF_FALSE from an elif).
             self.bytecode.append((current_else_label,))


        # Define the final end label. All explicit JUMPs target this.
        # If there were no elif/else, execution flows here naturally after the else_label definition.
        self.bytecode.append((end_label,))

    def generate_while_loop(self, node):
        start_label = f"L{self.label_counter}"
        end_label = f"L{self.label_counter + 1}"
        self.label_counter += 2
        self.loop_stack.append((start_label, end_label))
        
        # Loop start label
        self.bytecode.append((start_label,))
        
        # Check condition
        self.generate(node.condition)
        self.bytecode.append(('JUMP_IF_FALSE', end_label))
        
        # Loop body
        for statement in node.body:
            # Check for assignment statements in the loop body that might update loop variables
            if isinstance(statement, BinaryOperation) and statement.operator == '=':
                # This is an assignment operation inside the loop - handle carefully
                self.generate(statement)
            else:
                self.generate(statement)
        
        # Jump back to loop start
        self.bytecode.append(('JUMP', start_label))
        
        # End label
        self.bytecode.append((end_label,))
        self.loop_stack.pop()
    
    def generate_for_loop(self, node):
        # Initialize loop variable
        self.bytecode.append(('DECLARE_VAR', 'int', node.loop_var))
        
        # Get the iterable
        if isinstance(node.iterable, ArrayLiteral):
            temp_array_name = f"_temp_array_{self.label_counter}"
            self.label_counter += 1
            self.bytecode.append(('ARRAY_NEW', 'int', temp_array_name)) # Create empty array structure

            # Push elements onto the stack in reverse order for ARRAY_INIT
            for element_expr in reversed(node.iterable.elements):
                self.generate(element_expr)

            # Call ARRAY_INIT with the count of elements
            self.bytecode.append(('ARRAY_INIT', temp_array_name, len(node.iterable.elements)))
            iterable = temp_array_name
        else:
            # Assuming node.iterable is an Expression containing the variable name
            if isinstance(node.iterable, Expression):
                 iterable = node.iterable.value
            else:
                 # Handle other potential iterable types if necessary
                 raise TypeError(f"Unsupported iterable type in for loop: {type(node.iterable)}")
        
        # For loop setup
        start_label = f"L{self.label_counter}"
        end_label = f"L{self.label_counter + 1}"
        self.label_counter += 2
        self.loop_stack.append((start_label, end_label))
        
        # Initialize counter
        self.bytecode.append(('PUSH', 0))
        self.bytecode.append(('DECLARE_VAR', 'int', '_for_index'))
        self.bytecode.append(('ASSIGN', '_for_index'))
        
        # Loop start
        self.bytecode.append((start_label,))
        
        # Check if we've reached the end
        self.bytecode.append(('LOAD', '_for_index'))
        self.bytecode.append(('LOAD', iterable))
        self.bytecode.append(('CALL', 'length', 1))
        self.bytecode.append(('LT',))
        self.bytecode.append(('JUMP_IF_FALSE', end_label))
        
        # Get current element
        self.bytecode.append(('LOAD', iterable))
        self.bytecode.append(('LOAD', '_for_index'))
        self.bytecode.append(('ARRAY_LOAD',))
        self.bytecode.append(('ASSIGN', node.loop_var))
        
        # Execute loop body
        for statement in node.body:
            self.generate(statement)
        
        # Increment counter
        self.bytecode.append(('LOAD', '_for_index'))
        self.bytecode.append(('PUSH', 1))
        self.bytecode.append(('ADD',))
        self.bytecode.append(('ASSIGN', '_for_index'))
        
        # Jump back to start
        self.bytecode.append(('JUMP', start_label))
        self.bytecode.append((end_label,))
        self.loop_stack.pop()
    
    def generate_function_declaration(self, node):
        func_name = node.name
        # Ensure function isn't executed inline by adding a jump around it
        skip_label = f"SKIP_FUNC_{func_name}_{self.label_counter}"
        self.label_counter += 1
        self.bytecode.append(('JUMP', skip_label)) # Jump over the function body

        # Record the actual start address of the function's code (after the JUMP)
        func_start_address = len(self.bytecode)

        # Store function metadata (address, params) for the CALL instruction
        self.functions[func_name] = {
            'address': func_start_address,
            'params': [(pt, pn, ia) for pt, pn, ia in node.params],
            'return_type': 'int' # Simplified, ideally infer this
        }

        # --- Generate Function Body ---
        # The VM's CALL instruction will handle setting up the scope
        original_vars = self.variables.copy() # Keep track of global vars
        # Add params to local scope for generation if needed (helps generate_expression)
        for _, param_name, _ in node.params:
             self.variables.add(param_name)

        for statement in node.body:
            self.generate(statement)

        # Restore original variable context after generating body
        self.variables = original_vars
        # --- End of Function Body ---

        # Ensure an implicit return if the function doesn't end with one
        # Check if the last generated instruction was a RETURN
        if not self.bytecode or not self.bytecode[-1][0].startswith('RETURN'):
             self.bytecode.append(('PUSH', None)) # Default return value is None
             self.bytecode.append(('RETURN', True)) # Indicate a value (None) is on stack

        # Define the label to jump to, placed *after* the function body and implicit return
        self.bytecode.append((skip_label,))

    def generate_function_call(self, node):
        # Push arguments in reverse order
        for arg in reversed(node.arguments):
            self.generate(arg)
        
        self.bytecode.append(('CALL', node.name, len(node.arguments)))
    
    def generate_return_statement(self, node):
        if node.expression:
            self.generate(node.expression)
            # Indicate that RETURN should expect a value pushed by the expression
            self.bytecode.append(('RETURN', True))
        else:
            self.bytecode.append(('PUSH', None)) # Push None if no expression
            # Indicate that RETURN should expect a value (None)
            self.bytecode.append(('RETURN', True))
    
    def generate_array_literal(self, node):
        # For array literals, we'll create a temporary array
        temp_name = f"_temp_array_{self.label_counter}"
        self.label_counter += 1
        
        self.bytecode.append(('ARRAY_NEW', 'int', temp_name))
        for element in reversed(node.elements):
            self.generate(element)
        self.bytecode.append(('ARRAY_INIT', temp_name, len(node.elements)))
        
        return temp_name
    
    def generate_array_access(self, node):
        # Handle case where node.array is a string (variable name)
        if isinstance(node.array, str):
            self.bytecode.append(('LOAD', node.array))
        else:
            self.generate(node.array)
        self.generate(node.index)
        self.bytecode.append(('ARRAY_LOAD',))
    
    def generate_repeat_loop(self, node):
        self.generate(node.count)
        temp_var = f"_repeat_count_{self.label_counter}"
        self.label_counter += 1
        
        # Store count in temporary variable
        self.bytecode.append(('DECLARE_VAR', 'int', temp_var))
        self.bytecode.append(('ASSIGN', temp_var))
        
        # Loop setup
        start_label = f"L{self.label_counter}"
        end_label = f"L{self.label_counter + 1}"
        self.label_counter += 2
        self.loop_stack.append((start_label, end_label))
        
        self.bytecode.append((start_label,))
        
        # Check if done
        self.bytecode.append(('LOAD', temp_var))
        self.bytecode.append(('PUSH', 0))
        self.bytecode.append(('GT',))
        self.bytecode.append(('JUMP_IF_FALSE', end_label))
        
        # Decrement counter
        self.bytecode.append(('LOAD', temp_var))
        self.bytecode.append(('PUSH', 1))
        self.bytecode.append(('SUB',))
        self.bytecode.append(('ASSIGN', temp_var))
        
        # Execute body
        for statement in node.body:
            self.generate(statement)
        
        # Jump back
        self.bytecode.append(('JUMP', start_label))
        self.bytecode.append((end_label,))
        self.loop_stack.pop()
    
    def generate_method_call(self, node):
        # Load the object on which the method is being called
        self.bytecode.append(('LOAD', node.object_name))
        
        # Push arguments in reverse order (common calling convention)
        for arg in reversed(node.arguments):
            self.generate(arg)
        
        # Call the method with the object and arguments
        self.bytecode.append(('CALL_METHOD', node.method_name, len(node.arguments)))
        
        return None  # Method calls might return values but we don't capture here

def compile_code(source_code):
    # Tokenize and parse the source code
    from example import Parser, tokenize
    tokens = list(tokenize(source_code))
    parser = Parser(tokens)
    ast = parser.parse()

    # Generate bytecode
    generator = BytecodeGenerator()
    bytecode = generator.generate(ast)

    # Return both bytecode and the collected function definitions
    return bytecode, generator.functions

