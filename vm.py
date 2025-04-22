class VirtualMachine:
    # FIX: Update __init__ to accept functions
    def __init__(self, functions=None):
        # Execution state
        self.stack = []
        self.variables = {}
        self.variables_meta = {}
        self.constants = {}
        # Store the passed function definitions, default to empty dict if None
        self.functions = functions or {}
        self.call_stack = []
        self.pc = 0
        self.instructions = []
        self.output = []

        # Arrays storage
        self.arrays = {}

        # Built-in functions
        self.builtins = {
            'printk': self._builtin_printk,
            'length': self._builtin_length,
            'remove': self._builtin_remove,
            'add': self._builtin_add,
            'array_init': self._builtin_array_init
        }

    def load(self, bytecode):
        """Load bytecode instructions into the VM"""
        self.instructions = bytecode

        # Build a mapping of labels to instruction positions
        self.labels = {}
        for i, instruction in enumerate(bytecode):
            if isinstance(instruction, tuple) and len(instruction) == 1 and isinstance(instruction[0], str):
                label_name = instruction[0]
                # <<< FIX: More specific check for labels >>>
                # Check for L<digits> or SKIP_... patterns
                is_L_label = label_name.startswith('L') and label_name[1:].isdigit()
                is_skip_label = label_name.startswith('SKIP_')

                if is_L_label or is_skip_label:
                    if label_name in self.labels:
                         # Optional: Warn about duplicate labels if desired
                         print(f"Warning: Duplicate label '{label_name}' defined at index {i}. Previous definition at {self.labels[label_name]}.")
                    self.labels[label_name] = i
                # <<< END FIX >>>
            # Instructions like ('LT',), ('LTE',), ('ADD',) are not labels and should not be registered here.

    def run(self, max_steps=1000000):  # Increased max_steps to 1,000,000
        """Execute the loaded program and return output"""
        self.pc = 0
        steps = 0
        while self.pc < len(self.instructions) and steps < max_steps:
            steps += 1
            instruction = self.instructions[self.pc]
            try:
                self.execute(instruction)
                self.pc += 1
                
                # Add periodic status updates for long-running programs
                if steps % 100000 == 0:  # Print status every 100k steps
                    print(f"Executed {steps} instructions...")
                    
            except Exception as e:
                self.output.append(f"Error at instruction {self.pc}: {instruction} - {str(e)}")
                break
        
        if steps >= max_steps:
            self.output.append(f"WARNING: Execution exceeded {max_steps} steps - possible infinite loop")
            return '\n'.join(self.output)
        
        return '\n'.join(self.output)

    def execute(self, instruction):
        op = instruction[0]

        try:
            # --- Primary Instruction Handling ---
            if op == 'PUSH':
                self.stack.append(instruction[1])
            elif op == 'LOAD':
                var_name = instruction[1]
                if (var_name in self.variables):
                    self.stack.append(self.variables[var_name])
                elif (var_name in self.constants):
                    self.stack.append(self.constants[var_name])
                elif (var_name in self.arrays):
                    self.stack.append(var_name) # Push array name as reference
                else:
                    raise RuntimeError(f"Undefined variable or array: {var_name}")
            elif op == 'ASSIGN':
                var_name = instruction[1]
                if not self.stack:
                    raise RuntimeError(f"ASSIGN Error: Stack is empty when assigning to '{var_name}'")
                value = self.stack.pop()
                
                # Handle array assignment
                if var_name in self.arrays:
                    if isinstance(value, list):
                        # Direct array assignment
                        self.arrays[var_name]['values'] = value
                    elif isinstance(value, str) and value in self.arrays:
                        # Copy from another array
                        self.arrays[var_name]['values'] = self.arrays[value]['values'].copy()
                    else:
                        raise TypeError(f"Cannot assign {type(value)} to array '{var_name}'")
                else:
                    # Regular variable assignment
                    target_type = self.variables_meta.get(f"_type{var_name}")
                    if target_type:
                        try:
                            if target_type == 'int': value = int(value)
                            elif target_type == 'float': value = float(value)
                            elif target_type == 'string': value = str(value)
                        except (ValueError, TypeError) as e:
                            raise RuntimeError(f"Type Error: Cannot convert {value} to {target_type}")
                    self.variables[var_name] = value
            elif op == 'DECLARE_VAR':
                _, var_type, var_name = instruction
                if var_type == 'int': self.variables[var_name] = 0
                elif var_type == 'float': self.variables[var_name] = 0.0
                elif var_type == 'string': self.variables[var_name] = ""
                else: self.variables[var_name] = None
                self.variables_meta[f"_type{var_name}"] = var_type

            # --- Other Instructions ---
            elif op == 'DECLARE_CONST':
                _, const_type, const_name, const_value = instruction
                self.constants[const_name] = const_value
            elif op == 'POP':
                self.stack.pop()
            # Arithmetic
            elif op == 'ADD':
                b, a = self.stack.pop(), self.stack.pop()
                if isinstance(a, (int, float)) and isinstance(b, (int, float)): self.stack.append(a + b)
                elif isinstance(a, str): self.stack.append(a + str(b))
                else: self.stack.append(a + b) # Fallback or error?
            elif op == 'SUB': b, a = self.stack.pop(), self.stack.pop(); self.stack.append(a - b)
            elif op == 'MUL': b, a = self.stack.pop(), self.stack.pop(); self.stack.append(a * b)
            elif op == 'DIV': b, a = self.stack.pop(), self.stack.pop(); self.stack.append(a / b) # Consider integer division?
            elif op == 'MOD': b, a = self.stack.pop(), self.stack.pop(); self.stack.append(a % b)
            # Comparison
            elif op == 'EQ':  b, a = self.stack.pop(), self.stack.pop(); self.stack.append(a == b)
            elif op == 'NEQ': b, a = self.stack.pop(), self.stack.pop(); self.stack.append(a != b)
            elif op == 'GT':  b, a = self.stack.pop(), self.stack.pop(); self.stack.append(a > b)
            elif op == 'LT':  b, a = self.stack.pop(), self.stack.pop(); self.stack.append(a < b)
            elif op == 'GTE': b, a = self.stack.pop(), self.stack.pop(); self.stack.append(a >= b)
            elif op == 'LTE': b, a = self.stack.pop(), self.stack.pop(); self.stack.append(a <= b)
            # Control Flow
            elif op == 'JUMP':
                target = instruction[1]
                if isinstance(target, str):
                    if target in self.labels: self.pc = self.labels[target] - 1
                    else: raise RuntimeError(f"JUMP Error: Undefined label '{target}'. Known labels: {list(self.labels.keys())}")
                elif isinstance(target, int): self.pc = target - 1
                else: raise RuntimeError(f"JUMP Error: Invalid jump target type '{type(target)}' for target '{target}'")
            elif op == 'JUMP_IF_FALSE':
                condition = self.stack.pop()
                if not condition:
                    target = instruction[1]
                    if isinstance(target, str):
                        if target in self.labels: self.pc = self.labels[target] - 1
                        else: raise RuntimeError(f"JUMP_IF_FALSE Error: Undefined label '{target}'. Known labels: {list(self.labels.keys())}")
                    elif isinstance(target, int): self.pc = target - 1
                    else: raise RuntimeError(f"JUMP_IF_FALSE Error: Invalid jump target type '{type(target)}' for target '{target}'")
            # Function Calls
            elif op == 'CALL':
                func_name, arg_count = instruction[1], instruction[2]
                args = [self.stack.pop() for _ in range(arg_count)][::-1] # Pop args, keep reverse for now
                if func_name in self.builtins:
                    result = self.builtins[func_name](*args)
                    if result is not None: self.stack.append(result)
                elif func_name in self.functions:
                    self.call_stack.append({'pc': self.pc, 'variables': self.variables.copy(), 'stack': self.stack.copy()})
                    func_info = self.functions[func_name]
                    self.variables = {} # New scope
                    # TODO: Check arg count matches param count
                    for i, (param_type, param_name, _) in enumerate(func_info['params']):
                        self.variables[param_name] = args[i] # Assign args to params in new scope
                        # TODO: Add type metadata for params? self.variables_meta[f"_type{param_name}"] = param_type
                    self.pc = func_info['address'] - 1 # Jump to function start
                else:
                    raise RuntimeError(f"Undefined function: {func_name}")
            elif op == 'RETURN':
                # Check if a value was explicitly pushed by the return statement
                # instruction[1] should be True if there was a return expression, False/None otherwise
                has_return_value = len(instruction) > 1 and instruction[1]

                # Pop the return value if one exists, otherwise default to None
                return_value = self.stack.pop()

                if not self.call_stack:
                    self.output.append("Warning: RETURN outside function call or call stack empty.")
                    self.pc = len(self.instructions) # Halt
                    return # Stop processing this instruction

                # Restore state from call stack
                saved_state = self.call_stack.pop()
                self.pc = saved_state['pc']
                # CRITICAL FIX: Restore variables *before* restoring stack
                self.variables = saved_state['variables']
                self.stack = saved_state['stack']

                # Push the return value onto the *restored* caller's stack
                self.stack.append(return_value)

                # The main loop will increment PC, so we are ready for the instruction after CALL
            # Arrays
            elif op == 'ARRAY_NEW':
                _, array_type, array_name = instruction
                self.arrays[array_name] = {'type': array_type, 'values': []}
            elif op == 'ARRAY_INIT':
                _, array_name, element_count = instruction
                if array_name not in self.arrays: raise RuntimeError(f"ARRAY_INIT error: Array '{array_name}' not created.")
                if not isinstance(element_count, int) or element_count < 0: raise RuntimeError(f"ARRAY_INIT error: Invalid element count '{element_count}'.")
                if len(self.stack) < element_count: raise RuntimeError(f"ARRAY_INIT error: Stack underflow. Expected {element_count}, found {len(self.stack)}.")
                elements = [self.stack.pop() for _ in range(element_count)]
                self.arrays[array_name]['values'] = elements # Popped in reverse, store as is.
            elif op == 'ARRAY_LOAD':
                index = self.stack.pop()
                array_ref = self.stack.pop()
                if isinstance(array_ref, str) and array_ref in self.arrays:
                    actual_array = self.arrays[array_ref]['values']
                    if not isinstance(index, int): raise TypeError(f"Array index must be an integer, got {type(index)}")
                    if index < 0 or index >= len(actual_array): raise IndexError(f"Array index {index} out of bounds for array '{array_ref}' length {len(actual_array)}")
                    self.stack.append(actual_array[index])
                elif isinstance(array_ref, list): # Literal list
                    if not isinstance(index, int): raise TypeError(f"List index must be an integer, got {type(index)}")
                    if index < 0 or index >= len(array_ref): raise IndexError(f"List index {index} out of bounds for list length {len(array_ref)}")
                    self.stack.append(array_ref[index])
                elif isinstance(array_ref, str): # String indexing
                    if not isinstance(index, int): raise TypeError(f"String index must be an integer, got {type(index)}")
                    if index < 0 or index >= len(array_ref): raise IndexError(f"String index {index} out of bounds for string length {len(array_ref)}")
                    self.stack.append(array_ref[index])
                else:
                    raise RuntimeError(f"Cannot index object of type {type(array_ref)}: {array_ref}")
            elif op == 'ARRAY_STORE':
                 # Assuming ('ARRAY_STORE', array_name, index_value) - Needs adjustment if index is on stack
                 # Let's assume index and value are on stack: value, index, array_name
                 value = self.stack.pop()
                 index = self.stack.pop()
                 array_name = self.stack.pop() # Assuming array name is pushed before index/value
                 if not isinstance(array_name, str) or array_name not in self.arrays:
                     raise RuntimeError(f"ARRAY_STORE error: Invalid array reference '{array_name}'")
                 if not isinstance(index, int):
                     raise TypeError(f"ARRAY_STORE error: Index must be integer, got {type(index)}")
                 actual_array = self.arrays[array_name]['values']
                 if index < 0 or index >= len(actual_array):
                     raise IndexError(f"ARRAY_STORE error: Index {index} out of bounds for array '{array_name}' length {len(actual_array)}")
                 # TODO: Type check value against self.arrays[array_name]['type']?
                 actual_array[index] = value
            # Function Definition (Metadata loading - should ideally happen during load, not execute)
            elif op == 'FUNCTION_DEF':
                 # This instruction might be redundant if functions are passed during VM init
                 _, name, params, return_type, address = instruction
                 if name not in self.functions: # Avoid overwriting if passed via init
                     self.functions[name] = {'params': params, 'return_type': return_type, 'address': address}
            # Method Calls
            elif op == 'CALL_METHOD':
                method_name, arg_count = instruction[1], instruction[2]
                args = [self.stack.pop() for _ in range(arg_count)] # Pop args, NO reverse needed
                if not self.stack: raise RuntimeError(f"CALL_METHOD Error: Stack empty, expected object reference for '{method_name}'")
                object_ref = self.stack.pop() # Pop object reference (e.g., array name)

                method_handler = None
                if method_name == 'remove': method_handler = self._builtin_remove
                elif method_name == 'add': method_handler = self._builtin_add
                elif method_name == 'length':
                    method_handler = self._builtin_length
                    args.insert(0, object_ref) # Prepend object for length handler
                # Add other methods

                if method_handler:
                    if method_name != 'length': result = method_handler(object_ref, *args) # Pass object_ref first
                    else: result = method_handler(*args) # Length handler expects object in args
                    if result is not None: self.stack.append(result)
                else:
                    raise RuntimeError(f"Unsupported method call: {object_ref}.{method_name}")

            # Label Definition (Ignore during execution)
            elif isinstance(instruction, tuple) and len(instruction) == 1 and isinstance(op, str):
                pass # Labels are targets, not executable operations

            # Unknown Instruction
            else:
                raise RuntimeError(f"Unknown instruction: {op}")

        except Exception as e:
            # Add more context to the error
            raise RuntimeError(f"Error executing instruction {self.pc} ({instruction}): {str(e)}")

    # Built-in functions
    def _builtin_printk(self, *args):
        """Built-in print function - Handles arrays"""
        output_parts = []
        for arg in args:
            # <<< FIX: Check if arg is an array name >>>
            if isinstance(arg, str) and arg in self.arrays:
                # It's an array name, print its contents using repr for list format
                output_parts.append(repr(self.arrays[arg]['values']))
            else:
                # Otherwise, just convert the argument to string
                output_parts.append(str(arg))
        
        output = ' '.join(output_parts)
        self.output.append(output)

    def _builtin_length(self, target):
        """Get length of an array (by name) or a string."""
        # Check if target is the name of an array stored in self.arrays
        if isinstance(target, str) and target in self.arrays:
            return len(self.arrays[target]['values'])
        # Check if target is a string literal itself
        elif isinstance(target, str):
            return len(target)
        # Check if target is a list literal (less common but possible)
        elif isinstance(target, list):
             return len(target)
        else:
            # Raise a more specific error if the type is unsupported
            raise TypeError(f"Builtin 'length': Cannot get length of type {type(target).__name__}")

    def _builtin_remove(self, arr_name, index_arg):
        """Remove element from array"""
        if arr_name not in self.arrays:
             raise RuntimeError(f"Builtin 'remove': Array '{arr_name}' not found.")

        # <<< FIX: Explicitly convert index to integer >>>
        try:
            index = int(index_arg)
        except (ValueError, TypeError):
            raise TypeError(f"Builtin 'remove': Index argument must be an integer, got {type(index_arg)} ({index_arg})")

        # Check bounds for pop
        if index < 0 or index >= len(self.arrays[arr_name]['values']):
             raise IndexError(f"Builtin 'remove': Index {index} out of bounds for array '{arr_name}' of length {len(self.arrays[arr_name]['values'])}")

        self.arrays[arr_name]['values'].pop(index)
    
    def _builtin_add(self, arr_name, *args):
        """Add element to array"""
        if arr_name not in self.arrays:
             raise RuntimeError(f"Builtin 'add': Array '{arr_name}' not found.")

        if len(args) == 1:
            # Add to end
            value = args[0]
            # Optional: Type check value against array type?
            self.arrays[arr_name]['values'].append(value)
        elif len(args) == 2:
            # Insert at index
            index_arg, value = args

            # <<< FIX: Explicitly convert index to integer >>>
            try:
                index = int(index_arg)
            except (ValueError, TypeError):
                raise TypeError(f"Builtin 'add': Index argument must be an integer, got {type(index_arg)} ({index_arg})")

            # Optional: Type check value against array type?

            # Check bounds for insert (allow index == len to append)
            if index < 0 or index > len(self.arrays[arr_name]['values']):
                 raise IndexError(f"Builtin 'add': Index {index} out of bounds for array '{arr_name}' of length {len(self.arrays[arr_name]['values'])}")

            self.arrays[arr_name]['values'].insert(index, value)
        else:
             raise ValueError(f"Builtin 'add' takes 1 or 2 arguments ({len(args)} given)")

    def _builtin_array_init(self, size):
        """Initialize array with given size"""
        return [None] * size
