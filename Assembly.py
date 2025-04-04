import re
class AssemblyGenerator:
    def __init__(self):
        self.asm_code = []
        self.label_counter = 0
        self.var_map = {}  # Maps variables to registers/memory
        self.registers = ['rax', 'rbx', 'rcx', 'rdx', 'rsi', 'rdi', 'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'r14', 'r15']
        self.free_registers = set(self.registers)
        self.stack_offset = 0
        self.current_function = None
        
    def new_label(self):
        label = f"L{self.label_counter}"
        self.label_counter += 1
        return label
        
    def get_register(self):
        if not self.free_registers:
            raise RuntimeError("No free registers available")
        return self.free_registers.pop()
        
    def free_register(self, reg):
        self.free_registers.add(reg)
        
    def generate(self, intermediate_code):
        # First pass: collect all variables
        self._collect_variables(intermediate_code)
        
        # Generate prologue
        self.asm_code.append("section .text")
        self.asm_code.append("global _start")
        self.asm_code.append("_start:")
        self.asm_code.append("    ; Setup stack frame")
        self.asm_code.append("    push rbp")
        self.asm_code.append("    mov rbp, rsp")
        
        # Reserve stack space for variables
        if self.stack_offset > 0:
            self.asm_code.append(f"    sub rsp, {self.stack_offset}")
        
        # Second pass: generate actual code
        for line in intermediate_code:
            self._generate_line(line)
        
        # Generate epilogue
        self.asm_code.append("    ; Exit program")
        self.asm_code.append("    mov rdi, 0")
        self.asm_code.append("    mov rax, 60")
        self.asm_code.append("    syscall")
        
        # Add data section for strings
        self.asm_code.append("\nsection .data")
        for var, info in self.var_map.items():
            if info['type'] == 'string':
                self.asm_code.append(f"{var} db '{info['value']}', 0")
        
        return self.asm_code
    
    def _collect_variables(self, intermediate_code):
        # Collect all variables and assign stack positions
        for line in intermediate_code:
            if '=' in line:
                left = line.split('=')[0].strip()
                if left not in self.var_map:
                    self._add_variable(left)
                    
                # Check right side for variables
                right = line.split('=')[1].strip()
                tokens = re.split(r'[+\-*/%<>=(),\[\]\s]+', right)
                for token in tokens:
                    if token and token[0].isalpha() and token not in self.var_map:
                        self._add_variable(token)
    
    def _add_variable(self, name):
        # Determine variable type (simplified - real implementation would track types)
        if name.startswith('t'):
            var_type = 'temp'
        elif name[0].isdigit() or (name[0] == '-' and name[1:].isdigit()):
            var_type = 'int'
        elif name.startswith('"'):
            var_type = 'string'
        else:
            var_type = 'var'
            
        self.var_map[name] = {
            'type': var_type,
            'stack_pos': self.stack_offset,
            'register': None
        }
        self.stack_offset += 8  # Allocate 8 bytes for each variable
    
    def _generate_line(self, line):
        line = line.strip()
        if not line:
            return
            
        # Handle labels
        if line.endswith(':'):
            self.asm_code.append(line)
            return
            
        # Handle function calls
        if line.startswith('call '):
            func_name = line[5:].split('(')[0]
            args = line.split('(')[1].rstrip(')').split(',')
            self._generate_function_call(func_name, args)
            return
            
        # Handle returns
        if line.startswith('return'):
            if len(line) > 6:
                value = line[7:].strip()
                self._generate_expression(value, 'rax')
            self.asm_code.append("    leave")
            self.asm_code.append("    ret")
            return
            
        # Handle print statements
        if line.startswith('print '):
            value = line[6:].strip()
            self._generate_print(value)
            return
            
        # Handle assignments
        if '=' in line:
            left, right = line.split('=', 1)
            left = left.strip()
            right = right.strip()
            self._generate_assignment(left, right)
            return
            
        # Handle control flow
        if line.startswith('if '):
            parts = line.split()
            condition = ' '.join(parts[1:-2])
            label = parts[-1]
            self._generate_condition(condition, label)
            return
            
        if line.startswith('goto '):
            label = line[5:].strip()
            self.asm_code.append(f"    jmp {label}")
            return
            
    def _generate_assignment(self, left, right):
        # Simple constant assignment
        if right.isdigit() or (right[0] == '-' and right[1:].isdigit()):
            self.asm_code.append(f"    mov QWORD [rbp - {self.var_map[left]['stack_pos']}], {right}")
            return
            
        # Variable to variable assignment
        if right in self.var_map:
            reg = self.get_register()
            self.asm_code.append(f"    mov {reg}, QWORD [rbp - {self.var_map[right]['stack_pos']}]")
            self.asm_code.append(f"    mov QWORD [rbp - {self.var_map[left]['stack_pos']}], {reg}")
            self.free_register(reg)
            return
            
        # Handle binary operations
        for op in ['+', '-', '*', '/', '%']:
            if op in right:
                left_op, right_op = right.split(op, 1)
                left_op = left_op.strip()
                right_op = right_op.strip()
                self._generate_binary_op(left, left_op, right_op, op)
                return
                
    def _generate_binary_op(self, dest, left, right, op):
        # Get registers for operands
        reg_left = self.get_register()
        reg_right = self.get_register()
        
        # Load left operand
        if left.isdigit():
            self.asm_code.append(f"    mov {reg_left}, {left}")
        else:
            self.asm_code.append(f"    mov {reg_left}, QWORD [rbp - {self.var_map[left]['stack_pos']}]")
            
        # Load right operand
        if right.isdigit():
            self.asm_code.append(f"    mov {reg_right}, {right}")
        else:
            self.asm_code.append(f"    mov {reg_right}, QWORD [rbp - {self.var_map[right]['stack_pos']}]")
            
        # Perform operation
        if op == '+':
            self.asm_code.append(f"    add {reg_left}, {reg_right}")
        elif op == '-':
            self.asm_code.append(f"    sub {reg_left}, {reg_right}")
        elif op == '*':
            self.asm_code.append(f"    imul {reg_left}, {reg_right}")
        elif op == '/':
            self.asm_code.append("    xor rdx, rdx")
            self.asm_code.append(f"    mov rax, {reg_left}")
            self.asm_code.append(f"    idiv {reg_right}")
            self.asm_code.append(f"    mov {reg_left}, rax")
        elif op == '%':
            self.asm_code.append("    xor rdx, rdx")
            self.asm_code.append(f"    mov rax, {reg_left}")
            self.asm_code.append(f"    idiv {reg_right}")
            self.asm_code.append(f"    mov {reg_left}, rdx")
            
        # Store result
        self.asm_code.append(f"    mov QWORD [rbp - {self.var_map[dest]['stack_pos']}], {reg_left}")
        
        # Free registers
        self.free_register(reg_left)
        self.free_register(reg_right)
        
    def _generate_condition(self, condition, label):
        # Simple comparison for now
        if '==' in condition:
            left, right = condition.split('==')
            left = left.strip()
            right = right.strip()
            
            reg_left = self.get_register()
            if left.isdigit():
                self.asm_code.append(f"    mov {reg_left}, {left}")
            else:
                self.asm_code.append(f"    mov {reg_left}, QWORD [rbp - {self.var_map[left]['stack_pos']}]")
                
            if right.isdigit():
                self.asm_code.append(f"    cmp {reg_left}, {right}")
            else:
                self.asm_code.append(f"    cmp {reg_left}, QWORD [rbp - {self.var_map[right]['stack_pos']}]")
                
            self.asm_code.append(f"    jne {label}")
            self.free_register(reg_left)
        else:
            raise NotImplementedError(f"Condition type not implemented: {condition}")
            
    def _generate_print(self, value):
        # Simplified print implementation
        if value in self.var_map:
            if self.var_map[value]['type'] == 'string':
                # For strings, use write syscall
                self.asm_code.append(f"    ; Print string {value}")
                self.asm_code.append(f"    mov rax, 1")  # sys_write
                self.asm_code.append(f"    mov rdi, 1")  # stdout
                self.asm_code.append(f"    mov rsi, {value}")
                self.asm_code.append(f"    mov rdx, {len(self.var_map[value]['value']) + 1}")
                self.asm_code.append(f"    syscall")
            else:
                # For numbers, we'd need a number-to-string conversion (simplified here)
                self.asm_code.append(f"    ; Print number {value}")
                self.asm_code.append(f"    mov rax, QWORD [rbp - {self.var_map[value]['stack_pos']}]")
                # In a real implementation, we'd convert the number to a string here
        else:
            # Literal value
            if value.startswith('"'):
                # Handle string literals (should have been added to var_map)
                pass
            else:
                # Numeric literal
                self.asm_code.append(f"    ; Print literal {value}")
                
    def _generate_function_call(self, func_name, args):
        # Push arguments in reverse order
        for arg in reversed(args):
            arg = arg.strip()
            if arg.isdigit():
                self.asm_code.append(f"    mov rax, {arg}")
                self.asm_code.append("    push rax")
            else:
                self.asm_code.append(f"    push QWORD [rbp - {self.var_map[arg]['stack_pos']}]")
                
        # Call function
        self.asm_code.append(f"    call {func_name}")
        
        # Clean up stack (assuming caller cleans up)
        if args:
            self.asm_code.append(f"    add rsp, {8 * len(args)}")
            
    def get_asm_code(self):
        return '\n'.join(self.asm_code)

# Example usage
def generate_assembly(intermediate_code):
    asm_gen = AssemblyGenerator()
    asm_gen.generate(intermediate_code)
    return asm_gen.get_asm_code()

# Test with your intermediate code
intermediate_code = [
    "t0 = 5 + 3",
    "x = t0",
    "print x"
]

print("Generated Assembly Code:")
print(generate_assembly(intermediate_code))
