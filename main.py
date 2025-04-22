# main.py
from bytecode import compile_code, BytecodeGenerator
from vm import VirtualMachine
from example import Parser, tokenize
from example import Evaluator

# Your source code
source_code = """
# Function to generate Fibonacci numbers up to a limit
func generate_fibonacci(int limit) begin
    int a = 0
    int b = 1
    int fibs[0]
    int next = 0
    
    fibs.add(a)
    if (limit > 0) begin
        fibs.add(b)
    end
    
    int continue_loop = 1
    while (continue_loop == 1) begin
        next = a + b
        if (next >= limit) begin
            continue_loop = 0
        else begin
            fibs.add(next)
            a = b
            b = next
        end
    end
    
    return fibs
end

# Function to check if a number is prime
func is_prime(int n) begin
    if (n <= 1) begin
        return 0
    end
    if (n == 2) begin
        return 1
    end
    if (n % 2 == 0) begin
        return 0
    end
    
    int i = 3
    while (i * i <= n) begin
        if (n % i == 0) begin
            return 0
        end
        i = i + 2
    end
    return 1
end

# Modified is_prime_power function to correctly identify prime powers
func is_prime_power(int n) begin
    if (n <= 1) begin
        return 0
    end
    
    # First check if it's prime itself (p^1)
    if (is_prime(n)) begin
        return 1
    end
    
    # Check for higher powers
    int p = 2
    while (p * p <= n) begin
        if (is_prime(p)) begin  # Only check prime bases
            int temp = n
            int power = 0
            while (temp % p == 0) begin
                temp = temp / p
                power = power + 1
            end
            if (temp == 1 and power > 0) begin
                return 1
            end
        end
        p = p + 1
    end
    
    return 0
end

# Function to find largest Fibonacci prime power under limit
func find_largest_fib_prime_power(int limit) begin
    int fibs[] = generate_fibonacci(limit)
    int largest = 0
    int i = 0
    
    printk {"\nChecking Fibonacci numbers under "}
    printk {limit}
    printk {" for prime powers:"}
    
    int idx = fibs.length() - 1
    int found = 0
    while (idx >= 0 and found == 0) begin  # Changed && to and
        i = fibs[idx]
        if (is_prime_power(i)) begin
            largest = i
            found = 1
        end
        idx = idx - 1
    end
    
    return largest
end

# Main program
printk {"\n=== Fibonacci Prime Powers ==="}

int limit1 = 100
int result1 = find_largest_fib_prime_power(limit1)
printk {"\n\nLargest Fibonacci prime power under "}
printk {limit1}
printk {": "}
printk {result1}

int limit2 = 10000
int result2 = find_largest_fib_prime_power(limit2)
printk {"\nLargest Fibonacci prime power under "}
printk {limit2}
printk {": "}
printk {result2}

int limit3 = 1000000
int result3 = find_largest_fib_prime_power(limit3)
printk {"\nLargest Fibonacci prime power under "}
printk {limit3}
printk {": "}
printk {result3}
"""

# 1. Display tokens
tokens = list(tokenize(source_code))
print("=== Tokens ===")
for i, token in enumerate(tokens):
    print(f"{i}: {token}")
print()

# 2. Parse the code to create AST
parser = Parser(tokens)
ast = parser.parse()
print("=== Abstract Syntax Tree ===")
print(ast)
print()

# Remove or comment out the Evaluator part if not needed for VM testing
# evaluator = Evaluator(parser.functions)
# evaluator.evaluate(ast)
# print(evaluator)

# 3. Compile the source code to bytecode
# FIX: Receive both bytecode and function definitions
bytecode, function_definitions = compile_code(source_code)
print("=== Bytecode ===")
for i, instr in enumerate(bytecode):
    print(f"{i}: {instr}")
print()

# 4. Create and run the VM
vm = VirtualMachine(functions=function_definitions)
vm.load(bytecode)

# Execute and get output with increased step limit
output = vm.run(max_steps=10000000)  # Increased to 10 million steps
print("=== Program Output ===")
print(output)
