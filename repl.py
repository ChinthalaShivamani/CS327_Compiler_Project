import os
import sys
import time
import re

# Handle readline import for different platforms
try:
    import readline  # Unix/Mac
except ImportError:
    try:
        import pyreadline3 as readline  # Windows alternative
    except ImportError:
        # If neither is available, continue without readline support
        pass

# Global variable storage for the REPL session
variables = {}

def print_klang_banner():
    """Prints a visually appealing Klang banner"""
    # Colors for terminal
    BLUE = "\033[1;34m"
    GREEN = "\033[1;32m"
    YELLOW = "\033[1;33m"
    RESET = "\033[0m"
    
    banner = [
        f"{BLUE}{RESET}",
        f"{BLUE}{RESET} {YELLOW}██╗  ██╗{RESET}{GREEN}██╗      █████╗ ███╗   ██╗ ██████╗{RESET} {BLUE}{RESET}",
        f"{BLUE}{RESET} {YELLOW}██║ ██╔╝{RESET}{GREEN}██║     ██╔══██╗████╗  ██║██╔════╝{RESET} {BLUE}{RESET}",
        f"{BLUE}{RESET} {YELLOW}█████╔╝ {RESET}{GREEN}██║     ███████║██╔██╗ ██║██║  ███╗{RESET} {BLUE}{RESET}",
        f"{BLUE}{RESET} {YELLOW}██╔═██╗ {RESET}{GREEN}██║     ██╔══██║██║╚██╗██║██║   ██║{RESET} {BLUE}{RESET}",
        f"{BLUE}{RESET} {YELLOW}██║  ██╗{RESET}{GREEN}███████╗██║  ██║██║ ╚████║╚██████╔╝{RESET} {BLUE}{RESET}",
        f"{BLUE}{RESET} {YELLOW}╚═╝  ╚═╝{RESET}{GREEN}╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝{RESET} {BLUE}{RESET}",
        f"{BLUE}{RESET}"
    ]
    
    # Print banner with animation
    for line in banner:
        print(line)
        time.sleep(0.07)  # Slightly slower for better effect
    
    print(f"\n{BLUE}{'═' * 50}{RESET}")
    print(f"{GREEN}  Interactive REPL for Klang Programming Language{RESET}")
    print(f"{YELLOW}  Version 1.0 - Type 'help' for available commands{RESET}")
    print(f"{BLUE}{'═' * 50}{RESET}\n")

def evaluate_expression(expr):
    """Evaluate a Klang expression, supporting variables"""
    global variables
    
    # Replace variables with their values
    for var_name, var_value in variables.items():
        if isinstance(var_value, str):
            # If it's a string, wrap in quotes for proper Python evaluation
            expr = expr.replace(var_name, f'"{var_value}"')
        else:
            expr = expr.replace(var_name, str(var_value))
    
    try:
        return eval(expr)
    except:
        return expr  # Return as is if can't evaluate

def execute_klang_code(code_str):
    """
    Simple interpreter for basic Klang language features
    """
    global variables
    code_str = code_str.strip()
    
    if not code_str:
        return ""

    # Handle printk statements
    printk_match = re.match(r'printk\s*\{(.*?)\}', code_str)
    if printk_match:
        content = printk_match.group(1).strip()
        
        # If content is a variable name
        if content in variables:
            return str(variables[content])
        # If content is a string literal (with quotes)
        elif (content.startswith('"') and content.endswith('"')) or \
             (content.startswith("'") and content.endswith("'")):
            return content[1:-1]  # Remove the quotes
        else:
            # Try to evaluate as an expression
            try:
                result = evaluate_expression(content)
                return str(result)
            except:
                return content

    # Handle variable declarations and assignments
    var_match = re.match(r'(int|float|string|var)\s+(\w+)\s*=\s*(.*)', code_str)
    if var_match:
        var_type, var_name, var_value = var_match.groups()
        
        # Evaluate the right side
        try:
            evaluated_value = evaluate_expression(var_value)
            
            # Type checking/conversion based on declared type
            if var_type == 'int':
                variables[var_name] = int(float(evaluated_value))
            elif var_type == 'float':
                variables[var_name] = float(evaluated_value)
            elif var_type == 'string':
                variables[var_name] = str(evaluated_value)
            else:  # var type (dynamic)
                variables[var_name] = evaluated_value
                
            return f"Variable {var_name} = {variables[var_name]}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    # Handle simple assignments (without type)
    assign_match = re.match(r'(\w+)\s*=\s*(.*)', code_str)
    if assign_match:
        var_name, var_value = assign_match.groups()
        
        # Check if variable exists
        if var_name not in variables:
            return f"Error: Variable '{var_name}' not declared"
        
        # Evaluate and assign
        try:
            variables[var_name] = evaluate_expression(var_value)
            return f"Variable {var_name} = {variables[var_name]}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    # Handle compound assignment operators (+=, -=, *=, /=)
    compound_match = re.match(r'(\w+)\s*(\+=|\-=|\*=|\/=)\s*(.*)', code_str)
    if compound_match:
        var_name, operator, value = compound_match.groups()
        
        # Check if variable exists
        if var_name not in variables:
            return f"Error: Variable '{var_name}' not declared"
        
        # Evaluate right side
        try:
            right_value = evaluate_expression(value)
            
            # Apply operation
            if operator == '+=':
                variables[var_name] += right_value
            elif operator == '-=':
                variables[var_name] -= right_value
            elif operator == '*=':
                variables[var_name] *= right_value
            elif operator == '/=':
                variables[var_name] /= right_value
                
            return f"Variable {var_name} = {variables[var_name]}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    # If code doesn't match any pattern, return a generic message
    return f"Executing: {code_str}"

def handle_special_commands(command):
    """Handle special REPL commands"""
    cmd = command.strip().lower()
    
    if cmd == "exit" or cmd == "quit":
        print("Exiting Klang REPL...")
        sys.exit(0)
    elif cmd == "clear":
        os.system('cls' if os.name == 'nt' else 'clear')
        return True
    elif cmd == "help":
        print_help()
        return True
    elif cmd == "vars":
        # New command to show all variables
        print_variables()
        return True
    elif cmd.startswith("load "):
        filename = cmd[5:].strip()
        try:
            with open(filename, 'r') as f:
                content = f.read()
                print(f"Loaded file: {filename}")
                print("-" * 40)
                print(content)
                print("-" * 40)
                
                # Split the file content by lines and execute each line
                results = []
                for line in content.split('\n'):
                    if line.strip():
                        result = execute_klang_code(line)
                        if result:
                            results.append(result)
                
                if results:
                    print("Results:")
                    for result in results:
                        print(result)
        except Exception as e:
            print(f"Error loading file: {str(e)}")
        return True
    
    return False  # Not a special command

def print_variables():
    """Print all variables in the current session"""
    if not variables:
        print("No variables defined")
        return
    
    print("\nCurrent Variables:")
    print("-" * 30)
    for name, value in variables.items():
        var_type = type(value).__name__
        print(f"{name} ({var_type}) = {value}")
    print()

def print_help():
    """Print help information"""
    GREEN = "\033[1;32m"
    YELLOW = "\033[1;33m"
    RESET = "\033[0m"
    
    print(f"\n{GREEN}Klang REPL Commands:{RESET}")
    print(f"  {YELLOW}help{RESET}         - Show this help message")
    print(f"  {YELLOW}vars{RESET}         - Display all defined variables")
    print(f"  {YELLOW}exit, quit{RESET}   - Exit the REPL")
    print(f"  {YELLOW}clear{RESET}        - Clear the screen")
    print(f"  {YELLOW}load <file>{RESET}  - Load and execute a Klang file\n")
    
    print(f"{GREEN}Examples:{RESET}")
    print(f"  {YELLOW}int a = 5{RESET}")
    print(f"  {YELLOW}float pi = 3.14{RESET}")
    print(f"  {YELLOW}string greeting = \"Hello\"{RESET}")
    print(f"  {YELLOW}a += 10{RESET}")
    print(f"  {YELLOW}printk{{a}}{RESET}")
    print(f"  {YELLOW}printk{{\"Hello, world!\"}}{RESET}")
    print(f"  {YELLOW}printk{{a + 5}}{RESET}\n")

def repl():
    """Main REPL function for Klang"""
    print_klang_banner()
    
    while True:
        try:
            # Prompt with different color if supported
            user_input = input("\033[1;36m> \033[0m")
            
            # Check for special commands
            if handle_special_commands(user_input):
                continue
                
            # Execute regular Klang code
            if user_input.strip():
                result = execute_klang_code(user_input)
                if result:
                    print(result)
                    
        except KeyboardInterrupt:
            print("\nUse 'exit' or 'quit' to exit")
        except EOFError:
            print("\nExiting Klang REPL...")
            break
        except Exception as e:
            print(f"\033[1;31mError: {str(e)}\033[0m")

if __name__ == "__main__":
    repl()
