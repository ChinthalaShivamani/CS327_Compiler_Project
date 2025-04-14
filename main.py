import sys
from lexer import tokenize
from parser import Parser
from evaluator import Evaluator

def run_code(code):
    try:
        tokens = list(tokenize(code))
        parser = Parser(tokens)
        ast = parser.parse()
        evaluator = Evaluator(parser.functions)
        evaluator.evaluate(ast)
        return True
    except SyntaxError as e:
        print(f"SyntaxError: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Run code from file
        file_path = sys.argv[1]
        try:
            with open(file_path, 'r') as file:
                code = file.read()
            run_code(code)
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found")
    else:
        print("Usage: python main.py <filename>")
