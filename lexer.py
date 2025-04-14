import re

def tokenize(code):
    token_specification = [
        ('INT', r'int\b'),
        ('FLOAT', r'float\b'),
        ('STRING_TYPE', r'string\b'),
        ('CONST', r'const\b'),
        ('VAR', r'var\b'),
        ('IF', r'if\b'),
        ('ELIF', r'elif\b'),
        ('ELSE', r'else\b'),
        ('WHILE', r'while\b'),
        ('FOR', r'for\b'),
        ('FUNC', r'func\b'),
        ('RETURN', r'return\b'),
        ('IN', r'in\b'),
        ('BEGIN', r'begin\b'),
        ('END', r'end\b'),
        ('REPEAT', r'repeat\b'),
        # Multi-character operators
        ('EQ', r'=='),
        ('GE', r'>='),
        ('LE', r'<='),
        # Compound assignment operators
        ('PLUS_ASSIGN', r'\+='),
        ('MINUS_ASSIGN', r'-='),
        ('MULT_ASSIGN', r'\*='),
        ('DIV_ASSIGN', r'/='),
        ('MOD_ASSIGN', r'%='),
        # Single-character operators
        ('ASSIGN', r'='),
        ('GT', r'>'),
        ('LT', r'<'),
        ('NUMBER', r'\d+(\.\d+)?'),
        ('STRING', r'".*?"'),
        ('IDENT', r'[a-zA-Z_][a-zA-Z_0-9]*'),
        ('OP', r'[+\-*/%]'),
        ('LPAREN', r'\('),
        ('RPAREN', r'\)'),
        ('LBRACE', r'\{'),
        ('RBRACE', r'\}'),
        ('COLON', r':'),
        ('COMMA', r','),
        ('LBRACKET', r'\['),
        ('RBRACKET', r'\]'),
        ('DOT', r'\.'),
        ('NEWLINE', r'\n'),
        ('SKIP', r'[ \t]+'),
        ('COMMENT', r'#.*'),
        ('MISMATCH', r'.')
    ]

    # Fix the token regex construction
    tok_regex = '|'.join(f"(?P<{pair[0]}>{pair[1]})" for pair in token_specification)
    
    # Track bracket balance
    bracket_stack = []
    
    i = 0
    while i < len(code):
        # Check for multiline string
        if i < len(code) and code[i] == '|':
            # Find the closing vertical bar
            end = code.find('\n|', i + 1)
            if end == -1:
                raise SyntaxError("Unterminated multiline string, missing closing '|'")
            
            # Extract the string content (excluding delimiters)
            content = code[i+1:end]
            
            # Handle indentation - remove common leading whitespace
            lines = content.split('\n')
            
            # Find minimum indentation (ignoring empty lines)
            non_empty_lines = [line for line in lines if line.strip()]
            if non_empty_lines:
                min_indent = min(len(line) - len(line.lstrip()) for line in non_empty_lines)
                # Remove the common indentation
                processed_lines = []
                for line in lines:
                    if line.strip():  # If line is not empty
                        processed_lines.append(line[min_indent:])
                    else:
                        processed_lines.append(line)  # Keep empty lines as is
                content = '\n'.join(processed_lines)
            
            yield 'STRING', content
            i = end + 2  # Skip past the ending | and newline
            continue
        
        # Regular token matching
        match = re.match(tok_regex, code[i:])
        if match:
            kind = match.lastgroup
            value = match.group(kind)
            
            if kind == 'LBRACKET':
                bracket_stack.append('[')
            elif kind == 'RBRACKET':
                if bracket_stack and bracket_stack[-1] == '[':
                    bracket_stack.pop()
                else:
                    raise SyntaxError("Unexpected closing bracket ']' without matching opening bracket")
            
            if kind == 'NUMBER':
                value = float(value) if '.' in value else int(value)
            elif kind == 'STRING':
                # Process escape sequences in string literals
                value = value.strip('"')
                # Convert escape sequences like \n to actual newlines
                try:
                    value = bytes(value, "utf-8").decode("unicode_escape")
                except:
                    # If there's an error in escape sequence processing, keep original
                    pass
            elif kind in ['SKIP', 'COMMENT']:
                i += match.end()
                continue
            elif kind == 'MISMATCH':
                raise SyntaxError(f"Unexpected token: {value}")
            
            yield kind, value
            
            i += match.end()
        else:
            i += 1
    
    # Check for unclosed brackets at the end
    if bracket_stack:
        raise SyntaxError(f"Array is not closed. Missing closing bracket ']'")
