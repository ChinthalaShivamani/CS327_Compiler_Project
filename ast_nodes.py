# Define AST nodes for the language

class ASTNode:
    pass

class Program(ASTNode):
    def __init__(self, statements):
        self.statements = statements

class VariableDeclaration(ASTNode):
    def __init__(self, var_type, name, value, is_array=False, array_size=None):
        self.var_type = var_type
        self.name = name
        self.value = value
        self.is_array = is_array
        self.array_size = array_size

class PrintStatement(ASTNode):
    def __init__(self, expression):
        self.expression = expression

class RepeatLoop(ASTNode):
    def __init__(self, count, body):
        self.count = count
        self.body = body

class BinaryOperation(ASTNode):
    def __init__(self, left, operator, right):
        self.left = left
        self.operator = operator
        self.right = right

class Expression(ASTNode):
    def __init__(self, value):
        self.value = value

class IfElse(ASTNode):
    def __init__(self, condition, true_branch, elif_branches, false_branch):
        self.condition = condition
        self.true_branch = true_branch
        self.elif_branches = elif_branches
        self.false_branch = false_branch

class WhileLoop(ASTNode):
    def __init__(self, condition, body):
        self.condition = condition
        self.body = body

class ForLoop(ASTNode):
    def __init__(self, loop_var, iterable, body):
        self.loop_var = loop_var
        self.iterable = iterable
        self.body = body

class FunctionDeclaration(ASTNode):
    def __init__(self, name, params, body):
        self.name = name
        self.params = params
        self.body = body

class FunctionCall(ASTNode):
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments

class ReturnStatement(ASTNode):
    def __init__(self, expression):
        self.expression = expression

class ArrayLiteral(ASTNode):
    def __init__(self, elements):
        self.elements = elements

class ArrayAccess(ASTNode):
    def __init__(self, array, index):
        self.array = array
        self.index = index

class MethodCall(ASTNode):
    def __init__(self, object_name, method_name, arguments):
        self.object_name = object_name
        self.method_name = method_name
        self.arguments = arguments

class CompoundAssignment(ASTNode):
    def __init__(self, name, operator, value):
        self.name = name
        self.operator = operator
        self.value = value
