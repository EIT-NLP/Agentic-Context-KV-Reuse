"""Bounded arithmetic AST evaluation. No Python expression execution."""
import ast
import io
import tokenize
import json
import math
import operator
import re
import subprocess
import sys
from pathlib import Path


FUNCTIONS = {'sqrt', 'sin', 'cos', 'tan', 'exp', 'log', 'factorial', 'abs', 'binomial'}


def _math_tokens(text, symbolic):
    """Normalize only explicit math tokens, without a Python parser/eval rewrite."""
    tokens = [t for t in tokenize.generate_tokens(io.StringIO(text).readline)
              if t.type not in (tokenize.ENDMARKER, tokenize.NEWLINE, tokenize.NL)]
    if len(tokens) > 150:
        raise ValueError('Expression too complex')
    output = []
    for token in tokens:
        value = token.string
        if value.isspace():
            continue
        if value == '!':
            if not output or output[-1] == '!':
                raise ValueError('Invalid factorial')
            start = len(output) - 1
            if output[-1] == ')':
                balance = 1
                while start > 0 and balance:
                    start -= 1
                    balance += (output[start] == ')') - (output[start] == '(')
                if balance:
                    raise ValueError('Unbalanced factorial')
                if start and output[start-1] in FUNCTIONS:
                    start -= 1
            elif not (output[-1].isalnum()):
                raise ValueError('Invalid factorial operand')
            output[start:] = ['factorial', '('] + output[start:] + [')']
            continue
        if token.type not in (tokenize.NAME, tokenize.NUMBER, tokenize.OP):
            raise ValueError('Unsupported math token')
        if token.type == tokenize.OP and value not in {'+', '-', '*', '/', '**', '(', ')', '[', ']', ','}:
            raise ValueError('Unsupported math operator')
        # n(n-3), 2n, (n+1)(n-1), and (n+1)n are products; whitelist calls remain calls.
        if output:
            previous = output[-1]
            previous_is_atom = (previous in {')', ']'} or previous[0].isdigit() or
                                previous[0].isalpha())
            begins_atom = token.type in (tokenize.NAME, tokenize.NUMBER) or value == '('
            is_call = previous in FUNCTIONS and value == '('
            if previous_is_atom and begins_atom and not is_call:
                if not symbolic and previous.isalpha() and previous not in {'pi', 'e', 'E'}:
                    raise ValueError('Unsupported function')
                output.append('*')
        output.append(value)
    return ' '.join(output)


def expression(text, symbolic=False):
    text = str(text).strip()
    if len(text) > 1024:
        raise ValueError('Expression too long')
    text = text.replace('^', '**').replace('\\pi', 'pi').replace('π', 'pi')
    for token, replacement in [('\\left', ''), ('\\right', ''), ('\\cdot', '*'), ('\\times', '*'), ('$', '')]:
        text = text.replace(token, replacement)
    for _ in range(8):
        previous = text
        text = re.sub(r'\\(?:d?frac)\{([^{}]*)\}\{([^{}]*)\}', r'((\1)/(\2))', text)
        text = re.sub(r'\\sqrt\{([^{}]*)\}', r'sqrt(\1)', text)
        if text == previous:
            break
    text = re.sub(r'\{([^{}]*)\}', r'(\1)', text)
    text = re.sub(r'\bC\(', 'binomial(', text)
    text = _math_tokens(text, symbolic)
    tree = ast.parse(text, mode='eval')
    if sum(1 for _ in ast.walk(tree)) > 100:
        raise ValueError('Expression too complex')
    if symbolic:
        import sympy as m
    else:
        m = math
    functions = {n: getattr(m, n) for n in ('sqrt', 'sin', 'cos', 'tan', 'exp', 'log', 'factorial')}
    functions.update(abs=abs, binomial=m.binomial if symbolic else math.comb)
    constants = {'pi': m.pi, 'E': m.E if symbolic else math.e, 'e': m.E if symbolic else math.e}
    operations = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                  ast.Div: operator.truediv, ast.Pow: operator.pow}
    def walk(node, depth=0):
        if depth > 16:
            raise ValueError('Expression nesting limit')
        child = lambda n: walk(n, depth+1)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = node.value
        elif isinstance(node, (ast.List, ast.Tuple)):
            if symbolic:
                raise ValueError('Scalar expression required')
            return [child(n) for n in node.elts]
        elif isinstance(node, ast.Name) and node.id in constants:
            value = constants[node.id]
        elif symbolic and isinstance(node, ast.Name) and len(node.id) == 1 and node.id.isalpha():
            value = m.Symbol(node.id)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = child(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        elif isinstance(node, ast.BinOp) and type(node.op) in operations:
            a, b = child(node.left), child(node.right)
            if isinstance(a, list) or isinstance(b, list):
                raise ValueError('Arithmetic requires scalars')
            if isinstance(node.op, ast.Pow) and (getattr(b, 'free_symbols', ()) or abs(float(b)) > 100):
                raise ValueError('Exponent out of range')
            value = operations[type(node.op)](a, b)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in functions and not node.keywords:
            args = [child(n) for n in node.args]
            if not 1 <= len(args) <= 2 or any(
                    not (symbolic and getattr(a, 'free_symbols', ())) and abs(float(a)) > 100
                    for a in args):
                raise ValueError('Function argument out of range')
            value = functions[node.func.id](*args)
        else:
            raise ValueError('Unsupported arithmetic syntax')
        if not symbolic and (type(value) not in (int, float) or not math.isfinite(value) or abs(value) > 1e100):
            raise ValueError('Numeric result out of range')
        return value
    return walk(tree.body)


def symbolic_equal(left, right):
    # SymPy simplification is also bounded in a separate, killable process.
    if max(len(left), len(right)) > 1024:
        return False
    try:
        run = subprocess.run([sys.executable, str(Path(__file__).resolve())],
            input=json.dumps([left, right]), text=True, capture_output=True, timeout=4)
        return run.returncode == 0 and run.stdout.strip() == 'true'
    except (OSError, subprocess.TimeoutExpired):
        return False


if __name__ == '__main__':
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
        # macOS rejects lowering these limits for its large reserved address space.
        # CPU/wall limits and AST bounds still apply; production Linux enforces AS.
        if sys.platform != 'darwin':
            resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2, 512 * 1024**2))
        import sympy
        left, right = json.load(sys.stdin)
        print(json.dumps(bool(sympy.simplify(expression(left, True) - expression(right, True)) == 0)))
    except Exception:
        print('false')
