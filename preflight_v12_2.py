import ast, pathlib, sys
p=pathlib.Path(__file__).with_name('app.py')
s=p.read_text(encoding='utf-8'); tree=ast.parse(s)
defs={n.name for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))}
called={n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}
allowed=set(dir(__builtins__))|{'print'}
missing=sorted(x for x in called if x.startswith('_') and x not in defs and x not in allowed)
if missing:
    print('MISSING INTERNAL FUNCTIONS:', ', '.join(missing)); sys.exit(2)
print('Preflight OK: internal function references are present.')
