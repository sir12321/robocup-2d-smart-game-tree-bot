#!/usr/bin/env python3
"""Build the single file ``agent.py`` from the ``agent/`` package.

The course harness loads exactly one file (``src/student/agent.py``), so the
readable, split-up source in ``agent/`` is concatenated into ``agent.py``:

* the modules are emitted in dependency order, each under a SECTION banner;
* relative imports (``from .config import ...``) are dropped, because every
  name now lives in the same file;
* absolute imports are merged into one header at the top;
* an ``__all__`` footer is appended.

Usage:
    python3 tools/bundle.py            # rewrite agent.py from agent/
    python3 tools/bundle.py --check    # exit 1 if agent.py is out of date
    python3 tools/bundle.py --verify OLD_AGENT_PY
        # check that agent.py and OLD_AGENT_PY are the same program:
        # identical code once docstrings, comments and import layout are
        # ignored (useful after documentation-only edits)
"""
import argparse
import ast
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "agent")
OUT = os.path.join(ROOT, "agent.py")

# (module file, banner title), in the order the code must appear.
SECTIONS = [
    ("config.py", "CONFIG & HYPERPARAMETERS"),
    ("stamina.py", "STAMINA PREDICTOR"),
    ("physics.py", "PHYSICS & TRAJECTORY ENGINE"),
    ("state.py", "STATE & KICK RESULT"),
    ("evaluators.py", "EVALUATOR MIXIN"),
    ("search.py", "SEARCH & EXPECTIMAX MIXIN"),
    ("on_ball.py", "ON-BALL ACTIONS MIXIN"),
    ("defense.py", "DEFENSE & MAN-MARKING MIXIN"),
    ("off_ball.py", "OFF-BALL POSITIONING MIXIN"),
    ("player.py", "AGENT PLAYERS"),
]

EXPORTS = ["Agent1", "Agent2", "AgentConfig", "StaminaPredictor",
           "_Phys", "_State", "_KickResult"]

HEADER = '''\
"""
Agent1: a search-based 5-a-side RoboCup 2D player (COL333 Assignment 3).

GENERATED FILE - do not edit by hand. The readable source is the agent/
package; rebuild this file with `python3 tools/bundle.py`.
"""
'''

STDLIB = {"math", "time", "itertools"}
BAR = "# " + "=" * 74


def _split_module(path):
    """Return (absolute import nodes, module body text without imports
    and without its leading module docstring)."""
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    drop = set()
    imports = []
    for i, node in enumerate(tree.body):
        is_doc = (i == 0 and isinstance(node, ast.Expr)
                  and isinstance(node.value, ast.Constant)
                  and isinstance(node.value.value, str))
        if isinstance(node, (ast.Import, ast.ImportFrom)) or is_doc:
            drop.update(range(node.lineno - 1, node.end_lineno))
            if isinstance(node, ast.ImportFrom) and node.level == 0:
                imports.append(node)
            elif isinstance(node, ast.Import):
                imports.append(node)
    body = "".join(l for j, l in enumerate(lines) if j not in drop)
    return imports, body.strip("\n")


def _merge_imports(nodes):
    plain, froms = [], {}
    for n in nodes:
        if isinstance(n, ast.Import):
            for a in n.names:
                line = f"import {a.name}" + (f" as {a.asname}" if a.asname else "")
                if line not in plain:
                    plain.append(line)
        else:
            names = froms.setdefault(n.module, [])
            for a in n.names:
                s = a.name + (f" as {a.asname}" if a.asname else "")
                if s not in names:
                    names.append(s)

    def top(mod_line):
        return mod_line.split()[1].split(".")[0]

    lines = plain + [f"from {m} import {', '.join(ns)}" for m, ns in froms.items()]
    std = sorted((l for l in lines if top(l) in STDLIB), key=lambda l: (l.startswith("from"), l))
    third = sorted((l for l in lines if top(l) not in STDLIB and top(l) != "src"),
                   key=lambda l: (l.startswith("from"), l))
    own = sorted(l for l in lines if top(l) == "src")
    return "\n".join(std) + "\n\n" + "\n".join(third + own) + "\n"


def build():
    imports, parts = [], []
    for fname, title in SECTIONS:
        imp, body = _split_module(os.path.join(PKG, fname))
        imports += imp
        parts.append(f"{BAR}\n# SECTION: {title} (agent/{fname})\n{BAR}\n\n\n{body}\n")
    out = HEADER + "\n" + _merge_imports(imports) + "\n\n" + "\n\n".join(parts)
    out += f"\n\n__all__ = {EXPORTS!r}\n"
    return out


class _StripDocs(ast.NodeTransformer):
    def _strip(self, node):
        self.generic_visit(node)
        b = node.body
        if (b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant)
                and isinstance(b[0].value.value, str)):
            node.body = b[1:] or [ast.Pass()]
        return node

    visit_Module = visit_ClassDef = visit_FunctionDef = visit_AsyncFunctionDef = _strip


def _program(path):
    """Code of a file with docstrings removed and imports split out."""
    tree = _StripDocs().visit(ast.parse(open(path, encoding="utf-8").read()))
    names = set()
    body = []
    for n in tree.body:
        if isinstance(n, ast.Import):
            names.update((a.name, a.asname) for a in n.names)
        elif isinstance(n, ast.ImportFrom):
            names.update((n.module, a.name, a.asname) for a in n.names)
        else:
            body.append(ast.dump(n))
    return names, body


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--check", action="store_true",
                   help="exit 1 if agent.py differs from a fresh build")
    g.add_argument("--verify", metavar="OLD_AGENT_PY",
                   help="check agent.py runs the same code as OLD_AGENT_PY")
    a = ap.parse_args()

    if a.verify:
        same = _program(OUT) == _program(a.verify)
        print("same program" if same else "DIFFERENT program")
        return 0 if same else 1

    text = build()
    if a.check:
        current = open(OUT, encoding="utf-8").read() if os.path.exists(OUT) else ""
        if current != text:
            print("agent.py is out of date: run `python3 tools/bundle.py`")
            return 1
        print("agent.py is up to date")
        return 0

    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"wrote {os.path.relpath(OUT, ROOT)} ({text.count(chr(10))} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
