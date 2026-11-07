#!/usr/bin/env python3
"""Topic views of agent.py, for reading and debugging one front at a time.
agent.py stays the real code (the harness loads only that file); these are
views of it, split by topic. Run from anywhere:

    python3 agent_breakdown/parts.py split   agent.py -> agent_breakdown/*.py (agent.py untouched)
    python3 agent_breakdown/parts.py check   are agent_breakdown/ and agent.py the same code?
    python3 agent_breakdown/parts.py build   agent_breakdown/*.py -> agent.py (backup: agent.py.bak)

Each part file holds verbatim slices ("chunks") of agent.py between
`# ==== begin ID ====` / `# ==== end ID ====` markers; everything outside
the markers is ignored by `build`. `build` puts the chunks back in the
original order (recorded in agent_breakdown/.manifest.json), so split followed
by build reproduces agent.py byte for byte. A new method can be added to
a part inside its own begin/end markers; it is placed after the last chunk
of that part's class.
"""
import ast
import json
import os
import re
import shutil
import sys

PARTS = os.path.dirname(os.path.abspath(__file__))
AGENT = os.path.join(PARTS, "..", "agent.py")
MANIFEST = os.path.join(PARTS, ".manifest.json")
TOOL = os.path.basename(__file__)       # this script lives among the parts
MAIN = "Agent1"

# part file -> (what it covers, chunk ids). Anything not listed lands in misc.py.
LAYOUT = {
    "physics.py": ("Module header, stamina recovery tables, server physics, the race model, search states.",
                   ["<module>", "_sig", "_norm_deg", "_Phys", "_kick_traj", "_race", "_first_true",
                    "_State", "_KickResult"]),
    "stamina.py": ("Stamina estimates for every player, sprint budgets, recovery mode and its signal.",
                   ["StaminaPredictor", "Agent1._stam_cycle_start", "Agent1._flat_out",
                    "Agent1._race_budgets", "Agent1._recover_cap", "Agent1._rec_signal"]),
    "config.py": ("Agent1 class header and tuning constants, per-cycle context, risk, time budget.",
                  ["Agent1", "Agent1.__init__", "Agent1._phys", "Agent1._allow_ms", "Agent1._context",
                   "Agent1._root_state", "Agent1._risk", "Agent1._counter_state"]),
    "on_ball.py": ("on_ball entry point: committed kicks, fallback evaluator, pressure, hold, clear, execution.",
                   ["Agent1.on_ball", "Agent1._on_ball", "Agent1._pressure", "Agent1._multi_keep",
                    "Agent1._eval_hold", "Agent1._eval_clear", "Agent1._execute", "Agent1._touch_out",
                    "Agent1._execute_base", "Agent1._by_their_line"]),
    "chain.py": ("Game-tree beam search over pass/dribble chains and its scoring.",
                 ["Agent1._field_eval", "Agent1._chain_scores", "Agent1._step_cost",
                  "Agent1._on_ball_chain", "Agent1._beam_chain", "Agent1._beam_run",
                  "Agent1._beam_keep", "Agent1._beam_expand", "Agent1._child"]),
    "passing.py": ("Pass targets, kick evaluation (race), kick timing, look-ahead, first-time and special kicks.",
                   ["Agent1._pass_targets", "Agent1._eval_kicks", "Agent1._eval_kicks1",
                    "Agent1._deep_value", "Agent1._kick_delay", "Agent1._vmax", "Agent1._live",
                    "Agent1._channel_kick", "Agent1._cone_mask", "Agent1._bounce_pass"]),
    "shoot.py": ("Shot evaluation (P(goal) incl. the keeper race) and the carry-in before a shot.",
                 ["Agent1._eval_shots", "Agent1._shot1_speeds", "Agent1._carry_in"]),
    "dribble.py": ("Dribbles and self-passes simulated with player physics, their scores and values.",
                   ["Agent1._dribs", "Agent1._drib_options", "Agent1._eval_dribbles"]),
    "off_ball.py": ("off_ball entry point: zone tackle, chase/contest, blocker engage, dispatch; home slot, dash power.",
                    ["Agent1.off_ball", "Agent1._off_ball", "Agent1._home", "Agent1._busy",
                     "Agent1._dash_power", "Agent1._contest"]),
    "support.py": ("Attacking movement without the ball: support spots, runs/diagonals, outlets.",
                   ["Agent1._support_point", "Agent1._run_spot", "Agent1._is_outlet"]),
    "defence.py": ("Defending: goal-side recovery, blocker plan, man-marking with min-max assignment, set-piece covers.",
                   ["Agent1._goal_side", "Agent1._carrier", "Agent1._block_plan", "Agent1._serious",
                    "Agent1._defend_point", "Agent1._defense_plan", "Agent1._set_piece",
                    "Agent1._sp_covers", "Agent1._man_marks", "Agent1._goal_view", "Agent1._lane_spot"]),
}
MISC = "misc.py"
BEGIN = re.compile(r"^[ \t]*# ==== begin (\S+) ====[ \t]*$")
END = re.compile(r"^[ \t]*# ==== end (\S+) ====[ \t]*$")


def chunks_of(src):
    """[(id, text)] covering src exactly, in order. A chunk is a top-level
    def/class (Agent1 split into its header and one chunk per method) plus
    the blank lines and comments in front of it."""
    lines = src.splitlines(keepends=True)
    tree = ast.parse(src)
    spans = []                      # (id, first line, last line), 1-based
    body = tree.body
    first_def = next((n for n in body if isinstance(n, (ast.FunctionDef, ast.ClassDef))), None)
    head_end = (min([first_def.lineno] + [d.lineno for d in first_def.decorator_list]) - 1) if first_def else len(lines)
    spans.append(["<module>", 1, head_end])
    prev = head_end
    for n in body:
        if not isinstance(n, (ast.FunctionDef, ast.ClassDef)) or n.lineno <= head_end:
            continue
        start = prev + 1
        if isinstance(n, ast.ClassDef) and n.name == MAIN:
            methods = [m for m in n.body if isinstance(m, ast.FunctionDef)]
            m0 = min([methods[0].lineno] + [d.lineno for d in methods[0].decorator_list])
            # the class header keeps its constants; the comments right above
            # the first method go with that method
            hdr_end = m0 - 1
            while hdr_end > n.lineno and (lines[hdr_end - 1].strip() == ""
                                          or lines[hdr_end - 1].strip().startswith("#")):
                hdr_end -= 1
            spans.append([MAIN, start, hdr_end])
            p = hdr_end
            for m in methods:
                spans.append([f"{MAIN}.{m.name}", p + 1, m.end_lineno])
                p = m.end_lineno
            prev = n.end_lineno
            spans[-1][2] = prev
            continue
        spans[-1][2] = prev
        spans.append([n.name, start, n.end_lineno])
        prev = n.end_lineno
    spans[-1][2] = len(lines)
    out = []
    for cid, a, b in spans:
        out.append((cid, "".join(lines[a - 1:b])))
    assert "".join(t for _, t in out) == src, "chunking lost text"
    return out


def part_of(cid):
    for fname, (_, ids) in LAYOUT.items():
        if cid in ids:
            return fname
    return MISC


def render(fname, items):
    desc = LAYOUT.get(fname, ("Methods not assigned to a topic file yet.", []))[0]
    out = [f'"""{desc}\n\nPart view of agent.py, written by `agent_breakdown/parts.py split`.\n'
           "Edit the code between the begin/end markers, then run `agent_breakdown/parts.py build`\n"
           'to write it back into agent.py. Text outside the markers is ignored."""\n\n']
    top = [(c, t) for c, t in items if "." not in c]
    meth = [(c, t) for c, t in items if "." in c]
    for cid, text in top:
        out.append(f"# ==== begin {cid} ====\n{text}")
        if not text.endswith("\n"):
            out.append("\n")
        out.append(f"# ==== end {cid} ====\n\n")
    if meth:
        if MAIN not in [c for c, _ in top]:
            out.append(f"class {MAIN}:  # part view only: these are methods of agent.py's {MAIN}\n")
        for cid, text in meth:
            out.append(f"    # ==== begin {cid} ====\n{text}")
            if not text.endswith("\n"):
                out.append("\n")
            out.append(f"    # ==== end {cid} ====\n\n")
    return "".join(out)


def read_parts():
    """{id: (file, text)} from agent_breakdown/, in file order."""
    got = {}
    for fname in sorted(os.listdir(PARTS)):
        if not fname.endswith(".py") or fname == TOOL:
            continue
        cur, buf = None, []
        for i, line in enumerate(open(os.path.join(PARTS, fname)), 1):
            m = BEGIN.match(line)
            if m:
                if cur is not None:
                    sys.exit(f"{fname}:{i}: begin {m.group(1)} inside {cur}")
                cur, buf = m.group(1), []
                continue
            m = END.match(line)
            if m:
                if m.group(1) != cur:
                    sys.exit(f"{fname}:{i}: end {m.group(1)} does not close {cur}")
                if cur in got:
                    sys.exit(f"{fname}:{i}: chunk {cur} also in {got[cur][0]}")
                got[cur] = (fname, "".join(buf))
                cur = None
                continue
            if cur is not None:
                buf.append(line)
        if cur is not None:
            sys.exit(f"{fname}: chunk {cur} never ended")
    return got


def assemble():
    order = json.load(open(MANIFEST))["order"]
    got = read_parts()
    missing = [c for c in order if c not in got]
    if missing:
        sys.exit("chunks missing from agent_breakdown/: " + ", ".join(missing))
    extra = [c for c in got if c not in order]
    seq = list(order)
    for cid in extra:
        # new chunk: after the last chunk from the same file of the same kind
        fname = got[cid][0]
        same = [i for i, c in enumerate(seq) if got[c][0] == fname and ("." in c) == ("." in cid)]
        if "." in cid and not same:
            same = [max(i for i, c in enumerate(seq) if c.startswith(MAIN + "."))]
        seq.insert((same[-1] + 1) if same else len(seq), cid)
    return "".join(got[c][1] for c in seq), extra


def split(force=False):
    src = open(AGENT).read()
    if os.path.exists(MANIFEST) and not force:
        built, _ = assemble()
        if built != src:
            sys.exit("agent_breakdown/ has edits that are not in agent.py: run `build` first, "
                     "or `split --force` to discard them")
    chunks = chunks_of(src)
    for f in os.listdir(PARTS):
        if f.endswith(".py") and f != TOOL:
            os.remove(os.path.join(PARTS, f))
    by_file = {}
    for cid, text in chunks:
        by_file.setdefault(part_of(cid), []).append((cid, text))
    for fname, items in by_file.items():
        open(os.path.join(PARTS, fname), "w").write(render(fname, items))
    json.dump({"order": [c for c, _ in chunks]}, open(MANIFEST, "w"), indent=1)
    for fname, items in sorted(by_file.items()):
        n = sum(t.count("\n") for _, t in items)
        print(f"  agent_breakdown/{fname:12s} {n:5d} lines  " + ", ".join(c.split(".")[-1] for c, _ in items))
    print("agent.py untouched; split -> build reproduces it exactly:", assemble()[0] == src)


def check():
    src = open(AGENT).read()
    built, extra = assemble()
    if built == src:
        print("in sync: agent_breakdown/ and agent.py are the same code")
        return 0
    import difflib
    sys.stdout.writelines(difflib.unified_diff(src.splitlines(True), built.splitlines(True),
                                               "agent.py", "agent_breakdown (built)", n=2))
    if extra:
        print("new chunks: " + ", ".join(extra))
    return 1


def build():
    src = open(AGENT).read()
    built, extra = assemble()
    if built == src:
        print("nothing to do: agent.py already matches agent_breakdown/")
        return
    try:
        ast.parse(built)
    except SyntaxError as e:
        sys.exit(f"not written, the result does not parse: line {e.lineno}: {e.msg}")
    shutil.copy(AGENT, AGENT + ".bak")
    open(AGENT, "w").write(built)
    json.dump({"order": [c for c, _ in chunks_of(built)]}, open(MANIFEST, "w"), indent=1)
    print("agent.py rebuilt from agent_breakdown/ (previous version in agent.py.bak)"
          + (f"; new chunks: {', '.join(extra)}" if extra else ""))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "split":
        split(force="--force" in sys.argv)
    elif cmd == "check":
        sys.exit(check())
    elif cmd == "build":
        build()
    else:
        sys.exit(__doc__)
