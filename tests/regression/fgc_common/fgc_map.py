#!/usr/bin/env python3
"""Map simx per-PC counters (VORTEX_FGC_STATS) to source steps and events.

The kernel carries no markers. Steps and events are comments in its source:
  // @step <name>    lines from here on belong to <name>, until the next tag
  // @step end       lines from here on belong to no step of their own
  // @event <name>   counts how often this line ran (put it inside the branch)
Build the kernel with -gline-tables-only, which leaves the code unchanged.

Each PC maps to its source lines through llvm-symbolizer, inlined frames
included. A PC belongs to the step of its innermost frame whose line lies in a
step; a line in no step (before a file's first tag or after "@step end")
takes its caller's step, so a shared helper is charged to whoever called it.
A PC with no frame in a step is "other".

Usage: fgc_map.py stats.json kernel.elf [--lines] [--cores]
"""

import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict

SYMBOLIZER = os.path.expanduser("~/tools_v3/llvm-vortex/bin/llvm-symbolizer")
TAG = re.compile(r"//\s*@(step|event)\s+(\S+)")


def symbolize(elf, pcs):
    """PC -> list of (file, line) frames, innermost first. A PC the compiler
    gives line 0 takes the frames of the PC before it."""
    out = subprocess.run([SYMBOLIZER, "--obj=" + elf, "--inlining", "--output-style=JSON"],
                         input="\n".join(hex(pc) for pc in pcs) + "\n",
                         capture_output=True, text=True, check=True).stdout
    frames, prev = {}, []
    for pc, text in zip(pcs, out.splitlines()):
        fr = [(os.path.realpath(os.path.join(sym.get("Directory", ""), sym["FileName"])), sym["Line"])
              for sym in json.loads(text).get("Symbol", []) if sym.get("FileName")]
        if fr and fr[0][1] == 0:
            fr = prev
        frames[pc] = fr
        prev = fr
    return frames


def read_tags(path, cache={}):
    """file -> (sorted step tags [(line, name)], {line: event name}), or None."""
    if path in cache:
        return cache[path]
    steps, events = [], {}
    try:
        with open(path) as f:
            for n, text in enumerate(f, 1):
                m = TAG.search(text)
                if m:
                    if m.group(1) == "step":
                        steps.append((n, m.group(2)))
                    else:
                        events[n] = m.group(2)
    except OSError:
        pass
    cache[path] = (steps, events) if (steps or events) else None
    return cache[path]


def step_of(frames):
    for path, line in frames:
        tags = read_tags(path)
        if tags is None or not tags[0]:
            continue
        name = None
        for n, s in tags[0]:
            if n > line:
                break
            name = None if s == "end" else s
        if name is not None:
            return name
    return "other"


def events_of(frames):
    found = []
    for path, line in frames:
        tags = read_tags(path)
        if tags and line in tags[1]:
            found.append(tags[1][line])
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stats")
    ap.add_argument("elf")
    ap.add_argument("--lines", action="store_true", help="also print per source line")
    ap.add_argument("--cores", action="store_true", help="also print per core")
    a = ap.parse_args()

    d = json.load(open(a.stats))
    F = d["pc_fields"]
    threads = d["config"]["threads"]
    levels = [lv for lv in ("l1", "l2", "l3", "dram") if d.get(lv, {}).get("by_pc")]
    pcs = sorted({int(pc, 16) for c in d["cores"] for pc in c["pcs"]}
                 | {int(pc, 16) for lv in levels for pc in d[lv]["by_pc"]} - {0})
    frames = symbolize(a.elf, pcs)

    def add(acc, row):
        for i, f in enumerate(F):
            acc[f] += row[i]

    steps = defaultdict(lambda: defaultdict(int))
    lines = defaultdict(lambda: defaultdict(int))
    per_core = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    ev_pcs = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # event -> pc -> [commits, lanes]
    for c in d["cores"]:
        for pcs_hex, row in c["pcs"].items():
            pc = int(pcs_hex, 16)
            fr = frames.get(pc, [])
            s = step_of(fr)
            add(steps[s], row)
            add(per_core[c["core"]][s], row)
            if fr:
                add(lines["%s:%d" % (os.path.basename(fr[0][0]), fr[0][1])], row)
            for e in events_of(fr):
                ev_pcs[e][pc][0] += row[F.index("commits")]
                ev_pcs[e][pc][1] += row[F.index("lanes")]

    def table(title, groups):
        total = sum(g["issue"] + g["scrb"] + g["head"] + g["fetch"] + g["wait"] + g["idle"] for g in groups.values())
        print(title)
        print("%-14s %8s %6s %6s %6s %6s %6s %6s %6s %10s %6s %8s" % (
            "", "wcycles", "%", "issue%", "scrb%", "head%", "fetch%", "wait%", "idle%", "instrs", "simt%", "lat/ins"))
        for k, g in sorted(groups.items(), key=lambda kv: -(kv[1]["issue"] + kv[1]["scrb"] + kv[1]["head"] + kv[1]["fetch"] + kv[1]["wait"] + kv[1]["idle"])):
            w = g["issue"] + g["scrb"] + g["head"] + g["fetch"] + g["wait"] + g["idle"]
            if w == 0 and g["commits"] == 0:
                continue
            pct = lambda x: 100.0 * x / w if w else 0.0
            print("%-14s %8d %6.1f %6.1f %6.1f %6.1f %6.1f %6.1f %6.1f %10d %6.1f %8.1f" % (
                k[:14], w, 100.0 * w / total if total else 0, pct(g["issue"]), pct(g["scrb"]),
                pct(g["head"]), pct(g["fetch"]), pct(g["wait"]), pct(g["idle"]), g["commits"],
                100.0 * g["lanes"] / (g["issue"] * threads) if g["issue"] else 0,
                g["latency"] / g["commits"] if g["commits"] else 0))
        print()

    table("steps (all cores)", steps)
    if a.cores:
        for core in sorted(per_core):
            table("core %d" % core, per_core[core])
    if a.lines:
        table("source lines (innermost frame)", lines)
    mem_tables(d, levels, frames)
    if ev_pcs:
        print("events (warp executions = most-run PC of the line; lanes = its active lanes)")
        for e, rows in sorted(ev_pcs.items()):
            commits, lanes = max(rows.values())
            print("  %-20s %10d warp  %10d lanes" % (e, commits, lanes))


def mem_tables(d, levels, frames):
    """Per level: requests in from above and out to below, by step and by tag.
    Flow = [rd, wr, wr_bytes, amo]. Write-backs and runtime traffic carry no PC."""
    for lv in levels:
        by_step = defaultdict(lambda: [[0] * 4, [0] * 4])
        for pc_hex, f in d[lv]["by_pc"].items():
            pc = int(pc_hex, 16)
            s = "(no pc)" if pc == 0 else step_of(frames.get(pc, []))
            for side, key in ((0, "in"), (1, "out")):
                for i in range(4):
                    by_step[s][side][i] += f[key][i]
        print("%s requests by step: in from above (rd wr amo, wr bytes) | out to below (rd wr amo, wr bytes)" % lv)
        for s, (fi, fo) in sorted(by_step.items(), key=lambda kv: -sum(kv[1][1][:2])):
            print("  %-14s %8d %8d %8d %10d | %8d %8d %8d %10d" % (
                s[:14], fi[0], fi[1], fi[3], fi[2], fo[0], fo[1], fo[3], fo[2]))
        print("%s requests by tag" % lv)
        for t, f in d[lv]["by_tag"].items():
            fi, fo = f["in"], f["out"]
            if any(fi) or any(fo):
                print("  %-14s %8d %8d %8d %10d | %8d %8d %8d %10d" % (
                    t[:14], fi[0], fi[1], fi[3], fi[2], fo[0], fo[1], fo[3], fo[2]))
        print()


if __name__ == "__main__":
    sys.exit(main())
