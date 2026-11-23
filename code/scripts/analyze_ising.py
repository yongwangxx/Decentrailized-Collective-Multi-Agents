"""Ising single-particle tests: readouts from the records written by run_ising.py. No model is called.

For every query the first attempt with a readable answer is used.

    letters       follow: queries choosing the neighbour majority in the cells without a tie (k != 2)
                  keep at tie: queries keeping the own state at k = 2
                  identical prompts (random_tie): how often the answer is the letter A, for each of the four texts
    temperature   majority chosen at h != 0; at h = 0, answers +1 split by the own spin;
                  P(+1) for every temperature and local field next to the heat-bath value 1 / (1 + exp(-2h/T))
    with the tool queries that called the tool; consistent decisions: answer +1 exactly when the last number returned
                  is below the heat-bath probability of +1

Usage: python scripts/analyze_ising.py [directory with ising_<group>_<model>.jsonl, default output/ising]
"""
import _path  # noqa: F401
import glob
import os
import sys
from collections import defaultdict

from particle_agents import ising
from particle_agents.runner import first_valid, load
from particle_agents.stats import frac, wilson


def interval(k, n):
    lo, hi = wilson(k, n)
    return f"{frac(k, n)} [95%: {100 * lo:.1f}-{100 * hi:.1f}%]" if n else "-"


def letters(group, R):
    no_tie = [r for r in R if r["k"] != 2]
    tie = [r for r in R if r["k"] == 2 and r["mode"] == "shuffled"]
    majority = lambda r: 1 if r["k"] > 2 else -1
    print(f"  {group}: {len(R)} answers")
    if no_tie:
        print(f"    follow the majority (no tie): {interval(sum(r['state'] == majority(r) for r in no_tie), len(no_tie))}")
    if tie:
        print(f"    keep own state at the tie:    {interval(sum(r['state'] == r['own'] for r in tie), len(tie))}")
    ident = [r for r in R if r["mode"] == "identical"]
    if ident:
        print("    identical prompts, answer A (own state, letters -> count):")
        for own in (1, -1):
            for labels in ("AB", "BA"):
                g = [r for r in ident if r["own"] == own and r["labels"] == labels]
                shown = labels[0] if own == 1 else labels[1]
                print(f"      own {own:+d} shown as {shown}: A in {sum(r['answer'] == 'A' for r in g)}/{len(g)}")


def temperature(group, R):
    print(f"  {group}: {len(R)} answers")
    nt = [r for r in R if r["h"] != 0]
    if nt:
        exp = sum(max(ising.p_plus(r["h"], r["T"]), 1 - ising.p_plus(r["h"], r["T"])) for r in nt) / len(nt)
        print(f"    majority chosen at h != 0: {interval(sum(r['state'] == (1 if r['h'] > 0 else -1) for r in nt), len(nt))}; heat-bath expectation {100 * exp:.1f}%")
    for own in (1, -1):
        g = [r for r in R if r["h"] == 0 and r["own"] == own]
        if g:
            print(f"    h = 0, own spin {own:+d}: +1 in {frac(sum(r['state'] == 1 for r in g), len(g))}")
    if group == "temperature_tool":
        used = [r for r in R if any(c.get("valid") for c in r["calls"])]
        last = lambda r: [c for c in r["calls"] if c.get("valid")][-1]["ret"]
        ok = sum((r["state"] == 1) == (last(r) < ising.p_plus(r["h"], r["T"])) for r in used)
        print(f"    called the tool: {frac(len(used), len(R))}; decisions consistent with 'number < p_+ gives +1': {frac(ok, len(used))}")
    print("    P(+1) measured / heat bath, by temperature (rows) and local field h = -4, -2, 0, +2, +4 (columns):")
    for T in ising.TEMPERATURES:
        cells = []
        for h in (-4, -2, 0, 2, 4):
            g = [r for r in R if r["T"] == T and r["h"] == h]
            cells.append((f"{sum(r['state'] == 1 for r in g) / len(g):.2f}" if g else "  - ") + f"/{ising.p_plus(h, T):.2f}")
        print(f"      T = {T:<5}" + "  ".join(cells))


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join("output", "ising")
    files = defaultdict(dict)
    for f in sorted(glob.glob(os.path.join(d, "ising_*.jsonl"))):
        name = os.path.basename(f)[len("ising_"):-len(".jsonl")]
        g = next(g for g in sorted(ising.GROUPS, key=len, reverse=True) if name.startswith(g + "_"))
        files[name[len(g) + 1:]][g] = f
    if not files:
        sys.exit(f"no ising_*.jsonl files in {d}")
    for model, groups in files.items():
        print(f"== {model}")
        for g in ising.GROUPS:
            if g not in groups:
                continue
            A = load(groups[g])
            R = first_valid(A)
            print(f"  [{g}] attempts {len(A)}, queries with an answer {len(R)}, attempts without a valid answer {len(A) - len(R)}")
            (temperature if g.startswith("temperature") else letters)(g, R)


if __name__ == "__main__":
    main()
