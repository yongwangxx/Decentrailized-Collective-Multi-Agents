"""Counterflow in three lanes: readouts from the records written by run_lanes_group.py and run_lanes_repeat.py.
No model is called.

Collective runs (first --rounds rounds, default 100), per run:
    moves per round; rounds in which the lanes are sorted (no lane holds both directions), the first such round and
    the round from which they stay sorted; lane order after the last round (fraction of birds moving in the majority
    direction of their lane)
    head-on encounters (pairs in the same lane facing each other, adjacent or one empty cell apart), counted per pair
    and round: how many were resolved in the round (at least one of the two moved), separately for symmetric
    encounters (the two birds see identical cells) and the others
    a bird in a head-on encounter that sees in two successive rounds exactly the same cells from the same place:
    how often it repeats its answer
    where the die is rolled: rolls by situation of the bird (head-on with nothing else in view / head-on with other
    birds in view / own lane clear two cells ahead / blocked)
Repeated queries, per model: answers (stay / straight / change) for each situation with fixed and with renamed lanes.

Usage: python scripts/analyze_lanes.py [directory with lanes_*.jsonl, default output/lanes] [--rounds 100]
"""
import _path  # noqa: F401
import argparse
import glob
import os
from collections import Counter

from particle_agents import lanes
from particle_agents.runner import first_valid, load


def view_key(birds, i):
    return frozenset((y, k, s) for (y, k), s in lanes.view_of(birds, i).items())


def run_stats(R):
    T = len(R)
    moved = [r["moved"] for r in R]
    srt = [r["sorted"] for r in R]
    enc, rep, use = Counter(), Counter(), Counter()
    prev = None
    for t, r in enumerate(R):
        b = [(tuple(pos), h) for pos, h in r["before"]]
        n = len(b)
        V = [view_key(b, i) for i in range(n)]
        occ = {pos: i for i, (pos, h) in enumerate(b)}
        in_pair = {}
        for i, j, gap in lanes.encounters(b):
            key = "symmetric" if V[i] == V[j] else "asymmetric"
            enc[key, "n"] += 1
            enc[key, "resolved"] += bool(r["ok"][i] or r["ok"][j])
            alone = len(V[i]) == 1 and len(V[j]) == 1
            for k in (i, j):
                in_pair[k] = alone
        if prev:
            pb, pV, pr = prev
            for i in range(n):
                if i in in_pair and pb[i] == b[i] and pV[i] == V[i]:
                    rep[pr["answer_lane"][i] == r["answer_lane"][i]] += 1
        for i in range(n):
            (x, y), h = b[i]
            ahead = [occ.get(((x + h * k) % lanes.LENGTH, y)) for k in (1, 2)]
            if i in in_pair:
                cat = "head-on, nothing else in view" if in_pair[i] else "head-on, others in view"
            else:
                cat = "own lane clear" if all(a is None for a in ahead) else "blocked"
            use[cat, "n"] += 1
            use[cat, "rolls"] += bool(r["rolls"][i])
        prev = (b, V, r)
    last = [(tuple(pos), h) for pos, h in R[-1]["after"]]
    return dict(T=T, per_round=sum(moved) / T, rounds_sorted=sum(srt),
                first_sorted=next((r["t"] for r in R if r["sorted"]), None),
                stays_sorted_from=next((R[t]["t"] for t in range(T) if all(srt[t:])), None),
                order=lanes.lane_order(last),
                rolls=sum(bool(c) for r in R for c in r["rolls"]), retries=sum(r["retries"] for r in R),
                enc=enc, rep=rep, use=use)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dir", nargs="?", default=os.path.join("output", "lanes"))
    p.add_argument("--rounds", type=int, default=100)
    a = p.parse_args()
    for f in sorted(glob.glob(os.path.join(a.dir, "lanes_*_seed*.jsonl"))):
        R = load(f)[:a.rounds]
        if not R:
            continue
        S = run_stats(R)
        e, rp, u = S["enc"], S["rep"], S["use"]
        print(f"== {os.path.basename(f)}: {S['T']} rounds")
        print(f"  moves per round {S['per_round']:.2f}; lanes sorted in {S['rounds_sorted']} rounds, first in round {S['first_sorted']}, "
              f"sorted from round {S['stays_sorted_from']} on; lane order at the end {S['order']:.2f}")
        print(f"  head-on encounters resolved in the round: symmetric {e['symmetric', 'resolved']}/{e['symmetric', 'n']}, "
              f"others {e['asymmetric', 'resolved']}/{e['asymmetric', 'n']}")
        print(f"  head-on encounter recurring unchanged, answer repeated: {rp[True]}/{rp[True] + rp[False]}")
        if R[0]["die"]:
            print(f"  rolls of the die: {S['rolls']} of {12 * S['T']} decisions; " + "; ".join(
                f"{c} {u[c, 'rolls']}/{u[c, 'n']}" for c in ("head-on, nothing else in view", "head-on, others in view", "own lane clear", "blocked")))
        print(f"  queries asked again: {S['retries']}")
    for f in sorted(glob.glob(os.path.join(a.dir, "lanes_repeat_*.jsonl"))):
        R = first_valid(load(f))
        print(f"== {os.path.basename(f)}: {len(R)} answers")
        for s in lanes.REPEAT_SITUATIONS:
            for g in ("fixed", "relabel"):
                c = Counter(r["action"] for r in R if r["situation"] == s and r["group"] == g)
                print(f"  {s:9s} {g:7s} n = {sum(c.values()):2d}: " + ", ".join(f"{k} {c[k]}" for k in ("stay", "straight", "change")))


if __name__ == "__main__":
    main()
