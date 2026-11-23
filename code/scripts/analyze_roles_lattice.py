"""Three-role lattice runs: readouts from the records written by run_roles_lattice.py. No model is called.

For each run directory and initial state:
    conflicts after every round, and the round of the first conflict-free state
    updates by situation of the particle (noconf / localopt / tie / better), how many changed role, how many rolled
    departures from a local optimum, and how many of them were made on a weighted roll of the die
        (criterion: roles.weighted_roll, which uses only the tool calls and the answer)
    moves that raised the particle's own number of conflicts
    mean die weights at the local optimum (stay : role with two conflicts : role with three; no roll counts as 1 : 0 : 0)
In runs with --reask-nonbest the first answer to each question is the one counted for the departures (the answer that
was not applied if there is one), while the conflicts follow the answers that were applied.
Only rounds 1 ... --rounds (default 40) are counted.

Usage: python scripts/analyze_roles_lattice.py <run directory> [<run directory> ...] [--rounds 40]
"""
import _path  # noqa: F401
import argparse
import glob
import os
from collections import Counter

from particle_agents import roles
from particle_agents.runner import load
from analyze_roles_single import last_roll, mean_weights


def first_answer(r):
    """(role answered first, its die calls): the first answer that was not applied if there is one, else the applied one."""
    if r.get("rejected"):
        return r["rejected"][0]["new"], r["rejected"][0]["calls"]
    return r["new"], r["calls"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dirs", nargs="+")
    p.add_argument("--rounds", type=int, default=40)
    a = p.parse_args()
    for d in a.dirs:
        rounds = load(os.path.join(d, "rounds.jsonl"))
        for f in sorted(glob.glob(os.path.join(d, "s[0-9]*.jsonl"))):
            R = [r for r in load(f) if r["t"] < a.rounds]
            if not R:
                continue
            s = R[0]["init"]
            conf = [x["conflicts"] for x in rounds if x["init"] == s and x["round"] <= a.rounds]
            zero = next((i + 1 for i, c in enumerate(conf) if c == 0), None)
            print(f"== {d}, initial state {s}: {len(conf)} rounds, {len(R)} updates")
            print("  conflicts after each round: " + " ".join(map(str, conf)))
            print(f"  first conflict-free state: {'round ' + str(zero) if zero else 'not reached'}")
            n = Counter()
            for r in R:
                new, calls = first_answer(r)
                c = r["situation"]
                n[c, "n"] += 1
                n[c, "changed"] += new != r["own"]
                n[c, "rolled"] += any(x.get("valid") for x in calls)
                if c == "localopt":
                    n["weighted"] += roles.weighted_roll(calls, r["n"], r["own"], new) and new != r["own"]
                n["raised"] += r["n"][new] > r["n"][r["own"]]
            for c in ("noconf", "localopt", "tie", "better"):
                print(f"  {c:9s} updates {n[c, 'n']:5d}   changed role {n[c, 'changed']:5d}   rolled the die {n[c, 'rolled']:5d}")
            print(f"  departures from a local optimum on a weighted roll: {n['weighted']} of {n['localopt', 'n']} updates at a local optimum")
            print(f"  moves that raised the particle's own conflicts: {n['raised']}")
            lo = [dict(r, calls=first_answer(r)[1]) for r in R if r["situation"] == "localopt"]
            w = mean_weights(lo)
            if w and any(last_roll(r) for r in lo):
                print(f"  mean die weights at the local optimum, stay : two conflicts : three conflicts = {w[0]:.3f} : {w[1]:.3f} : {w[2]:.3f}")
            if any("n_rejected" in r for r in R):
                print(f"  answers not applied and asked again: {sum(r.get('n_rejected', 0) for r in R)}; questions where all answers were not applied: {sum(bool(r.get('all_rejected')) for r in R)}")


if __name__ == "__main__":
    main()
