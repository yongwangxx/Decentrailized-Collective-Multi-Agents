"""Three-role problem, single particle: readouts from the records written by run_roles_single.py. No model is called.

For every query the first attempt with a readable answer is used.

Main test, one line per condition and model:
    changed without a conflict   answers that change role, over the conflict-free situations with a unique best role
    non-minimal with a conflict  answers that take a role without the fewest conflicts, over the situations with a
                                 conflict and a unique best role
    local optimum left           answers that leave the role at the local optimum (teammates 3-2-1, own role on the 1)
    rolled                       local-optimum queries with at least one valid roll of the die
    left on a weighted roll      local-optimum queries that leave on a weighted roll (criterion: roles.weighted_roll)
    prescribed rules             answers equal to the prescribed role
    mean die weights             at the local optimum: normalized weights of the last valid roll as
                                 stay : role with two conflicts : role with three; a query without a roll counts as 1 : 0 : 0
Local-optimum variants: queries, left, rolled, left on a weighted roll, with a 95% Wilson interval for "left".

Usage: python scripts/analyze_roles_single.py [directory with roles_*.jsonl, default output/roles]
"""
import _path  # noqa: F401
import glob
import os
import sys

from particle_agents import roles
from particle_agents.runner import first_valid, load
from particle_agents.stats import frac, wilson


def unique_best(r):
    n = sorted(r["n"])
    return n[0] < n[1]


def last_roll(r):
    v = [c for c in r["calls"] if c.get("valid") and c.get("w_role") is not None]
    return v[-1] if v else None


def mean_weights(R):
    """Mean normalized weights (stay, role with two conflicts, role with three) over local-optimum queries."""
    acc = [0.0, 0.0, 0.0]
    for r in R:
        c = last_roll(r)
        if c is None:
            w = [1.0, 0.0, 0.0]
        else:
            others = sorted((x for x in range(3) if x != r["own"]), key=lambda x: r["n"][x])
            tot = sum(c["w_role"])
            w = [c["w_role"][r["own"]] / tot, c["w_role"][others[0]] / tot, c["w_role"][others[1]] / tot]
        acc = [a + x for a, x in zip(acc, w)]
    return [a / len(R) for a in acc] if R else None


def local_optimum_line(R):
    left = sum(r["role"] != r["own"] for r in R)
    lo, hi = wilson(left, len(R))
    rolled = sum(1 for r in R if last_roll(r))
    weighted = sum(roles.weighted_roll(r["calls"], r["n"], r["own"], r["role"]) and r["role"] != r["own"] for r in R)
    return f"left {frac(left, len(R))} [95%: {100 * lo:.2f}-{100 * hi:.2f}%], rolled {rolled}, left on a weighted roll {weighted}"


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join("output", "roles")
    files = sorted(glob.glob(os.path.join(d, "roles_*.jsonl")))
    if not files:
        sys.exit(f"no roles_*.jsonl files in {d}")
    for f in files:
        A = load(f)
        R = first_valid(A)
        print(f"== {os.path.basename(f)}: attempts {len(A)}, queries with an answer {len(R)}")
        if "local_optimum" in os.path.basename(f):
            print("  " + local_optimum_line(R))
            w = mean_weights(R)
            if w and any(last_roll(r) for r in R):
                print(f"  mean die weights stay : two conflicts : three conflicts = {w[0]:.3f} : {w[1]:.3f} : {w[2]:.3f}")
            continue
        for label in roles.CONDITIONS:
            V = [r for r in R if r["condition"] == label]
            if not V:
                continue
            own_conf = lambda r: r["n"][r["own"]]
            nc = [r for r in V if unique_best(r) and own_conf(r) == 0]
            hc = [r for r in V if unique_best(r) and own_conf(r) > 0]
            lo = [r for r in V if r["sit"] == roles.LOCAL_OPTIMUM]
            line = (f"  {label:6s} changed without a conflict {sum(r['role'] != r['own'] for r in nc)}/{len(nc)}; "
                    f"non-minimal with a conflict {sum(r['n'][r['role']] != min(r['n']) for r in hc)}/{len(hc)}; "
                    f"local optimum left {sum(r['role'] != r['own'] for r in lo)}/{len(lo)}")
            if roles.kind(label) == "die":
                weighted = sum(roles.weighted_roll(r["calls"], r["n"], r["own"], r["role"]) and r["role"] != r["own"] for r in lo)
                line += f"; rolled {sum(1 for r in lo if last_roll(r))}; left on a weighted roll {weighted}"
                w = mean_weights(lo)
                if w:
                    line += f"; mean weights {w[0]:.2f}:{w[1]:.2f}:{w[2]:.2f}"
            pres = [r for r in V if r.get("correct") is not None]
            if pres:
                line += f"; equal to the prescribed role {sum(r['correct'] for r in pres)}/{len(pres)}"
            print(line)


if __name__ == "__main__":
    main()
