"""Reference simulation of the three-role lattice with a two-parameter rule. No language model is called.

Same lattice and update order as the agent runs: 9 x 9 periodic triangular lattice, three sublattices (i - j) mod 3 per
round in a random order, each sublattice in parallel, random initial states.
Let n_X be the number of neighbours of a particle in role X and s its own role (n_s = its number of conflicts).

    n_s = 0                               with probability pc take one of the other two roles at random, else keep s
    0 < n_s < n_X for both other roles    (local optimum) with probability ph take the other role with the fewest
                                          conflicts (at random if tied), else keep s
    otherwise                             take a role with the fewest conflicts, at random among ties

The success fraction S(pc, ph) is the fraction of runs that reach zero conflicts within 40 rounds.
The rule is a reference with two parameters, not a model of the agents.

    python scripts/sim_reference.py                      the full map over the grid PC x PH below, --runs runs per point,
                                                         seed --seed + 1000 * i + j for the i-th ph and j-th pc -> output/reference_map.json
    python scripts/sim_reference.py --point 0.0 0.22     S at one point (pc, ph); --runs and --seed may be given
    python scripts/sim_reference.py --keep-ties          variant for pc = 0: a particle keeps its role whenever it is among
                                                         the roles with the fewest conflicts (seed --seed + 7000 + i)
    python scripts/sim_reference.py --runs 50            a faster, coarser map
Requires numpy.
"""
import argparse
import json
import os

import numpy as np

L = 9
N = L * L
PC = [0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
PH = [0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.7, 1.0]

_idx = np.arange(N).reshape(L, L)
# neighbours (i+-1, j), (i, j+-1), (i-1, j+1), (i+1, j-1)
NB = np.stack([np.roll(_idx, s, a) for s, a in ((1, 0), (-1, 0), (1, 1), (-1, 1))]
              + [np.roll(np.roll(_idx, 1, 0), -1, 1), np.roll(np.roll(_idx, -1, 0), 1, 1)], -1).reshape(N, 6)
_I, _J = np.indices((L, L))
SUB = [np.where(((_I - _J) % 3).reshape(-1) == k)[0] for k in range(3)]


def conflicts(S):
    return (S[:, :, None] == S[:, NB]).sum((1, 2)) // 2


def run(pc, ph, n=400, rounds=40, seed=0, tie="random"):
    """n independent runs. Returns for each run the round at which zero conflicts were first reached, or -1."""
    rng = np.random.default_rng(seed)
    S = rng.integers(3, size=(n, N))
    t = np.full(n, -1)
    for r in range(rounds):
        for k in rng.permutation(3):
            s = SUB[k]
            cnt = (S[:, NB[s]][:, :, :, None] == np.arange(3)).sum(2)          # neighbours in each role
            own = S[:, s]
            n_own = np.take_along_axis(cnt, own[:, :, None], 2)[:, :, 0]
            mn = cnt.min(2, keepdims=True)
            is_best = cnt == mn
            u = rng.random(own.shape)
            w = rng.random(cnt.shape)                                          # random keys for uniform choices
            pick_other = np.argmin(w + (np.arange(3) == own[:, :, None]) * 9, 2)
            pick_best = np.argmin(w + (~is_best) * 9, 2)
            if tie == "keep":
                pick_best = np.where(np.take_along_axis(is_best, own[:, :, None], 2)[:, :, 0], own, pick_best)
            second = np.where(is_best, 99, cnt).min(2, keepdims=True)
            pick_second = np.argmin(w + (cnt != second) * 9, 2)
            local_opt = (n_own == mn[:, :, 0]) & (is_best.sum(2) == 1) & (n_own > 0)
            new_conflict = np.where((u < ph) & local_opt, pick_second, pick_best)
            S[:, s] = np.where(n_own == 0, np.where(u < pc, pick_other, own), new_conflict)
        c = conflicts(S)
        t = np.where((t < 0) & (c == 0), r + 1, t)
    return t


def success(t):
    return float((t >= 0).mean())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--point", type=float, nargs=2, metavar=("PC", "PH"))
    p.add_argument("--runs", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--keep-ties", action="store_true")
    p.add_argument("--out", default=os.path.join("output", "reference_map.json"))
    a = p.parse_args()
    if a.point:
        t = run(a.point[0], a.point[1], n=a.runs, seed=a.seed, tie="keep" if a.keep_ties else "random")
        print(f"pc = {a.point[0]}, ph = {a.point[1]}, {a.runs} runs: zero conflicts within 40 rounds in a fraction {success(t):.3f}")
        return
    out = dict(pc=PC, ph=PH, runs=a.runs)
    if a.keep_ties:
        T = [run(0, ph, n=a.runs, seed=a.seed + 7000 + i, tie="keep") for i, ph in enumerate(PH)]
        out.update(pc=[0], tie="keep", within40=[[success(t)] for t in T])
    else:
        out.update(within40=[])
        for i, ph in enumerate(PH):
            T = [run(pc, ph, n=a.runs, seed=a.seed + 1000 * i + j) for j, pc in enumerate(PC)]
            out["within40"].append([success(t) for t in T])
            print(f"ph = {ph}: within 40 rounds, for pc = {PC}: " + " ".join(f"{x:.2f}" for x in out["within40"][-1]), flush=True)
    path = a.out if not a.keep_ties or a.out != os.path.join("output", "reference_map.json") else os.path.join("output", "reference_map_keep_ties.json")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f)
    if a.keep_ties:
        print("keep ties, pc = 0, within 40 rounds for ph =", PH, ":", " ".join(f"{x[0]:.2f}" for x in out["within40"]))
    print("written:", path)


if __name__ == "__main__":
    main()
