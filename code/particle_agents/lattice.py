"""The 9 x 9 periodic triangular lattice of the three-role problem: geometry, initial states, update order, queries.

    site k = i * L + j; its six neighbours are (i-1, j), (i+1, j), (i, j-1), (i, j+1), (i-1, j+1), (i+1, j-1), modulo L
    sublattice of a site = (i - j) mod 3; three sublattices of 27 sites, no two sites of one sublattice are neighbours
    conflicts = number of neighbouring pairs with the same role = energy of the antiferromagnetic three-state Potts model

One round updates the three sublattices one after another in a random order; the 27 particles of a sublattice are
queried in parallel on the same state. Every query uses the goal-and-die prompt of particle_agents.roles with a freshly
drawn letter assignment and teammate order.

Seeds (see particle_agents/seeds.py; s = index of the initial state, t = round counted from 0, k = site)
    initial state      seeds.rng("lattice-init", s), randrange(3) for the 81 sites in order
    sublattice order   seeds.rng("lattice-order", s, t).sample(range(3), 3)
    query              rg = seeds.rng("lattice-query", s, t, k); rg.shuffle(teammates); letters = "".join(rg.sample("ABC", 3))
    die                call number c of a query uses random.Random(seeds.key("lattice-die", s, t, k) + "|" + c).random();
                       a query that is asked again appends "-rej<i>" (re-ask option) or "-retry<i>" (failed call) to
                       the part before "|"
None of these depends on the model, so with the same base seed all models start from the same states, follow the same
sublattice orders and, in the same situation, see the same prompt.
"""
from . import seeds
from .tools import Dice

L = 9
N = L * L
ROUNDS = 40             # a run stops after 40 rounds ...
EXTRA_ROUNDS = 3        # ... or three rounds after the first conflict-free state


def _neighbours(k):
    i, j = divmod(k, L)
    return [((i + di) % L) * L + (j + dj) % L for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1))]


NEIGHBOURS = [_neighbours(k) for k in range(N)]
SUBLATTICES = [[k for k in range(N) if (k // L - k % L) % 3 == s] for s in range(3)]


def initial_state(s):
    rg = seeds.rng("lattice-init", s)
    return [rg.randrange(3) for _ in range(N)]


def sublattice_order(s, t):
    return seeds.rng("lattice-order", s, t).sample(range(3), 3)


def conflicts(state):
    return sum(state[k] == state[n] for k in range(N) for n in NEIGHBOURS[k]) // 2


def make_query(state, s, t, k):
    """The query of site k in round t of the run from initial state s, given the current state."""
    rg = seeds.rng("lattice-query", s, t, k)
    mates = [state[n] for n in NEIGHBOURS[k]]
    rg.shuffle(mates)
    perm = "".join(rg.sample("ABC", 3))
    return dict(condition="0", own=state[k], mates=mates, n=[mates.count(c) for c in range(3)], perm=perm)


def dice_seed(s, t, k, suffix=""):
    return seeds.key("lattice-die", s, t, k) + suffix


def dice_for(s, t, k, suffix=""):
    return Dice(dice_seed(s, t, k, suffix), "indexed")


def leaves_best_role(n, own, new):
    """True if the answer moves to a role that does not have the fewest conflicts, or leaves a role that is the only
    one with the fewest conflicts. A change between roles tied for the fewest conflicts is not counted."""
    best = [r for r in range(3) if n[r] == min(n)]
    return new != own and (new not in best or (own in best and len(best) == 1))


def check_geometry():
    for k in range(N):
        assert len(set(NEIGHBOURS[k])) == 6 and k not in NEIGHBOURS[k]
        assert all(k in NEIGHBOURS[n] for n in NEIGHBOURS[k])
    for s in range(3):
        assert len(SUBLATTICES[s]) == 27
        assert all(not set(NEIGHBOURS[k]) & set(SUBLATTICES[s]) for k in SUBLATTICES[s])
