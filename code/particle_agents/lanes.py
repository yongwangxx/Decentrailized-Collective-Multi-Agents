"""Counterflow in three lanes: prompts, lane naming, initial states, move resolution, encounters.

Track: a ring of LENGTH cells with three parallel lanes 0, 1, 2 (lane 1 in the middle); a cell holds at most one
particle ("bird"). A bird is ((x, y), h): cell x, lane y, direction h = +1 or -1. It may move one cell ahead in its own
lane, one cell ahead into a neighbouring lane, or stay. All birds move at once; a move into an occupied cell, two
moves into the same cell and two moves that would cross do not happen.

What a bird is shown: the cells with offsets -2 ... +2 in all three lanes (14 cells besides its own), offsets counted
along its own direction, each cell empty or holding a bird that faces the + side (same direction) or the - side.
`view` is a dict {(lane, offset): "+" or "-"}; cells that are not in the dict are empty.

Lane naming. In every query the three lanes get the names A, B, C in one of 6 assignments and the lanes are listed
in one of 6 orders (36 combinations). `naming = (names, order)`: names[y] is the name of lane y, order is the sequence
of lanes in which the cells are listed. naming = None means names A, B, C in order.

Seeds (see particle_agents/seeds.py; `seed` is the run seed given on the command line)
    random initial state         seeds.rng("lanes-init-random", seed).sample(all 120 cells, 12); the first six face +1
    evenly spaced initial state  seeds.rng("lanes-init-even", seed), see initial_state_even
    naming of a query            r = seeds.rng("lanes-naming", seed, t, i) for bird i in round t (from 1);
                                 names: r.shuffle(["A","B","C"]); then order: r.shuffle([0,1,2])
    die                          random.random() (not seeded); the number drawn is recorded with every roll
"""
import re

from . import seeds
from .tools import Dice, OptionDraw

LENGTH, LANES, PER_DIRECTION, SIGHT = 40, 3, 6, 2

L1 = "You are a bird on a track of square cells with three lanes, {a}, {b} and {c}, that run side by side: lane {m} lies between lane {e1} and lane {e2}. A cell holds at most one bird."
L2 = ("Cells are named by lane and by offset along the track from your own cell, which is {own} 0. Offset +1 is the next cell on the side you are facing, "
      "offset -1 is the next cell on the opposite side, and so on.")
L3 = "A bird can stay where it is or move one cell toward the side it faces, ending in its own lane or in a lane next to its own. Your moves are named by the lane they end in:"
OWN = "The wind you feel right now blows toward the + side."
FACE = "A bird faces the way the wind it feels blows. Each bird sees only the cells around itself and chooses its own move."
SEE = "You can see the cells with offsets from -{v} to +{v} in all three lanes. Right now:"
EMPTY = "{c}: empty"
BIRD = "{c}: a bird facing the {s} side"
PHYSICS = "A move into a cell that is occupied does not happen. If two birds try to move into the same cell, neither of them moves. If the moves of two birds would cross each other, neither of them moves."
CLASH = "Rule: a move that does not happen counts as one clash for the bird that tried it."
GOAL = "The birds' shared goal is that every bird travels as smoothly as it can along the wind it feels, with as few clashes as possible."
TOOL = ("A tool named draw is available: give it a weight for any of {o} and Stay (an option you leave out gets weight 0), and it returns one of them "
        "drawn at random with probability proportional to the weights. You may use it or not.")
CHOOSE = "Now choose your move."
END = 'No need to explain. End your final reply with a line that contains only "Answer:" followed by {o}.'

DEFAULT_NAMING = (["A", "B", "C"], [0, 1, 2])


# ---------------------------------------------------------------- naming and prompt
def _nm(naming):
    return DEFAULT_NAMING if naming is None else naming


def naming_for_query(seed, t, i):
    """Lane names and listing order of the query of bird i in round t, from a random stream of its own (independent of
    initial state and die)."""
    r = seeds.rng("lanes-naming", seed, t, i)
    names = list("ABC")
    r.shuffle(names)
    order = [0, 1, 2]
    r.shuffle(order)
    return names, order


def cells(lane, naming=None):
    """The 14 cells shown, in the order in which they are listed."""
    return [(y, k) for y in _nm(naming)[1] for k in (2, 1, 0, -1, -2) if not (y == lane and k == 0)]


def cell_name(c, naming=None):
    return f"{_nm(naming)[0][c[0]]} {c[1]:+d}".replace("+0", "0")


def moves(lane, naming=None):
    """Names of the lanes this bird can move into, in listing order."""
    return [_nm(naming)[0][y] for y in _nm(naming)[1] if abs(y - lane) <= 1]


def options(lane, naming=None):
    return moves(lane, naming) + ["Stay"]


def _or_list(xs):
    return ", ".join(xs[:-1]) + " or " + xs[-1]


def prompt(lane, view, die, naming=None):
    """The prompt of a bird in lane `lane` that sees `view`. die=True adds the one sentence that offers the tool."""
    names, order = _nm(naming)
    assert all(c in cells(lane, naming) for c in view)
    listed = [names[y] for y in order]
    edge = [names[y] for y in order if y != 1]
    m = moves(lane, naming)
    lines = [L1.format(a=listed[0], b=listed[1], c=listed[2], m=names[1], e1=edge[0], e2=edge[1]), L2.format(own=names[lane]), L3]
    lines += [f"{x}: to {x} +1" for x in m]
    lines += [OWN, FACE, SEE.format(v=SIGHT)]
    lines += [EMPTY.format(c=cell_name(c, naming)) if c not in view else BIRD.format(c=cell_name(c, naming), s=view[c]) for c in cells(lane, naming)]
    lines += [PHYSICS, CLASH + " " + GOAL]
    if die:
        lines.append(TOOL.format(o=", ".join(m)))
    lines += [CHOOSE, END.format(o=_or_list(options(lane, naming)))]
    return "\n".join(lines)


def parse_answer(lane, text, naming=None):
    """The option answered (a lane name or "Stay"), or None. The last "Answer: X" in the reply counts, and X must be one
    of the options open to this bird."""
    m = re.findall(r"Answer:\s*\**\s*(Stay|[ABC])\b", text or "")
    return m[-1] if m and m[-1] in options(lane, naming) else None


def lane_of(answer, naming=None):
    """Lane named by an answer; None for "Stay"."""
    return None if answer == "Stay" else _nm(naming)[0].index(answer)


def tool_for(lane, naming=None):
    return OptionDraw(options(lane, naming))


def dice():
    return Dice()


# ---------------------------------------------------------------- track
def initial_state_random(seed):
    """12 of the 120 cells drawn at random; the first six birds face +1, the other six -1."""
    r = seeds.rng("lanes-init-random", seed)
    picked = r.sample([(x, y) for x in range(LENGTH) for y in range(LANES)], 2 * PER_DIRECTION)
    return [(c, 1 if k < PER_DIRECTION else -1) for k, c in enumerate(picked)]


def initial_state_even(seed):
    """The ring is divided into 12 equal segments with one bird in each (the whole pattern shifted by a random offset);
    every lane holds four birds, two for each direction; which bird sits in which segment is drawn at random."""
    r = seeds.rng("lanes-init-even", seed)
    n = 2 * PER_DIRECTION
    off = r.randrange(LENGTH)
    xs = [(off + round(k * LENGTH / n)) % LENGTH for k in range(n)]
    slots = []
    for h in (1, -1):
        lanes = [y for y in range(LANES) for _ in range(PER_DIRECTION // LANES)]
        r.shuffle(lanes)
        slots += [(y, h) for y in lanes]
    r.shuffle(slots)
    birds = [((x, y), h) for x, (y, h) in zip(xs, slots)]
    return sorted(birds, key=lambda b: -b[1])       # birds facing +1 first, as in the random initial state


def view_of(birds, i):
    """What bird i sees: {(lane, offset): "+" | "-"}."""
    (x, y), h = birds[i]
    occ = {p: hh for p, hh in birds}
    v = {}
    for yy in range(LANES):
        for k in (2, 1, 0, -1, -2):
            if yy == y and k == 0:
                continue
            hh = occ.get(((x + h * k) % LENGTH, yy))
            if hh is not None:
                v[(yy, k)] = "+" if hh == h else "-"
    return v


def crosses(p1, h1, c1, p2, h2, c2):
    """Do the moves p1 -> c1 and p2 -> c2 cross? (Two moves into the same cell and a move into an occupied cell are
    handled separately.)"""
    if h1 == h2:
        return p1[0] == p2[0] and (p1[1] - p2[1]) * (c1[1] - c2[1]) < 0
    return c1[0] == p2[0] and c2[0] == p1[0] and (p1[1] - c2[1]) * (c1[1] - p2[1]) < 0


def resolve(birds, targets):
    """targets[i] = cell that bird i tries to enter, or None for staying. Returns for every bird whether its move happens."""
    occ = {p for p, _ in birds}
    count = {}
    for c in targets:
        if c:
            count[c] = count.get(c, 0) + 1
    ok = [c is not None and c not in occ and count[c] == 1 for c in targets]
    blocked = set()
    for i in range(len(birds)):
        for j in range(i + 1, len(birds)):
            if (targets[i] and targets[j] and targets[i] not in occ and targets[j] not in occ
                    and crosses(birds[i][0], birds[i][1], targets[i], birds[j][0], birds[j][1], targets[j])):
                blocked.update((i, j))
    return [o and k not in blocked for k, o in enumerate(ok)]


def target_cell(bird, lane):
    """Cell entered by a move that ends in `lane` (None = stay)."""
    (x, _), h = bird
    return None if lane is None else ((x + h) % LENGTH, lane)


def lanes_sorted(birds):
    """True if no lane holds birds of both directions."""
    return all(len({h for (p, h) in birds if p[1] == y}) <= 1 for y in range(LANES))


def lane_order(birds):
    """Fraction of birds that move in the majority direction of their own lane (1 = lanes sorted)."""
    return sum(max(sum(1 for (p, h) in birds if p[1] == y and h == s) for s in (1, -1)) for y in range(LANES)) / len(birds)


def encounters(birds):
    """Head-on encounters: pairs (i, j, gap) of birds in the same lane that face each other, adjacent (gap 1) or with one
    empty cell between (gap 2). i is the bird facing +1."""
    occ = {p: i for i, (p, h) in enumerate(birds)}
    out = []
    for i, ((x, y), h) in enumerate(birds):
        if h < 0:
            continue
        for gap in (1, 2):
            j = occ.get(((x + gap) % LENGTH, y))
            if j is None or birds[j][1] > 0 or (gap == 2 and ((x + 1) % LENGTH, y) in occ):
                continue
            out.append((i, j, gap))
    return out


# ---------------------------------------------------------------- repeated queries
REPEAT_SITUATIONS = {"adjacent": {(0, 1): "-"}, "one_apart": {(0, 2): "-"}}     # bird in outer lane 0, one bird head-on, nothing else in view


def all_namings():
    import itertools
    return [(list(n), list(o)) for n, o in itertools.product(itertools.permutations("ABC"), itertools.permutations(range(3)))]


def repeat_queries():
    """The repeated-query test: each situation repeatedly with fixed lane names and once with each naming, without the die."""
    Q = []
    for s, view in REPEAT_SITUATIONS.items():
        code = {"adjacent": "S1", "one_apart": "S2"}[s]
        Q += [dict(id=f"{code}-fixed-{k}", situation=s, group="fixed", lane=0, view=view, naming=None) for k in range(24)]
        Q += [dict(id=f"{code}-relabel-{k}", situation=s, group="relabel", lane=0, view=view, naming=nm) for k, nm in enumerate(all_namings())]
    return Q
