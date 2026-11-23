"""Three-role problem (antiferromagnetic three-state Potts model on the triangular lattice): prompts, situations,
single-particle query tables, answer parsing and readouts.

A particle has one of three roles, written 0, 1, 2 inside the program, and six teammates. In every query the program
replaces the roles by the letters A, B, C with a permutation `perm` (perm[r] is the letter of role r) and lists the
six teammates in a shuffled order, so that a preference for a letter or a position cannot select a physical role.

Conditions (labels as in the paper)

    greedy    prescribed: "Choose the role that gives you the fewest conflicts."
    T0.4      prescribed: heat bath at T = 0.4 with a random number written into the prompt
    T0.7      the same at T = 0.7
    Tc0.5     prescribed: heat bath at T = 0.5 only if the particle has a conflict
    0         goal and die ("goal and tool")
    H         + hint that greedy choices can leave the team stuck
    F         + facts about the whole team
    I         + "a member who has a conflict sometimes has to take a role that is not the best" (only in combinations)
    S         + members choose in turn
    HF, HFI, HS, SF, HSF, HSFI   combinations

Local-optimum variants (situation: teammates 3-2-1, own role on the 1)
    base      the goal-and-die prompt, further queries
    nodie     the same prompt without the sentence on the tool, and no tool
    explain   the goal-and-die prompt with the last sentence replaced by a request to give the reason first

Seeds (see particle_agents/seeds.py)
    teammates     seeds.rng("roles", id).shuffle; for the heat-bath conditions the random number
                  randrange(10000) / 10000 comes next from the same generator
    letters       the six permutations are used in turn: PERMS[(i + start) % 6], with
                  start = seeds.rng("roles-letters", label, situation).randrange(6) in the main test and 0 in the variants
    die           call number c of an attempt uses random.Random(seeds.key("roles-die", id, attempt) + "|" + c).random()
"""
import itertools
import math
import re

from . import seeds
from .tools import Dice, RoleDraw

ORDINALS = ["First", "Second", "Third", "Fourth", "Fifth", "Sixth"]
RULE = "Rule: each teammate who has the same role as you counts as one conflict."
GOAL = " The team's goal is that nobody has any conflict."
SEES = "Each member sees only their own six teammates."
HINT = " It is known that if every member always takes the role that gives them the fewest conflicts, the team can end up with conflicts that never go away."
TURNS = "Members choose in turn, not all at the same time: your teammates' roles will not change while you choose."
FACTS = "The whole team is much larger than the seven of you, each member has a different set of six teammates, and an assignment of roles in which nobody has any conflict exists."
INSTRUCTION = "To get the team out of such a stuck state, a member who has a conflict sometimes has to take a role that is not the best for them at that moment."
TOOL = "A tool named draw is available: given a weight for each of A, B and C, it returns one role drawn at random with probability proportional to the weights. You may use it or not."
CHOOSE = "Now choose your role."
END_PLAIN = 'No need to explain. On the last line write only "Answer:" followed by A, B or C, nothing else.'
END_TOOL = 'No need to explain. End your final reply with a line that contains only "Answer:" followed by A, B or C.'
END_EXPLAIN = 'First say in one or two sentences what you see and why you choose as you do. Then end your final reply with a line that contains only "Answer:" followed by A, B or C.'

CONDITIONS = ["greedy", "T0.4", "T0.7", "Tc0.5", "0", "H", "F", "HF", "HFI", "S", "HS", "SF", "HSF", "HSFI"]

# The 16 situations: (number of teammates in roles 0, 1, 2; own role). Index 14 is the local optimum.
SITUATIONS = [((6, 0, 0), 0), ((6, 0, 0), 1), ((5, 1, 0), 0), ((5, 1, 0), 1), ((5, 1, 0), 2), ((4, 2, 0), 0), ((4, 2, 0), 1),
              ((4, 2, 0), 2), ((4, 1, 1), 0), ((4, 1, 1), 1), ((3, 3, 0), 0), ((3, 3, 0), 2), ((3, 2, 1), 0), ((3, 2, 1), 1),
              ((3, 2, 1), 2), ((2, 2, 2), 0)]
LOCAL_OPTIMUM = 14
PERMS = ["".join(p) for p in itertools.permutations("ABC")]


def kind(label):
    """'die' for the conditions with goal and die; 'greedy', 'heat' (heat bath) or 'heat_conflict' (heat bath only at
    conflicts) for the prescribed rules."""
    if label == "greedy":
        return "greedy"
    if label.startswith("Tc"):
        return "heat_conflict"
    if label.startswith("T"):
        return "heat"
    return "die"


def temperature(label):
    return float(label.lstrip("Tc")) if kind(label).startswith("heat") else None


def head(q):
    L = q["perm"]
    return (["You are one member of a team. Each member takes one of three roles: A, B or C.",
             f"Your role right now is: {L[q['own']]}",
             "You work directly with six teammates. Their roles right now are:"]
            + [f"{o}: {L[r]}" for o, r in zip(ORDINALS, q["mates"])])


def prompt(q):
    """The prompt of a query. q: condition, perm, own, mates and, for heat-bath conditions, T and r;
    optional variant ("nodie" or "explain")."""
    c = q["condition"]
    k = kind(c)
    if k == "greedy":
        body = [RULE + " Choose the role that gives you the fewest conflicts.", CHOOSE, END_PLAIN]
    elif k == "heat":
        body = [RULE,
                f"The team works at temperature T = {q['T']}. Choose each role X with probability proportional to exp(-n_X / T), where n_X is the number of your six teammates who have role X.",
                f"Here is a random number drawn uniformly from [0, 1): {q['r']}. Use it to make the choice: go through the roles in the order A, B, C and take the first one whose cumulative probability exceeds the random number.",
                CHOOSE, END_PLAIN]
    elif k == "heat_conflict":
        body = [RULE,
                "If none of your six teammates has your current role, you have no conflict: keep that role and do not use the random number below.",
                f"Otherwise, choose each role X (your current role included) with probability proportional to exp(-n_X / T) with T = {q['T']}, where n_X is the number of your six teammates who have role X.",
                f"Here is a random number drawn uniformly from [0, 1): {q['r']:.4f}. Use it only if you have at least one conflict: go through the roles in the order A, B, C and take the first one whose cumulative probability exceeds the random number.",
                CHOOSE, END_PLAIN]
    else:
        add = "" if c == "0" else c
        assert set(add) <= set("HSFI"), c
        variant = q.get("variant")
        body = [RULE + GOAL, SEES + (HINT if "H" in add else "")]
        body += [x for a, x in (("S", TURNS), ("F", FACTS), ("I", INSTRUCTION)) if a in add]      # always in this order
        body += ([] if variant == "nodie" else [TOOL]) + [CHOOSE, END_EXPLAIN if variant == "explain" else END_TOOL]
    return "\n".join(head(q) + body)


def heat_bath_choice(n, perm, T, r):
    """The role prescribed by the heat-bath conditions: add up the shares exp(-n/T) in the order of the letters A, B, C
    and return the role on which the random number r falls."""
    w = {perm[k]: math.exp(-n[k] / T) for k in range(3)}
    Z = sum(w.values())
    c = 0
    for ch in "ABC":
        c += w[ch] / Z
        if c > r:
            return perm.index(ch)
    return perm.index("C")


def repetitions(label, s):
    """Number of queries of situation s in a condition. With the die, the local optimum, the two conflict-free situations
    4-2-0 and 3-3-0 and the two ties get more queries than the other situations."""
    if kind(label) != "die":
        return 10
    n, own = SITUATIONS[s]
    if n[own] == 0:
        return 96 if sorted(n) in ([0, 2, 4], [0, 3, 3]) else 8
    if s == LOCAL_OPTIMUM:
        return 96
    return 32 if n[own] == min(n) else 8


def make_query(label, s, i):
    """Query number i of situation s in a condition of the main test."""
    n, own = SITUATIONS[s]
    id = f"{label}|{s}|{i}"
    rg = seeds.rng("roles", id)
    mates = [r for r in range(3) for _ in range(n[r])]
    rg.shuffle(mates)
    perm = PERMS[(i + seeds.rng("roles-letters", label, s).randrange(6)) % 6]
    k = kind(label)
    q = dict(id=id, condition=label, sit=s, n=list(n), own=own, mates=mates, perm=perm,
             T=temperature(label), r=rg.randrange(10000) / 10000 if k.startswith("heat") else None)
    if k == "heat":
        q["prescribed"] = heat_bath_choice(n, perm, q["T"], q["r"])
    if k == "heat_conflict":
        q["prescribed"] = own if n[own] == 0 else heat_bath_choice(n, perm, q["T"], q["r"])
    return q


def queries(labels):
    """The main single-particle test for the given condition labels."""
    return [make_query(c, s, i) for c in labels for s in range(len(SITUATIONS)) for i in range(repetitions(c, s))]


def local_optimum_query(variant, i):
    """Query number i of a local-optimum variant ("base", "nodie" or "explain")."""
    n, own = SITUATIONS[LOCAL_OPTIMUM]
    id = f"localopt-{variant}|{i}"
    rg = seeds.rng("roles", id)
    mates = [r for r in range(3) for _ in range(n[r])]
    rg.shuffle(mates)
    q = dict(id=id, condition="0", sit=LOCAL_OPTIMUM, n=list(n), own=own, mates=mates,
             perm=PERMS[i % 6])
    if variant != "base":
        q["variant"] = variant
    return q


def tool_for(q):
    return RoleDraw() if kind(q["condition"]) == "die" and q.get("variant") != "nodie" else None


def dice_for(q, attempt=1):
    """Random source of the die for one attempt at a query (see the module docstring)."""
    if tool_for(q) is None:
        return None
    return Dice(seeds.key("roles-die", q["id"], attempt), "indexed")


# \uFF1A is the full-width colon, \u3002 the ideographic full stop; the quote characters stripped are " ' and their curly forms
_LAST_LINE = re.compile(r"(?:(?i:(?:final\s+)?answer))?\s*[:\uFF1A]?\s*(?:(?i:role)\s+)?([ABC])\s*[.\u3002]?")
_ANYWHERE = re.compile(r"(?i:answer)\s*[:\uFF1A]\s*([ABC])(?![A-Za-z])")


def parse(text, require_answer_word=False):
    """The letter answered, or None. If the last non-empty line as a whole is an answer ("Answer: B", "B", ": B",
    "Role B"), take it; otherwise take the last "Answer: X" anywhere in the text. Only capital letters count.
    With require_answer_word (used for the 'explain' variant, where the reply contains prose) the last line counts
    only if it contains the word "answer"."""
    lines = [l.strip().strip("*`\"'\u201C\u201D\u2018\u2019 ") for l in (text or "").strip().split("\n") if l.strip()]
    if lines and _LAST_LINE.fullmatch(lines[-1]) and (not require_answer_word or re.search(r"(?i)answer", lines[-1])):
        return _LAST_LINE.fullmatch(lines[-1]).group(1)
    m = _ANYWHERE.findall(text or "")
    return m[-1] if m else None


def situation_class(n, own):
    """noconf    no conflict (n[own] == 0)
    localopt  in conflict, own role is the unique role with the fewest conflicts
    tie       in conflict, own role shares the fewest conflicts with another role
    better    in conflict, some other role has strictly fewer conflicts"""
    m = min(n)
    if n[own] == 0:
        return "noconf"
    if n[own] > m:
        return "better"
    return "localopt" if list(n).count(m) == 1 else "tie"


def calls_by_role(calls, perm):
    """Add to each valid die record the weights and the result in terms of roles: w_role[r], ret_role."""
    for c in calls:
        if c.get("valid") and "weights" in c:
            c["w_role"] = [c["weights"][perm[r]] for r in range(3)]
            c["ret_role"] = perm.index(c["ret"])
    return calls


def weighted_roll(calls, n, own, new):
    """Criterion of the paper for 'the answer was made on a weighted roll of the die', for a particle at a local optimum.
    Uses the last valid roll of the query. True if that roll gave staying a positive weight, gave at least two roles a
    positive weight, gave weights that do not increase with the number of conflicts (staying >= the role with two
    conflicts >= the role with three), and the answer `new` is the role rolled. A departure on a weighted roll is
    weighted_roll(...) and new != own. Only tool calls and the answer are used, not the reply text."""
    valid = [c for c in calls if c.get("valid") and c.get("w_role") is not None]
    if not valid:
        return False
    w = valid[-1]["w_role"]
    others = sorted((r for r in range(3) if r != own), key=lambda r: n[r])
    a, b, c = w[own], w[others[0]], w[others[1]]
    return a > 0 and sum(t > 0 for t in (a, b, c)) >= 2 and a >= b >= c and new == valid[-1]["ret_role"]
