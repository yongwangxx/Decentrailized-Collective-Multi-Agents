"""Ising single-particle tests: prompts, query tables, answer parsing.

A spin with four neighbours. Physical states are +1 and -1; `k` is the number of neighbours in state +1 and
`h = 2k - 4` the local field.

Tests with letters (the two states are shown as A and B)
    rule         scoring rule stated; every cell (own state x k), `reps` queries each
    norule       the same prompt without the rule sentence; every cell
    tie_extra    scoring rule, further queries at the tie (k = 2)
    switch_tie   rule + "if both choices give the same score, switch ..."; every cell
    random_tie   rule + "if both choices give the same score, choose one at random":
                 randomized prompts at k = 1, 2, 3 and identical prompts at the tie
                 (own state x letter assignment, four texts)
    In the randomized prompts the program draws, for every query, which letter denotes +1 (`labels`: "AB" means
    A = +1) and the order of the four neighbours. In the identical prompts both are fixed.

Temperature tests (the spins are shown as +1 and -1, neighbours in state +1 listed first, identical wording per cell)
    temperature        the agent is told that it is a spin at temperature T under heat-bath dynamics; no random source
    temperature_tool   the same prompt; the tool `random_uniform` is offered (the prompt does not mention it)
    for each temperature in TEMPERATURES, every cell with `reps` identical queries

Seeds (see particle_agents/seeds.py). For a randomized query, seeds.rng("ising", id) gives first the neighbour order
(shuffle) and then the letter assignment (choice). The random numbers of the tool come from one stream per query and
attempt, seeds.rng("ising-tool", id, attempt), rounded to four decimals.
"""
import math
import re

from . import seeds
from .tools import Dice, RandomUniform

RULE = ("Scoring rule: among the other four, for each one whose state is the same as the state you choose now, you gain 1 point; "
        "for each one that differs, you lose 1 point. Please choose the state that gives you the highest score.")
TIE_RANDOM = " If both choices give the same score, choose one at random."
TIE_SWITCH = " If both choices give the same score, switch to the state different from your last one."
RULE_TEXT = {"rule": RULE, "norule": None, "random": RULE + TIE_RANDOM, "switch": RULE + TIE_SWITCH}
TEMPERATURES = [0.5, 1.5, 2.27, 4, 10]
GROUPS = ["rule", "norule", "tie_extra", "switch_tie", "random_tie", "temperature", "temperature_tool"]

# \uFF1A is the full-width colon; \u201C \u201D are curly quotes; \u2212 \u2013 are the minus sign and the en dash
ANSWER_LETTER = re.compile(r"Answer\s*[:\uFF1A]\s*\**\s*([AB])\s*\**\s*\.?\s*$", re.I)
ANSWER_SPIN = re.compile(r"Answer\s*[:\uFF1A]\s*[\"'\u201C\u201D*]*\s*([+\-\u2212\u2013]\s*1)\s*[\"'\u201C\u201D*]*\s*\.?\s*$", re.I)


def _sign(s):
    return "+1" if s == 1 else "-1"


def prompt_letters(q):
    """Prompt of the tests with letters. q: own, neighbours (list of +1/-1 in the order shown), labels, sentence."""
    letter = lambda s: q["labels"][0] if s == 1 else q["labels"][1]
    lines = ["There are five people here, each in one of two states, A or B.",
             f"Last time your state was: {letter(q['own'])}",
             "Last time the other four were in:"]
    lines += [f"{o}: {letter(s)}" for o, s in zip(["First", "Second", "Third", "Fourth"], q["neighbours"])]
    rule = RULE_TEXT[q["sentence"]]
    if rule:
        lines.append(rule)
    lines += ["Now please choose your state again.",
              'No need to explain. On the last line write only "Answer:" followed by A or B, nothing else.']
    return "\n".join(lines)


def prompt_temperature(q):
    """Prompt of the temperature tests. q: T, own, neighbours."""
    return "\n".join([
        f"You are one site of a two-dimensional Ising model with coupling J = 1 and no external field. The system is at temperature T = {q['T']} (in units of J/k_B) and evolves by heat-bath (Glauber) dynamics.",
        f"Your current spin is: {_sign(q['own'])}",
        "Your four nearest neighbours' spins are: " + ", ".join(_sign(s) for s in q["neighbours"]),
        "Perform one update of your spin according to the dynamics at this temperature.",
        'No need to explain. On the last line write only "Answer:" followed by +1 or -1, nothing else.'])


def prompt(q):
    return prompt_temperature(q) if q["group"].startswith("temperature") else prompt_letters(q)


def _shuffled(id, k):
    """Neighbour order and letter assignment of a randomized query (in this order from one generator)."""
    rg = seeds.rng("ising", id)
    nb = [1] * k + [-1] * (4 - k)
    rg.shuffle(nb)
    return nb, rg.choice(("AB", "BA"))


def queries(group, reps=32):
    """The query table of one group, in a fixed order. Each query is a dict with id, group, own, k, h, neighbours and,
    for the letter tests, labels, sentence and mode ("shuffled" or "identical")."""
    Q = []

    def add(id, own, k, sentence, fixed=None):
        nb, labels = fixed if fixed else _shuffled(id, k)
        Q.append(dict(id=id, group=group, own=own, k=k, h=2 * k - 4, neighbours=nb, labels=labels, sentence=sentence,
                      mode="identical" if fixed else "shuffled"))

    if group in ("rule", "norule"):
        for own in (1, -1):
            for k in range(5):
                for i in range(reps):
                    add(f"{group}|{own:+d}|{k}|{i}", own, k, group)
    elif group == "tie_extra":
        for own in (1, -1):
            for i in range(16):
                add(f"{group}|{own:+d}|2|{i}", own, 2, "rule")
    elif group == "switch_tie":
        for own in (1, -1):
            for k in range(5):
                for i in range(reps):
                    add(f"{group}|{own:+d}|{k}|{i}", own, k, "switch")
    elif group == "random_tie":
        for own in (1, -1):
            for k in (1, 2, 3):
                for i in range(32):
                    add(f"{group}|{own:+d}|{k}|{i}", own, k, "random")
            for labels in ("AB", "BA"):
                for i in range(32):
                    add(f"{group}|identical|{own:+d}|{labels}|{i}", own, 2, "random", fixed=([1, -1, 1, -1], labels))
    elif group in ("temperature", "temperature_tool"):
        for T in TEMPERATURES:
            for own in (1, -1):
                for k in range(5):
                    for i in range(reps):
                        Q.append(dict(id=f"{group}|T{T}|{own:+d}|{k}|{i}", group=group, T=T, own=own, k=k, h=2 * k - 4,
                                      neighbours=[1] * k + [-1] * (4 - k)))
    else:
        raise ValueError(group)
    return Q


def tool_for(q):
    return RandomUniform() if q["group"] == "temperature_tool" else None


def dice_for(q, attempt=1):
    """Random source of the tool: one stream per query and attempt."""
    if q["group"] != "temperature_tool":
        return None
    return Dice(seeds.key("ising-tool", q["id"], attempt), "stream")


def parse(q, text):
    """Returns (said, state): the answer as written and the physical state +1 / -1, or (None, None).
    The answer must be on the last line in the requested format."""
    text = (text or "").strip()
    if q["group"].startswith("temperature"):
        m = ANSWER_SPIN.search(text)
        if not m:
            return None, None
        s = 1 if m.group(1).replace(" ", "").startswith("+") else -1
        return _sign(s), s
    m = ANSWER_LETTER.search(text)
    if not m:
        return None, None
    a = m.group(1).upper()
    return a, (1 if a == q["labels"][0] else -1)


def p_plus(h, T):
    """Heat-bath probability of the state +1 at local field h and temperature T (J = k_B = 1)."""
    return 1.0 / (1.0 + math.exp(-2.0 * h / T))
