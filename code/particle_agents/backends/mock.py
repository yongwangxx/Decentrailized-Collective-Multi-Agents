"""Offline mock backend: answers every prompt by a fixed hand-written rule, without any network access.

It exists so that the whole pipeline (prompts, die, parsing, re-asking, records, analysis) can be run for free.
Its answers say nothing about any model. The rules, by prompt family:

    Ising, letters        take the state of the majority of the four neighbours; keep the own state at a tie
    Ising, temperature    without the tool: the majority, own state at a tie;
                          with the tool: call random_uniform once and answer +1 if the number is below
                          1 / (1 + exp(-2h/T)), else -1
    three roles           prescribed rules: follow them exactly (the heat-bath choice is computed from the prompt);
                          goal with the die: move to a role with fewer conflicts if there is one (first letter if tied);
                          at a local optimum roll the die with weights 3 : 1 : 0 (stay : role with two conflicts : role
                          with three) and take the role rolled; otherwise stay
                          goal without the die: the same, but stay at a local optimum
    three lanes           if a bird in the own lane is head-on within two cells: with the die, roll with equal weights
                          over the moves into empty cells and Stay, and take the option rolled; without the die, move
                          into the first listed empty neighbouring lane, else stay.
                          Otherwise move straight ahead if that cell is empty, else into the first listed empty
                          neighbouring lane, else stay.

Set PA_MOCK_BAD_REPLIES to a fraction (e.g. 0.2) to make that share of first replies unreadable, in order to see the
re-asking of the run scripts at work; the choice depends only on the prompt and on how often it was asked.
"""
import hashlib
import math
import os
import re
import threading

from .base import new_reply, run_tool


class MockBackend:
    name = "mock"
    model = "mock"

    def __init__(self):
        self.bad = float(os.environ.get("PA_MOCK_BAD_REPLIES", 0))
        self._seen = {}
        self._lock = threading.Lock()

    def ask(self, prompt, tool=None, dice=None, system=None, max_turns=4):
        out = new_reply()
        out["model"] = "mock"
        out["n_requests"] = 1
        with self._lock:
            count = self._seen[prompt] = self._seen.get(prompt, 0) + 1
        if self.bad and count == 1 and int(hashlib.sha256(prompt.encode()).hexdigest(), 16) % 1000 < 1000 * self.bad:
            out["text"] = "I am not sure."
            return out
        if prompt.startswith("There are five people"):
            answer = self._ising_letters(prompt)
        elif prompt.startswith("You are one site"):
            answer = self._ising_temperature(prompt, tool, dice, out)
        elif prompt.startswith("You are one member"):
            answer = self._roles(prompt, tool, dice, out)
        elif prompt.startswith("You are a bird"):
            answer = self._lanes(prompt, tool, dice, out)
        else:
            out["error"] = "mock: unknown prompt"
            return out
        out["text"] = f"Answer: {answer}"
        return out

    # ------------------------------------------------------------ Ising
    @staticmethod
    def _ising_letters(p):
        own = re.search(r"your state was: ([AB])", p).group(1)
        nb = re.findall(r"^(?:First|Second|Third|Fourth): ([AB])$", p, re.M)
        a, b = nb.count("A"), nb.count("B")
        return own if a == b else ("A" if a > b else "B")

    @staticmethod
    def _ising_temperature(p, tool, dice, out):
        T = float(re.search(r"T = ([0-9.]+) \(", p).group(1))
        own = int(re.search(r"current spin is: ([+-]1)", p).group(1))
        h = sum(int(x) for x in re.search(r"spins are: (.*)", p).group(1).split(", "))
        if tool is not None:
            r = float(run_tool(tool, dice, tool.name, {}, out["calls"]))
            out["n_requests"] += 1
            s = 1 if r < 1 / (1 + math.exp(-2 * h / T)) else -1
        else:
            s = own if h == 0 else (1 if h > 0 else -1)
        return "+1" if s == 1 else "-1"

    # ------------------------------------------------------------ three roles
    @staticmethod
    def _roles(p, tool, dice, out):
        own = re.search(r"Your role right now is: ([ABC])", p).group(1)
        mates = re.findall(r"^(?:First|Second|Third|Fourth|Fifth|Sixth): ([ABC])$", p, re.M)
        n = {c: mates.count(c) for c in "ABC"}
        m = re.search(r"exp\(-n_X / T\)(?:, where| with T = ([0-9.]+))", p)
        if m:                                                    # prescribed heat bath with the random number in the prompt
            T = float(m.group(1) or re.search(r"temperature T = ([0-9.]+)\.", p).group(1))
            r = float(re.search(r"uniformly from \[0, 1\): ([0-9.]+)\.", p).group(1))
            if "do not use the random number below" in p and n[own] == 0:
                return own
            w = {c: math.exp(-n[c] / T) for c in "ABC"}
            c = 0
            for ch in "ABC":
                c += w[ch] / sum(w.values())
                if c > r:
                    return ch
            return "C"
        best = min(n.values())
        if n[own] > best:                                        # a role with fewer conflicts exists
            return min(c for c in "ABC" if n[c] == best)
        local_optimum = n[own] > 0 and list(n.values()).count(best) == 1
        if local_optimum and tool is not None:
            others = sorted((c for c in "ABC" if c != own), key=lambda c: n[c])
            reply = run_tool(tool, dice, tool.name, {f"weight_{own}": 3, f"weight_{others[0]}": 1, f"weight_{others[1]}": 0}, out["calls"])
            out["n_requests"] += 1
            return reply.rsplit(" ", 1)[-1]
        return own

    # ------------------------------------------------------------ three lanes
    @staticmethod
    def _lanes(p, tool, dice, out):
        own = re.search(r"which is ([ABC]) 0\.", p).group(1)
        cell = dict(re.findall(r"^([ABC] [+-]?\d): (empty|a bird facing the [+-] side)$", p, re.M))
        mv = re.findall(r"^([ABC]): to [ABC] \+1$", p, re.M)
        free = [x for x in mv if cell[f"{x} +1"] == "empty"]
        head_on = "- side" in cell[f"{own} +1"] or (cell[f"{own} +1"] == "empty" and "- side" in cell[f"{own} +2"])
        if head_on and tool is not None:
            reply = run_tool(tool, dice, tool.name, {f"weight_{o}": 1 for o in free + ["Stay"]}, out["calls"])
            out["n_requests"] += 1
            return reply.rsplit(" ", 1)[-1]
        side = [x for x in free if x != own]
        if head_on:
            return side[0] if side else "Stay"
        return own if own in free else (side[0] if side else "Stay")
