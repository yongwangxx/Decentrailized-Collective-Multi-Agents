"""Tools that can be offered to a particle, and the random sources behind them.

Three tools are used in the experiments:

    role die      `draw(weight_A, weight_B, weight_C)`: returns one of the roles A, B, C with probability
                  proportional to the weights (three-role problem)
    option die    `draw(weight_<option>, ...)`: the same for the moves of a bird; every weight is optional and a
                  weight that is left out counts as 0 (three lanes)
    random number `random_uniform()`: returns a uniform random number in [0, 1) with four decimals
                  (Ising temperature test)

A tool is a plain description (name, description, JSON schema) plus a pure function
`handle(args, u) -> (record, reply_text)`, where `u` is a uniform random number supplied by the program.
The model never supplies the randomness; it only supplies the weights.

The random number of a call comes from a `Dice` object:

    Dice(seed, "indexed")   call number i (from 0) uses random.Random(f"{seed}|{i}").random()
    Dice(seed, "stream")    one stream random.Random(seed); call number i takes the i-th number of the stream
    Dice(None, "system")    random.random(), not reproducible; the number is recorded with the call
"""
import math
import random

ROLE_DRAW_DESCRIPTION = (
    "Draws one of the roles A, B, C at random. The probability of each role is its weight divided by the sum of "
    "the three weights, so a larger weight makes that role more likely and a weight of 0 means it is never drawn. "
    "Weights must be non-negative and at least one must be positive. Returns the role drawn.")
ROLE_DRAW_ERROR = "Error: give three non-negative numbers, not all zero. No role was drawn."
OPTION_DRAW_ERROR = "Error: give non-negative numbers, at least one of them positive. Nothing was drawn."
RANDOM_UNIFORM_DESCRIPTION = "Returns a random real number drawn uniformly from the interval [0, 1)."


class Dice:
    """Source of the uniform random numbers used by a tool within one query."""

    def __init__(self, seed=None, mode="system"):
        assert mode in ("indexed", "stream", "system")
        assert (seed is None) == (mode == "system"), "a seed is required for 'indexed' and 'stream', and only for them"
        self.seed, self.mode, self.count = seed, mode, 0
        self._stream = random.Random(seed) if mode == "stream" else None

    def next(self):
        i = self.count
        self.count += 1
        if self.mode == "indexed":
            return random.Random(f"{self.seed}|{i}").random()
        if self.mode == "stream":
            return self._stream.random()
        return random.random()




def weighted_choice(options, weights, u, fallback):
    """Go through the options in order, add up their shares, and return the first whose cumulative share exceeds u.
    `fallback` is returned only if rounding leaves the last cumulative share at or below u (u within rounding of 1)."""
    total = sum(weights)
    c = 0.0
    for o, w in zip(options, weights):
        c += w / total
        if c > u:
            return o
    return fallback


class RoleDraw:
    """The die of the three-role problem. All three weights are required."""
    kind = "role_draw"
    name = "draw"
    description = ROLE_DRAW_DESCRIPTION
    options = ["A", "B", "C"]

    def parameters(self):
        return {"type": "object",
                "properties": {f"weight_{c}": {"type": "number", "minimum": 0, "description": f"Weight for role {c}"} for c in "ABC"},
                "required": ["weight_A", "weight_B", "weight_C"], "additionalProperties": False}

    def handle(self, args, u):
        """Returns (record, reply). record: raw_args, u, valid and, if valid, weights (by letter) and ret (the letter drawn)."""
        rec = dict(raw_args=args, u=u, valid=False)
        try:
            raw = [args[f"weight_{c}"] for c in "ABC"]
            if any(isinstance(x, bool) for x in raw):
                raise ValueError
            w = [float(x) for x in raw]
            if any((not math.isfinite(x)) or x < 0 for x in w) or sum(w) <= 0:
                raise ValueError
            ret = weighted_choice("ABC", w, u, fallback="C")
            rec.update(valid=True, weights=dict(zip("ABC", w)), ret=ret)
            return rec, f"Role drawn: {ret}"
        except Exception:
            return rec, ROLE_DRAW_ERROR



class OptionDraw:
    """The die of the three-lane problem. `options` are the moves open to this bird, e.g. ["A", "B", "Stay"]."""
    kind = "option_draw"
    name = "draw"

    def __init__(self, options):
        self.options = list(options)
        self.description = (
            "Draws one of the options " + ", ".join(self.options) + " at random. The probability of each option is its "
            "weight divided by the sum of all the weights, so a larger weight makes that option more likely. An option "
            "whose weight is left out or is 0 is never drawn. Weights must be non-negative and at least one must be "
            "positive. Returns the option drawn.")

    def parameters(self):
        return {"type": "object",
                "properties": {f"weight_{x}": {"type": "number", "minimum": 0, "description": f"Weight for {x}"} for x in self.options},
                "required": [], "additionalProperties": False}

    def handle(self, args, u):
        """Returns (record, reply). record: raw_args, options, u, valid and, if valid, p (normalized positive weights) and ret."""
        rec = dict(raw_args=args, options=self.options, u=u, valid=False)
        try:
            if (not isinstance(args, dict) or set(args) - {f"weight_{o}" for o in self.options}
                    or any(isinstance(x, bool) for x in args.values())):
                raise ValueError
            w = [float(args.get(f"weight_{o}") or 0) for o in self.options]
            if any((not math.isfinite(x)) or x < 0 for x in w) or sum(w) <= 0:
                raise ValueError
            ret = weighted_choice(self.options, w, u, fallback=[o for o, x in zip(self.options, w) if x > 0][-1])
            rec.update(valid=True, ret=ret, p={o: x / sum(w) for o, x in zip(self.options, w) if x > 0})
            return rec, f"Option drawn: {ret}"
        except Exception:
            return rec, OPTION_DRAW_ERROR



class RandomUniform:
    """The random-number tool of the Ising temperature test. It takes no arguments."""
    kind = "random_uniform"
    name = "random_uniform"
    description = RANDOM_UNIFORM_DESCRIPTION
    options = []

    def parameters(self):
        return {"type": "object", "properties": {}, "required": []}

    def handle(self, args, u):
        r = round(u, 4)
        return dict(u=u, valid=True, ret=r), str(r)



def openai_spec(tool):
    """The tool in the function-calling format of an OpenAI-compatible chat interface."""
    return [{"type": "function", "function": {"name": tool.name, "description": tool.description, "parameters": tool.parameters()}}]
