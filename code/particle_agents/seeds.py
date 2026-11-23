"""All randomization of the experiments derives from one base seed that you choose.

    PA_SEED     environment variable, any string (default "1")

Every random choice (letter assignment, order of neighbours or teammates, lane names, initial states, update order,
numbers drawn for the die) uses a generator of its own, random.Random("<PA_SEED>|<purpose>|<indices>"). The same base
seed therefore reproduces the same sequence of prompts, initial states and update orders, independent of the order or
parallelism in which the queries are sent; a different base seed gives an independent realization.
"""
import os
import random

BASE = os.environ.get("PA_SEED", "1")


def key(*parts):
    """The seed string for a purpose and its indices."""
    return "|".join(str(p) for p in (BASE,) + parts)


def rng(*parts):
    return random.Random(key(*parts))
