"""Particle agents: one stateless language-model query per particle update.

Modules
    tools      the tools offered to a particle (weighted die, uniform random number) and their random sources
    ising      prompts and query tables of the Ising single-particle tests
    roles      prompts and query tables of the three-role (three-colouring) single-particle tests
    lattice    the 9 x 9 periodic triangular lattice: geometry, initial states, update order, queries
    lanes      counterflow in three lanes: prompts, lane naming, initial states, move resolution
    stats      Wilson intervals
    backends   how a query is sent to a model (OpenAI-compatible interface, offline mock, or your own)
"""
__version__ = "1.0"
