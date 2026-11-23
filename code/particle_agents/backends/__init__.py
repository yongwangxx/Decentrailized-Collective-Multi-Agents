"""Backends: how one query is sent to a model.

Every backend has the same interface,

    reply = backend.ask(prompt, tool=None, dice=None, system=None, max_turns=4)

and returns a dict with

    text        the final reply of the model (str)
    reasoning   list of reasoning texts, one per request, where the interface returns them (never used for the answer)
    calls       list of tool-call records (see particle_agents.tools), in order
    error       None, or a short string if the query failed (network error, turn limit, ...)
    n_requests  number of requests to the model within this query (a tool call adds one)
    model       model identifier as reported by the interface, if any

One call of `ask` is one particle update: a fresh conversation with no history.

    --backend mock                 offline, rule-based answers; costs nothing (backends/mock.py)
    --backend openai               any OpenAI-compatible chat-completions interface (backends/openai_compat.py)
    --backend <module>:<Class>     your own backend: any importable class with the interface above and an attribute
                                   `model`; backends/template.py is a starting point
"""
import importlib

from .base import DEFAULT_SYSTEM


def get_backend(name):
    if name == "mock":
        from .mock import MockBackend
        return MockBackend()
    if name == "openai":
        from .openai_compat import OpenAICompatBackend
        return OpenAICompatBackend()
    if ":" in name:
        module, cls = name.split(":", 1)
        return getattr(importlib.import_module(module), cls)()
    raise SystemExit(f"unknown backend {name!r}; use mock, openai or <module>:<Class>")
