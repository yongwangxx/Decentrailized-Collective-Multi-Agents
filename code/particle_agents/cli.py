"""Command-line options shared by the run scripts."""
import argparse
import os

from .backends import get_backend


def parser(description):
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", default="mock", help="mock (offline, default), openai (OpenAI-compatible interface) or <module>:<Class> (your own)")
    p.add_argument("--out", default=None, help="output file or directory (default: under ./output/)")
    p.add_argument("--parallel", type=int, default=8, help="number of queries sent in parallel")
    p.add_argument("--max-attempts", type=int, default=10,
                   help="how often a query without a valid answer is asked before the run stops (1 = never ask again)")
    p.add_argument("--system-prompt", default=None,
                   help="system prompt sent before the prompt (default: none, except where a script says otherwise)")
    p.add_argument("--dry-run", action="store_true", help="print the number of queries of the run and stop")
    return p


def backend_of(args):
    b = get_backend(args.backend)
    return b, getattr(b, "model", args.backend.split(":")[-1])


def default_out(name):
    return os.path.join("output", name)


def spread(items, n):
    """n items spread evenly over a list (all of them if n is None or not smaller than the list)."""
    if not n or n >= len(items):
        return list(items)
    return [items[(2 * i + 1) * len(items) // (2 * n)] for i in range(n)]
