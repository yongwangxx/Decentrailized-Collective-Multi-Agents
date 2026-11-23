"""Ising single-particle tests: ask the queries of one or more groups, one independent query per line of output.

Groups (see particle_agents/ising.py): rule, norule, tie_extra, switch_tie, random_tie, temperature, temperature_tool.
--dry-run prints the number of queries per group (temperature_tool needs one more request per query when the tool
is called).

Examples
    python scripts/run_ising.py --backend mock --sample 20
    python scripts/run_ising.py --backend openai --groups rule random_tie temperature temperature_tool

Output: <out>/ising_<group>_<model>.jsonl, one JSON object per attempt:
    id, group, own, k, h, neighbours, [labels, sentence, mode | T], prompt,
    answer (as written), state (+1 / -1), text, reasoning, calls, n_requests, error, model, attempt
"""
import _path  # noqa: F401
import os

from particle_agents import cli, ising
from particle_agents.runner import run_queries


def main():
    p = cli.parser(__doc__)
    p.add_argument("--groups", nargs="+", default=ising.GROUPS, choices=ising.GROUPS)
    p.add_argument("--sample", type=int, default=None, help="ask only this many queries per group, spread over the table")
    p.add_argument("--max-turns", type=int, default=5, help="requests per query when the tool is offered")
    args = p.parse_args()
    if args.dry_run:
        for g in args.groups:
            print(g, len(cli.spread(ising.queries(g), args.sample)), "queries")
        return
    backend, model = cli.backend_of(args)
    out_dir = args.out or cli.default_out("ising")

    def ask_one(q, attempt):
        text = ising.prompt(q)
        r = backend.ask(text, tool=ising.tool_for(q), dice=ising.dice_for(q, attempt), system=args.system_prompt, max_turns=args.max_turns)
        answer, state = (None, None) if r["error"] else ising.parse(q, r["text"])
        return dict(q, prompt=text, answer=answer, state=state, text=r["text"], reasoning=r["reasoning"], calls=r["calls"],
                    n_requests=r["n_requests"], error=r["error"], model=model)

    for g in args.groups:
        Q = cli.spread(ising.queries(g), args.sample)
        run_queries(Q, ask_one, os.path.join(out_dir, f"ising_{g}_{model}.jsonl"), args.parallel, args.max_attempts, label=f"[{g}] ")


if __name__ == "__main__":
    main()
