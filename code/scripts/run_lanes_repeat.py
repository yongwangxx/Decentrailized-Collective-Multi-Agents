"""Counterflow in three lanes: repeated queries of a symmetric head-on situation, without the die.

A bird in an outer lane meets one other bird head-on, adjacent or one cell apart, with nothing else in view. Each of the
two situations is asked repeatedly with fixed lane names (the identical text every time) and once with each of the
namings (lane names x listing order).

Example
    python scripts/run_lanes_repeat.py --backend mock

Output: <out>/lanes_repeat_<model>.jsonl, one JSON object per attempt:
    id, situation (adjacent / one_apart), group (fixed / relabel), lane, view ([[lane, offset, "+"|"-"], ...]), naming,
    prompt, answer (as written), action (stay / straight / change), text, reasoning, error, model, attempt
"""
import _path  # noqa: F401
import os

from particle_agents import cli, lanes
from particle_agents.runner import run_queries


def main():
    p = cli.parser(__doc__)
    p.add_argument("--sample", type=int, default=None, help="ask only this many of the queries, spread over the table")
    p.set_defaults(parallel=4)
    args = p.parse_args()
    Q = cli.spread(lanes.repeat_queries(), args.sample)
    if args.dry_run:
        print(len(Q), "queries")
        return
    backend, model = cli.backend_of(args)
    out = os.path.join(args.out or cli.default_out("lanes"), f"lanes_repeat_{model}.jsonl")

    def ask_one(q, attempt):
        text = lanes.prompt(q["lane"], q["view"], False, q["naming"])
        r = backend.ask(text, system=args.system_prompt)
        answer = None if r["error"] else lanes.parse_answer(q["lane"], r["text"], q["naming"])
        action = None if answer is None else "stay" if answer == "Stay" else "straight" if lanes.lane_of(answer, q["naming"]) == q["lane"] else "change"
        return dict(q, view=[[y, k, s] for (y, k), s in q["view"].items()], prompt=text, answer=answer, action=action, text=r["text"],
                    reasoning=r["reasoning"], error=r["error"], model=model)

    run_queries(Q, ask_one, out, args.parallel, args.max_attempts)


if __name__ == "__main__":
    main()
