"""Counterflow in three lanes: a collective run. 12 birds (six per direction) on a ring of 40 cells with three lanes.

Every round each bird gets one independent query (its 14 visible cells, freshly drawn lane names and listing order);
when all twelve answers are in, the moves are resolved together: a move into an occupied cell, two moves into the same
cell and two moves that would cross do not happen.
Arms: --die (the prompt offers the tool `draw`) or --no-die (that sentence is absent, no tool; otherwise identical);
initial state --init random or --init even. There is one query per bird and round. Give the arms with and without the die the same --seed and --init to start
them from the same initial state with the same lane names.
A query whose call fails or whose reply has no readable answer is kept in `attempt_errors` and asked again, up to
--max-attempts times; then the run stops. A stopped or finished run is resumed by running the same command again.

Examples
    python scripts/run_lanes_group.py --backend mock --seed 1 --init random --die --rounds 30
    python scripts/run_lanes_group.py --backend mock --seed 2 --init even --no-die --rounds 30

Output: <out>/lanes_<model>_<die|nodie>_seed<seed>_<init>.jsonl, one JSON object per round:
    t (round from 1), seed, init, die, model, before / after (lists of [[x, lane], direction]),
    naming (per bird: [names, order]), prompts, texts, answers (as written), answer_lane (lane entered or null for Stay),
    ok (did the move happen), moved (number of birds that moved), sorted (no lane holds both directions),
    rolls (per bird: the die calls), followed_die, retries, attempt_errors
"""
import _path  # noqa: F401
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

from particle_agents import cli, lanes
from particle_agents.runner import ask_until_valid


def ask_bird(backend, lane, view, die, naming, system, max_attempts):
    """One bird, one decision. Returns (record or None, errors of the failed attempts)."""
    text = lanes.prompt(lane, view, die, naming)

    def once(attempt):
        r = backend.ask(text, tool=lanes.tool_for(lane, naming) if die else None, dice=lanes.dice() if die else None, system=system)
        answer = None if r["error"] else lanes.parse_answer(lane, r["text"], naming)
        valid = [c for c in r["calls"] if c.get("valid")]
        return dict(prompt=text, answer=answer, text=r["text"], calls=r["calls"], error=r["error"], reasoning=r["reasoning"],
                    followed_die=None if (answer is None or not valid) else answer == valid[-1]["ret"])
    return ask_until_valid(once, max_attempts)


def main():
    p = cli.parser(__doc__)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--init", choices=["random", "even"], default="random")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--die", dest="die", action="store_true")
    g.add_argument("--no-die", dest="die", action="store_false")
    p.add_argument("--rounds", type=int, default=100)
    p.set_defaults(parallel=12)
    args = p.parse_args()
    if args.dry_run:
        print(2 * lanes.PER_DIRECTION * args.rounds, "queries")
        return
    backend, model = cli.backend_of(args)
    out = os.path.join(args.out or cli.default_out("lanes"), f"lanes_{model}_{'die' if args.die else 'nodie'}_seed{args.seed}_{args.init}.jsonl")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    birds = lanes.initial_state_even(args.seed) if args.init == "even" else lanes.initial_state_random(args.seed)
    t0 = 0
    if os.path.exists(out):
        with open(out) as f:
            for line in f:
                r = json.loads(line)
                birds = [(tuple(pos), h) for pos, h in r["after"]]
                t0 = r["t"]
    for t in range(t0 + 1, args.rounds + 1):
        # the naming stream depends only on (seed, round, bird): the arms with and without the die see the same names
        namings = [lanes.naming_for_query(args.seed, t, i) for i in range(len(birds))]
        views = [lanes.view_of(birds, i) for i in range(len(birds))]
        with ThreadPoolExecutor(args.parallel) as ex:
            res = list(ex.map(lambda i: ask_bird(backend, birds[i][0][1], views[i], args.die, namings[i], args.system_prompt, args.max_attempts),
                              range(len(birds))))
        if any(r is None for r, _ in res):
            bad = [i for i, (r, _) in enumerate(res) if r is None]
            print(f"STOPPED in round {t}: birds {bad} got no valid answer in {args.max_attempts} attempts: {[res[i][1][-1] for i in bad]}", flush=True)
            sys.exit(2)
        recs = [r for r, _ in res]
        answer_lane = [lanes.lane_of(r["answer"], nm) for r, nm in zip(recs, namings)]
        targets = [lanes.target_cell(b, y) for b, y in zip(birds, answer_lane)]
        ok = lanes.resolve(birds, targets)
        after = [(c if o else pos, h) for (pos, h), c, o in zip(birds, targets, ok)]
        rec = dict(t=t, seed=args.seed, init=args.init, die=args.die, model=model, before=birds, after=after,
                   naming=[list(nm) for nm in namings], prompts=[r["prompt"] for r in recs], texts=[r["text"] for r in recs],
                   answers=[r["answer"] for r in recs], answer_lane=answer_lane, ok=ok, moved=sum(ok), sorted=lanes.lanes_sorted(after),
                   rolls=[r["calls"] for r in recs], followed_die=[r["followed_die"] for r in recs],
                   retries=sum(len(e) for _, e in res), attempt_errors=[e for _, e in res if e])
        with open(out, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        birds = after
        print(f"round {t}: moved {rec['moved']}/12, rolls {sum(bool(c) for c in rec['rolls'])}, lanes sorted {rec['sorted']}, asked again {rec['retries']}", flush=True)
    print("written:", out)


if __name__ == "__main__":
    main()
