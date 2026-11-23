"""Three-role problem on the 9 x 9 periodic triangular lattice: 81 particles, each update one independent query.

Each round updates the three sublattices one after another in a random order; the 27 particles of a sublattice are
asked in parallel on the same state. Every query uses the goal-and-die prompt with fresh letters and teammate order, and by default the one-line system
prompt "Answer the question." (change it with --system-prompt).
The run stops after --rounds rounds (default 40) or three rounds after the first conflict-free state.
--init selects the initial state (a random assignment of roles; any index). There is one query per particle and round.

Answers
    default            every answer is applied. A reply without a readable answer leaves the role unchanged.
    --reask-nonbest    an answer that leaves the particle's best role (moves to a role that does not have the fewest
                       conflicts, or leaves the only role with the fewest conflicts) is recorded but not applied, and
                       the same question is asked again, at most --max-reask times (default 5); if every answer does so, the
                       role is kept.
A failed call (for example a network error) is repeated once; if it fails again the run stops and can be resumed.

Examples
    python scripts/run_roles_lattice.py --backend mock --init 1 --rounds 6
    python scripts/run_roles_lattice.py --backend openai --init 1 --reask-nonbest --parallel 27

Output in <out>/ (default output/lattice_<model>[_reask]/):
    s<init>.jsonl        one JSON object per update: init, t (round from 0), sub, k (site), nb (roles of the six
                         neighbours in lattice order), mates, n, own, perm, situation, prompt, answer, new (role after
                         the update), moved, is_min, unparsed, calls, followed_die, die_seed, text, reasoning,
                         n_requests, error, and with --reask-nonbest: rejected (the answers not applied), n_rejected,
                         all_rejected
    s<init>_state.json   states after every round, sublattice orders, progress (used for resuming)
    rounds.jsonl         one line per round: init, round (from 1), order, conflicts, state
"""
import _path  # noqa: F401
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

from particle_agents import cli, lattice, roles
from particle_agents.backends import DEFAULT_SYSTEM
from run_roles_single import make_record


class Stop(Exception):
    pass


def main():
    p = cli.parser(__doc__)
    p.add_argument("--init", type=int, nargs="+", default=[1], help="index of the initial state(s)")
    p.add_argument("--rounds", type=int, default=lattice.ROUNDS)
    p.add_argument("--reask-nonbest", action="store_true")
    p.add_argument("--max-reask", type=int, default=5)
    p.set_defaults(parallel=27, system_prompt=DEFAULT_SYSTEM)
    args = p.parse_args()
    if args.dry_run:
        print(lattice.N * args.rounds * len(args.init), "queries, plus any that are asked again")
        return
    backend, model = cli.backend_of(args)
    out_dir = args.out or cli.default_out(f"lattice_{model}" + ("_reask" if args.reask_nonbest else ""))
    os.makedirs(out_dir, exist_ok=True)
    stop = dict(reason=None)
    lock = threading.Lock()

    def ask(q, text, seed):
        """One query; a failed call is repeated once with its own die seed."""
        for retry in range(2):
            die_seed = seed + ("" if retry == 0 else f"-retry{retry}")
            r = backend.ask(text, tool=roles.RoleDraw(), dice=lattice.Dice(die_seed, "indexed"), system=args.system_prompt)
            if not r["error"] or r["error"] == "max_turns":
                return r, die_seed
        raise Stop(f"call failed twice: {r['error']}")

    def update(job):
        state, s, t, sub, k = job
        if stop["reason"]:
            return None
        q = lattice.make_query(state, s, t, k)
        text = roles.prompt(q)
        rejected = []
        try:
            for i in range(args.max_reask + 1 if args.reask_nonbest else 1):
                r, die_seed = ask(q, text, lattice.dice_seed(s, t, k, "" if i == 0 else f"-rej{i}"))
                rec = make_record(q, text, r, model)
                new = q["own"] if rec["role"] is None else rec["role"]
                if not (args.reask_nonbest and lattice.leaves_best_role(q["n"], q["own"], new)):
                    break
                rejected.append(dict(attempt=i, die_seed=die_seed, answer=rec["answer"], new=new, calls=rec["calls"],
                                     followed_die=rec["followed_die"], text=rec["text"], reasoning=rec["reasoning"]))
        except Stop as e:
            with lock:
                stop["reason"] = stop["reason"] or f"init {s} round {t + 1} site {k}: {e}"
            return None
        all_rejected = args.reask_nonbest and lattice.leaves_best_role(q["n"], q["own"], new)
        if all_rejected:
            new = q["own"]
        out = dict(init=s, t=t, sub=sub, k=k, nb=[state[n] for n in lattice.NEIGHBOURS[k]], mates=q["mates"], n=q["n"], own=q["own"],
                   perm=q["perm"], situation=roles.situation_class(q["n"], q["own"]), prompt=text, answer=rec["answer"], new=new,
                   moved=new != q["own"], is_min=q["n"][new] == min(q["n"]), unparsed=rec["answer"] is None, calls=rec["calls"],
                   followed_die=rec["followed_die"], die_seed=die_seed, text=rec["text"], reasoning=rec["reasoning"],
                   n_requests=r["n_requests"], error=r["error"], model=model)
        if args.reask_nonbest:
            out.update(rejected=rejected, n_rejected=len(rejected), all_rejected=all_rejected)
        return out

    def run(s, ex):
        f_state, f_rec = os.path.join(out_dir, f"s{s}_state.json"), os.path.join(out_dir, f"s{s}.jsonl")
        if os.path.exists(f_state):
            with open(f_state) as f:
                st = json.load(f)
        else:
            st = dict(init=s, states=[lattice.initial_state(s)], order=[], partial=None, zero_at=None)
        # resuming: drop records of sublattices whose state was not saved
        keep = {(t, sub) for t in range(len(st["states"]) - 1) for sub in range(3)}
        if st["partial"]:
            keep |= {(st["partial"]["t"], sub) for sub in st["partial"]["done"]}
        if os.path.exists(f_rec):
            with open(f_rec) as f:
                lines = [l for l in f if (json.loads(l)["t"], json.loads(l)["sub"]) in keep]
            with open(f_rec, "w") as f:
                f.writelines(lines)
        print(f"init {s}: {len(st['states']) - 1} rounds done, conflicts {lattice.conflicts(st['states'][0])} -> {lattice.conflicts(st['states'][-1])}", flush=True)
        while True:
            t = len(st["states"]) - 1
            if st["zero_at"] is not None and t >= st["zero_at"] + lattice.EXTRA_ROUNDS:
                print(f"init {s}: conflict-free at round {st['zero_at']}, ran {lattice.EXTRA_ROUNDS} more rounds, finished", flush=True)
                break
            if st["zero_at"] is None and t >= args.rounds:
                print(f"init {s}: {args.rounds} rounds done, conflicts {lattice.conflicts(st['states'][-1])}", flush=True)
                break
            order = lattice.sublattice_order(s, t)
            if st["partial"] and st["partial"]["t"] == t:
                state, done = list(st["partial"]["S"]), list(st["partial"]["done"])
            else:
                state, done = list(st["states"][-1]), []
            round_records = []
            for sub in order:
                if sub in done:
                    continue
                recs = list(ex.map(update, [(list(state), s, t, sub, k) for k in lattice.SUBLATTICES[sub]]))
                if stop["reason"] or any(r is None for r in recs):
                    raise Stop(stop["reason"])
                for r in recs:
                    state[r["k"]] = r["new"]
                with open(f_rec, "a") as f:
                    for r in recs:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                round_records += recs
                done.append(sub)
                st["partial"] = dict(t=t, done=done, S=state)
                with open(f_state, "w") as f:
                    json.dump(st, f)
            st["states"].append(state)
            st["order"].append(order)
            st["partial"] = None
            c = lattice.conflicts(state)
            if c == 0 and st["zero_at"] is None:
                st["zero_at"] = t + 1
            with open(f_state, "w") as f:
                json.dump(st, f)
            with open(os.path.join(out_dir, "rounds.jsonl"), "a") as f:
                f.write(json.dumps(dict(init=s, round=t + 1, order=order, conflicts=c, state=state)) + "\n")
            print(f"init {s} round {t + 1}: conflicts {c}, changed {sum(r['moved'] for r in round_records)}, "
                  f"rolled {sum(bool(r['calls']) for r in round_records)}, no answer {sum(r['unparsed'] for r in round_records)}"
                  + (f", not applied {sum(r['n_rejected'] for r in round_records)}" if args.reask_nonbest else ""), flush=True)

    try:
        with ThreadPoolExecutor(args.parallel) as ex:
            for s in args.init:
                run(s, ex)
    except Stop as e:
        print(f"STOPPED: {e}. Run the same command again to resume.", flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
