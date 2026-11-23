"""Three-role problem, single particle: ask the main test or a local-optimum variant.

Main test (--test main): for every condition, the 16 situations with the numbers of queries given by
roles.repetitions (--dry-run prints the total).
Condition labels: greedy T0.4 T0.7 Tc0.5 0 H F HF HFI S HS SF HSF HSFI ("0" is the goal-and-die prompt).

Local optimum (--test local_optimum): only the situation with teammates 3-2-1 and the own role on the 1,
queries number --start ... --n - 1 of one variant:
    base     the goal-and-die prompt
    nodie    the same prompt without the sentence on the tool, and no tool offered
    explain  the goal-and-die prompt, last sentence replaced by a request to state the reason first

Examples
    python scripts/run_roles_single.py --backend mock --test main --conditions greedy T0.4 Tc0.5 0 H
    python scripts/run_roles_single.py --backend openai --test local_optimum --variant base --n 200
    python scripts/run_roles_single.py --backend openai --test local_optimum --variant nodie --n 200

Output: <out>/roles_<test>_<model>.jsonl, one JSON object per attempt:
    id, condition, sit, n, own, mates, perm, [T, r, prescribed], [variant], prompt,
    answer (letter), role (0/1/2), moved, is_min, [correct], calls (with w_role, ret_role), followed_die,
    text, reasoning, n_requests, error, model, attempt
"""
import _path  # noqa: F401
import os

from particle_agents import cli, roles
from particle_agents.runner import run_queries


def make_record(q, prompt_text, reply, model, require_answer_word=False):
    """Turn a backend reply into a record. Also used by the lattice script."""
    letter = None if reply["error"] else roles.parse(reply["text"], require_answer_word)
    role = None if letter is None else q["perm"].index(letter)
    calls = roles.calls_by_role(reply["calls"], q["perm"])
    valid = [c for c in calls if c.get("valid")]
    rec = dict(q, prompt=prompt_text, answer=letter, role=role,
               moved=None if role is None else role != q["own"],
               is_min=None if role is None else q["n"][role] == min(q["n"]),
               followed_die=None if (role is None or not valid) else role == valid[-1]["ret_role"],
               calls=calls, text=reply["text"], reasoning=reply["reasoning"], n_requests=reply["n_requests"],
               error=reply["error"], model=model)
    if q.get("prescribed") is not None:
        rec["correct"] = None if role is None else role == q["prescribed"]
    return rec


def main():
    p = cli.parser(__doc__)
    p.add_argument("--test", choices=["main", "local_optimum"], default="main")
    p.add_argument("--conditions", nargs="+", default=roles.CONDITIONS, choices=roles.CONDITIONS)
    p.add_argument("--variant", choices=["base", "nodie", "explain"], default="base")
    p.add_argument("--n", type=int, default=100, help="local optimum: queries 0 ... n-1")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--sample", type=int, default=None, help="main test: ask only this many queries per condition, spread over the table")
    args = p.parse_args()
    if args.test == "main":
        Q = [q for c in args.conditions for q in cli.spread(roles.queries([c]), args.sample)]
    else:
        Q = [roles.local_optimum_query(args.variant, i) for i in range(args.start, args.n)]
    if args.dry_run:
        print(len(Q), "queries")
        return
    backend, model = cli.backend_of(args)
    out_dir = args.out or cli.default_out("roles")

    def ask_one(q, attempt):
        text = roles.prompt(q)
        r = backend.ask(text, tool=roles.tool_for(q), dice=roles.dice_for(q, attempt), system=args.system_prompt)
        return make_record(q, text, r, model, require_answer_word=q.get("variant") == "explain")

    name = f"roles_main_{model}.jsonl" if args.test == "main" else f"roles_local_optimum_{args.variant}_{model}.jsonl"
    run_queries(Q, ask_one, os.path.join(out_dir, name), args.parallel, args.max_attempts)


if __name__ == "__main__":
    main()
