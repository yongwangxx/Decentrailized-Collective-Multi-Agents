# particle-agents-repro

Code and prompts for experiments in which one stateless call of a language model is treated as a particle:
no memory, no communication, no identity. The package contains what is needed to repeat three sets of experiments,

1. **Ising, single particle**: a spin with four neighbours, given a rule, a temperature in words, or an external random-number tool;
2. **Three roles** (three-colouring; the antiferromagnetic three-state Potts model on the triangular lattice): single-particle tests with a goal and a die, 9 × 9 lattice runs with 81 particles, and a reference simulation with a two-parameter rule;
3. **Counterflow in three lanes**: 12 particles on a ring of 40 cells with three lanes, with and without a die, and repeated queries of one symmetric situation.

It contains the prompts, the randomization, the code that asks the models, the run scripts and the scripts that compute the readouts from the records. It contains no results.

## Layout

```
PROMPTS.md                      every prompt word for word, with filled-in examples and the tool definitions
requirements.txt
particle_agents/
    ising.py                    Ising prompts, query tables, parsing
    roles.py                    three-role prompts, situations, query tables, parsing, the "weighted roll" criterion
    lattice.py                  9 x 9 triangular lattice: geometry, initial states, update order, queries
    lanes.py                    three lanes: prompts, lane naming, initial states, move resolution, encounters
    tools.py                    the die and the random-number tool; their random sources
    seeds.py                    the base seed (PA_SEED) from which all randomization derives
    runner.py                   parallel execution, re-asking, JSON Lines records, resuming
    stats.py                    Wilson intervals
    cli.py                      options shared by the run scripts
    backends/
        openai_compat.py        any OpenAI-compatible chat interface
        mock.py                 offline rule-based answers, for trying everything without cost
        template.py             starting point for a backend of your own
scripts/
    run_ising.py                analyze_ising.py
    run_roles_single.py         analyze_roles_single.py
    run_roles_lattice.py        analyze_roles_lattice.py
    sim_reference.py            (no model: two-parameter rule, success map)
    run_lanes_group.py          analyze_lanes.py
    run_lanes_repeat.py
    make_prompts_md.py          regenerates PROMPTS.md from the code
```

Every script starts with a description of what it does, its options and the format of its output; `--help` prints it.

## Installation

Python 3.9 or newer. The scripts are run from the package root; apart from `numpy` for the reference
simulation, only the standard library is used.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`numpy` is needed only by `scripts/sim_reference.py`.

## Try everything offline (mock backend)

The mock backend answers by fixed hand-written rules (described in `particle_agents/backends/mock.py`). It needs no
key and no network, and its answers say nothing about any model; it is there to show the whole pipeline.

```bash
python scripts/run_ising.py --backend mock --sample 40
python scripts/analyze_ising.py

python scripts/run_roles_single.py --backend mock --test main --conditions greedy T0.4 Tc0.5 0 H
python scripts/run_roles_single.py --backend mock --test local_optimum --variant nodie --n 48
python scripts/run_roles_single.py --backend mock --test local_optimum --variant explain --n 48
python scripts/analyze_roles_single.py

python scripts/run_roles_lattice.py --backend mock --init 1 --rounds 8
python scripts/run_roles_lattice.py --backend mock --init 1 --rounds 8 --reask-nonbest
python scripts/analyze_roles_lattice.py output/lattice_mock output/lattice_mock_reask

python scripts/sim_reference.py --point 0 0.2 --runs 100

python scripts/run_lanes_group.py --backend mock --seed 1 --init random --die --rounds 30
python scripts/run_lanes_group.py --backend mock --seed 2 --init even --no-die --rounds 30
python scripts/run_lanes_repeat.py --backend mock
python scripts/analyze_lanes.py
```

All output goes to `./output/` unless `--out` is given. To see the re-asking at work, set `PA_MOCK_BAD_REPLIES=0.2`:
a fifth of the first replies are then unreadable and are asked again.

## Asking a model

How a model is reached is not part of the experiments and is left to you. A backend only has to do one thing: send
one prompt in a new conversation with no history, offer the tool if there is one, and return the reply. Three ways are
provided.

### OpenAI-compatible interface (`--backend openai`)

```bash
export PA_BASE_URL="https://<your-endpoint>/v1"
export PA_API_KEY="<your-key>"
export PA_MODEL="<model-identifier>"
python scripts/run_roles_single.py --backend openai --test local_optimum --variant base --n 20
```

| Variable | Meaning | Default |
|---|---|---|
| `PA_BASE_URL`, `PA_API_KEY`, `PA_MODEL` | address, key and model identifier | required |
| `PA_RPM` | requests per minute sent by one process | 100 |
| `PA_TEMPERATURE` | requested sampling temperature | 1.0 |
| `PA_MAX_TOKENS` | limit of output tokens | 20000 |
| `PA_EXTRA_BODY` | JSON merged into every request body, for fields specific to an interface or a model | nothing |

The key is read from the environment when a request is sent and is not written to any file. Reasoning text, where the
interface returns it, is stored in the records and is never used to determine the answer.

### Your own backend (`--backend <module>:<Class>`)

For a model that is reached in any other way, write a class with the method
`ask(prompt, tool=None, dice=None, system=None, max_turns=4)` and select it by its import path.
`particle_agents/backends/template.py` is a starting point: it contains the conversation loop and the handling of
tool calls, and leaves one method to fill in. The interface is described in `particle_agents/backends/__init__.py`.

```bash
python scripts/run_roles_single.py --backend mypackage.mybackend:MyBackend --test local_optimum --n 20
```

Whatever the route, the model should receive only the prompt (and the system prompt, if one is set) and the offered
tool: no earlier queries, no other tools, no further instructions. Sampling temperature, reasoning effort and similar
settings are yours to choose and should be reported with the results.

### System prompt

By default a query is a single user message. `--system-prompt "Answer the question."` adds a one-line system prompt,
for interfaces that require one or to keep conditions equal between models. `run_roles_lattice.py` sends this
system prompt by default.

## The experiments

One call of a backend is one particle update: a new conversation with no history. A query that uses a tool needs one
more request per tool call. Every run script accepts `--dry-run`, which prints the number of queries of the run and
stops without asking anything.

| Experiment | Command |
|---|---|
| Ising, all groups | `run_ising.py` |
| Ising, selected groups | `run_ising.py --groups rule norule tie_extra switch_tie random_tie temperature temperature_tool` |
| Three roles, single particle, all conditions | `run_roles_single.py --test main` |
| ... selected conditions | `--conditions 0 H F` |
| ... further local-optimum queries | `--test local_optimum --variant base --n <n>` |
| ... control without the die | `--test local_optimum --variant nodie --n <n>` |
| ... reason first | `--test local_optimum --variant explain --n <n>` |
| Three roles, lattice | `run_roles_lattice.py --init <index> --rounds <rounds>` (one query per particle and round) |
| ... asking again when the best role is left | add `--reask-nonbest` |
| Reference simulation | `sim_reference.py` (no model) |
| Three lanes, one arm | `run_lanes_group.py --seed <seed> --init random --die --rounds <rounds>` (and `--no-die`; `--init even`; one query per bird and round) |
| Three lanes, repeated queries | `run_lanes_repeat.py` |

All runs can be interrupted and resumed by repeating the command. `--sample` (Ising, three-role main test, repeated
queries) and `--rounds` (lattice, lanes) give reduced versions.

### Randomization

What is drawn anew, and from which generator, is documented at the top of `ising.py`, `roles.py`, `lattice.py` and
`lanes.py`. In short:

| Experiment | Randomized | Changes |
|---|---|---|
| Ising, tests with letters | which letter denotes +1; order of the four neighbours | every query |
| Ising, temperature tests | nothing: +1 and −1 as written, identical wording in each cell | — |
| Three roles, single particle | assignment of A, B, C to the roles (six, taken in turn); order of the six teammates | every query |
| Three roles, lattice | assignment of A, B, C (drawn independently); order of the six teammates | every query |
| | sublattice order | every round |
| | initial state | every run |
| Three lanes | names of the three lanes and order in which they are listed (36 combinations) | every query |
| | initial state (random or evenly spaced) | every run |
| Die, random-number tool | the number drawn by the program for the weights given by the model | every call |

All of this derives from one base seed that you choose, the environment variable `PA_SEED` (any string, default
`1`; see `particle_agents/seeds.py`). The same base seed reproduces the same prompts, initial states, update orders
and lane names, whatever the order or parallelism of the queries; another base seed gives an independent realization.
The lane runs take in addition a run seed (`--seed`), and the lattice an index of the initial state (`--init`). The
die is seeded per query and attempt in the Ising and three-role experiments and unseeded in the lane runs, where the
number drawn is recorded with every roll. The reference simulation has its own numeric `--seed`.

### Queries that have to be asked again

- Single-particle tests, lane runs, repeated queries: a query whose call failed or whose reply contains no readable
  answer does not count; its record is kept and the same query is asked again, up to `--max-attempts` times (default
  10), after which the run stops. `--max-attempts 1` never asks again. The analysis scripts use the first attempt with
  a readable answer.
- Lattice: every answer is applied, and a reply without a readable answer leaves the role unchanged. With
  `--reask-nonbest` an answer that leaves the particle's best role is recorded but not applied and the same question
  is asked again (at most `--max-reask` times, default 5). A failed call is repeated once; if it fails again the run stops and can be resumed.

## Output

All records are JSON Lines. The exact fields are listed at the top of each run script.

- Single-particle tests and repeated queries: one line per attempt, with the parameters of the query, the full prompt,
  the reply text, the parsed answer, the tool calls (arguments, the random number drawn, the result returned), the
  reasoning text if the interface returns one, the error if any, and the attempt number.
- Lattice: one line per update in `s<init>.jsonl`, one line per round in `rounds.jsonl` (state and number of
  conflicts), and `s<init>_state.json` with the states after every round.
- Lanes: one line per round with the positions before and after, the twelve prompts, replies and answers, the lane
  names used for each bird, which moves happened, and the rolls of the die.

## Readouts

| Script | Computes |
|---|---|
| `analyze_ising.py` | majority followed, behaviour at ties, answers to identical prompts, P(+1) against the heat-bath value, consistency of decisions made with the random-number tool |
| `analyze_roles_single.py` | role changed without a conflict, non-minimal role chosen, local optimum left, queries with a roll, departures on a weighted roll, mean die weights, agreement with prescribed rules |
| `analyze_roles_lattice.py` | conflicts per round, updates by situation, departures from a local optimum and those made on a weighted roll, moves that raise the particle's own conflicts, mean die weights |
| `sim_reference.py` | fraction of runs of the two-parameter rule that reach zero conflicts within the run length, as a map over (p_c, p_h) or at one point |
| `analyze_lanes.py` | moves per round, rounds with sorted lanes, lane order, head-on encounters resolved in the round (symmetric and others), answers repeated when an encounter recurs unchanged, where the die is rolled; actions in the repeated queries |

The criterion "a departure from a local optimum made on a weighted roll of the die" is `roles.weighted_roll`: the last
valid roll of the query gave staying a positive weight, gave at least two roles a positive weight, gave weights that do
not increase with the number of conflicts, and the answer is the role rolled and differs from the current role. It uses
only the tool calls and the answer, not the reply text.
