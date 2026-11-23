# Collective order from language models without communication, memory or shared context

Code, prompts and animations for the paper of the same title.

Each memoryless query to a language model is treated as a particle: it sees its neighbours, but it cannot
communicate, has no identifier and no shared context, and receives the same question as every other particle.
The repository contains what is needed to repeat the three sets of experiments: a single Ising spin, three roles
on a triangular lattice, and counterflow in three lanes.

**Watch decentralized collective order emerge:** twelve agents with no leader, no memory and no communication sort mixed traffic into one-way lanes — https://yongwangxx.github.io/Decentrailized-Collective-Multi-Agents/

- `code/` — prompts, randomization, run scripts and analysis scripts for all three sets of experiments.
  Start with `code/README.md`; every prompt is given word for word in `code/PROMPTS.md`.
- `docs/` — the web page with recordings of six three-lane runs and the prompt behind them.

## Citation

If you use this code, please cite the paper *Collective order from language models without communication, memory or shared context*.

## License

MIT, see `LICENSE`.
