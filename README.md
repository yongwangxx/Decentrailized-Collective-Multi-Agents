# Lanes without a leader

Twelve birds fly around a ring with three lanes, six in each direction. Every move of every bird is one separate
query to a language model: a bird sees only the cells around it, has no memory, cannot communicate and has no name.
Mixed traffic sorts itself into one-way lanes.

**Watch the runs:** https://yongwangxx.github.io/Decentrailized-Collective-Multi-Agents/

- `docs/` — the web page, with recordings of six runs and the prompt each bird receives.
- `code/` — prompts, randomization, run scripts and analysis scripts for the bird runs and for two further
  experiments (Ising single particle, three roles on a lattice). Start with `code/README.md`; all prompts are in
  `code/PROMPTS.md`.

## Citation

If you use this code, please cite the paper *Collective order from language models without communication, memory or shared context*.

## License

MIT, see `LICENSE`.
