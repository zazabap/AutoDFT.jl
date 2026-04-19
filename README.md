# AutoDFT.jl

An autoresearch harness over [ParametricDFT.jl](https://github.com/nzy1997/ParametricDFT.jl). Searches for new `AbstractSparseBasis` implementations that beat the four baselines (QFT, EntangledQFT, TEBD, MERA) on reconstruction MSE of a frozen 512×512 image at 10% sparsity.

Modeled on [autoresearch-hubbard](https://github.com/fliingelephant/autoresearch-hubbard): a SHA-hashed `frozen-manifest.toml` + deterministic probe pin the evaluation surface, so an autonomous Claude Code session can iterate on bases without silently moving goalposts.

## Quick start

```bash
git clone https://github.com/zazabap/AutoDFT.jl.git
cd AutoDFT.jl
make init                   # instantiate Manifest.toml (pins ParametricDFT SHA)
make test                   # frozen-surface verify + test suite
make baseline               # seeds results.tsv with 4 baselines (~10-60 min)
```

Then open an autonomous session:
```bash
CLAUDE_AUTOCOMPACT_PCT_OVERRIDE=28 claude --permission-mode bypassPermissions
# In the session:  "Read program.md and start a new autoresearch experiment."
```

## What's frozen

Spec: `docs/superpowers/specs/2026-04-19-autoresearch-parametricdft-basis-design.md`.

- `Project.toml` + `Manifest.toml` — pins ParametricDFT.jl at commit `79117aa8`.
- `src/harness/*.jl` — training + evaluation pipeline.
- `data/fixture_512.bin` — the 512×512 test image (seed-42 Gaussian random field, low-pass).
- Numerical config: `k=26_214`, `m=n=9`, `RiemannianAdam(lr=0.01)`, 500 steps, seed 42.

What's editable: `src/bases/*.jl` and `test/bases_tests.jl`. Full rules in `program.md`.

## Acceptance rule

A new basis is kept if `final_mse < best_so_far * (1 - 0.01)` (≥1% relative
improvement). Otherwise the trial commit is reverted and logged as dropped.

## References

- Spec: `docs/superpowers/specs/2026-04-19-autoresearch-parametricdft-basis-design.md`
- Runbook: `program.md`
- Results: `results.tsv`, leader: `best.tsv`
- Upstream: https://github.com/nzy1997/ParametricDFT.jl

## License

MIT (matches ParametricDFT.jl).
