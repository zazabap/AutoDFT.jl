# CLAUDE.md — AutoDFT.jl

AutoDFT.jl is an autoresearch harness over [ParametricDFT.jl](https://github.com/nzy1997/ParametricDFT.jl). Its only purpose is to discover new `AbstractSparseBasis` implementations that achieve lower reconstruction MSE than the four ParametricDFT baselines on a frozen 512×512 image at 10% sparsity.

**Design spec:** `docs/superpowers/specs/2026-04-19-autoresearch-parametricdft-basis-design.md` — read this before making any non-trivial change.

**Autoresearch runbook:** `program.md` — read before starting a trial session.

## Frozen vs. editable

Files are annotated `[F]` (frozen, SHA-hashed) or `[E]` (editable) in the spec. You **may not** edit any `[F]` file outside a dedicated "harness update" PR that also rotates `frozen-manifest.toml` and `[secret]` (requires `BASIS_FREEZE_SALT`). Attempting to edit a frozen file without rehashing will fail CI.

- **Editable:** `src/bases/*.jl` (new bases + `registry.jl`), `test/bases_tests.jl`, `results.tsv`, `best.tsv`, `docs/superpowers/plans/*`.
- **Frozen:** everything else.

## How to add a new basis (agent's main workflow)

1. Read `best.tsv` — that's the bar to beat.
2. Copy `src/bases/_example_basis.jl` → `src/bases/<slug>.jl`. Rename `IdentityBasis` → `<Name>Basis`.
3. Implement `forward_transform`, `inverse_transform`, `image_size`, `num_parameters`, `basis_hash`. Extend `ParametricDFT._init_circuit` and `ParametricDFT._build_basis` so `train_basis` can dispatch on your type.
4. Register in `src/bases/registry.jl`: `register_basis!("<Name>", <Name>Basis)`.
5. Add a `@testset` to `test/bases_tests.jl` verifying unitarity/interface conformance.
6. `git commit -am "trial: <Name>Basis — <one-line hypothesis>"`.
7. `make trial NAME=<Name>` — exits 0 if accepted (MSE ≤ best × 0.99), 1 otherwise.
8. If rejected: `git reset --hard HEAD~1`, then commit a dropped-trial note separately so it survives.

## Conventions

- Julia style: `.claude/rules/julia-conventions.md` (inherited from ParametricDFT.jl).
- Tests: every new basis gets a `@testset` block. Use `Random.seed!(42)` for reproducibility. Compare floats with `≈` + `atol`, never `==`.
- Frozen numerics: `k = 26_214`, `m = n = 9`, `RiemannianAdam(lr=0.01)`, 500 steps, seed 42. Do not change these — they're enforced by `prepare.jl`.

## Results schema

`results.tsv` (tab-separated):
```
timestamp  branch  commit_sha  basis_name  basis_hash  num_parameters  final_mse  probe_mse  train_steps  train_wallclock_ms  transform_time_ms  device  status  notes
```
`status` ∈ `{baseline, kept, dropped}`.
