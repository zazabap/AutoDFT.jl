---
title: "AutoDFT.jl: Autoresearch Harness for Sparse Basis Discovery"
status: approved
date: 2026-04-19
author: Shiwen An (@zazabap)
related:
  - https://github.com/nzy1997/ParametricDFT.jl
  - https://github.com/fliingelephant/autoresearch-hubbard
---

# AutoDFT.jl: Autoresearch Harness for Sparse Basis Discovery

## Context

`ParametricDFT.jl` implements a family of parametric sparse bases — `QFTBasis`,
`EntangledQFTBasis`, `TEBDBasis`, `MERABasis` — over a common
`AbstractSparseBasis` interface, trained with Riemannian optimizers on unitary
manifolds. The open research question: *can we discover a new basis that beats
all four baselines on the compression task at 512×512, 10% sparsity?*

This document specifies a new repository, `AutoDFT.jl`, modeled on
[autoresearch-hubbard](https://github.com/fliingelephant/autoresearch-hubbard).
It freezes the evaluation pipeline behind a SHA-hashed manifest and a
deterministic probe, then lets an unattended Claude Code session iterate on
new `AbstractSparseBasis` implementations, keeping only the commits that beat
the current leader.

## Goals

- **Primary.** Provide a harness under which new `AbstractSparseBasis`
  implementations can be proposed, trained, and accepted iff they lower
  reconstruction MSE vs. the best existing baseline at a fixed sparsity on a
  fixed 512×512 image.
- **Reproducibility.** Any accepted basis, together with the committed
  `Manifest.toml`, recomputes to the same MSE on any CUDA-capable host (or
  CPU fallback) to within `atol=1e-6`.
- **Agent discipline.** The autoresearch loop cannot silently move goalposts.
  Attempts to edit the frozen surface are caught locally (SHA check) *and* in
  CI (salted manifest SHA + identity probe).

## Non-goals

- Modifying `ParametricDFT.jl` internals (bases, optimizers, loss, training
  loop). AutoDFT depends on it as a pinned Julia package.
- Exploring other image sizes, other sparsity levels, or multi-image
  distributions. These are future extensions.
- Inventing new losses, optimizers, manifolds, or training procedures.
- Publishing AutoDFT as a reusable library. It is a research repo.

## Design

### Repository layout

`[F]` = frozen (SHA-hashed, CI-enforced). `[E]` = editable (agent may add/modify).

```
AutoDFT.jl/
├── .claude/
│   ├── CLAUDE.md                   [F]  project guidance
│   └── rules/julia-conventions.md  [F]
├── .github/workflows/
│   ├── CI.yml                      [F]
│   └── basis-freeze.yml            [F]
├── src/
│   ├── AutoDFT.jl                  [F]  top-level module
│   ├── harness/
│   │   ├── train.jl                [F]  wraps ParametricDFT.train_basis
│   │   ├── evaluate.jl             [F]  forward → topk → inverse → MSE
│   │   ├── fixture.jl              [F]  loads data/fixture_512.bin
│   │   └── probe.jl                [F]  BASIS_IDENTITY_PROBE
│   └── bases/
│       ├── registry.jl             [E]  name → constructor map
│       └── <new_basis>.jl          [E]
├── data/fixture_512.bin            [F]  Float32 512×512 image
├── scripts/
│   ├── generate_fixture.jl         [F]  regenerates data/fixture_512.bin
│   └── compute_manifest.jl         [F]  rehashes frozen-manifest.toml
├── test/
│   ├── runtests.jl                 [F]
│   ├── harness_tests.jl            [F]
│   └── bases_tests.jl              [E]
├── docs/superpowers/{specs,plans}/ [E]
├── Project.toml / Manifest.toml    [F]  pinned ParametricDFT.jl SHA
├── Makefile                        [F]  init / test / baseline / trial
├── frozen-manifest.toml            [F]
├── prepare.jl                      [F]
├── program.md                      [F]  autoresearch runbook
├── results.tsv                     [E]  append-only trial log
├── best.tsv                        [E]  current leader (atomic replace)
├── README.md                       [F]
└── LICENSE                         [F]  MIT
```

### Frozen surface and enforcement

Three independent layers catch different failure modes:

| Layer | What it catches | Local | CI |
|---|---|---|---|
| Per-file SHA256 in `[files]` of `frozen-manifest.toml` | byte edits to frozen files | ✓ | ✓ |
| Salted manifest SHA256 (`BASIS_FREEZE_SALT` from CI secret) | coordinated edit to both file and manifest entry | ✗ | ✓ |
| `BASIS_IDENTITY_PROBE` in `test/runtests.jl` | semantic drift via Manifest swap or numerics change | ✓ | ✓ |

**Probe.** `QFTBasis(9, 9)` has no learnable parameters, so
`forward_transform → topk_truncate(k=26_214) → inverse_transform` on the fixture
is fully deterministic. Its MSE is pinned under `[probe].qft_identity_mse` and
asserted `≈` with `atol=1e-10` on every run.

**Manifest.toml is frozen.** Committing `Manifest.toml` (not just
`Project.toml`) is the only way to pin the exact `ParametricDFT.jl` commit
along with every transitive dep. The manifest's own SHA is among the hashed
files, so dependency swaps are detected.

### Evaluation protocol

Fixed for every trial:

| Parameter | Value |
|---|---|
| Fixture | `data/fixture_512.bin` — Float32 512×512 from `scripts/generate_fixture.jl`, `Random.seed!(42)`, low-pass Gaussian random field |
| `k` (sparsity) | `26_214` (= `floor(0.1 * 2^18)`) |
| Loss | `MSELoss(k)` from ParametricDFT.jl |
| Optimizer | `RiemannianAdam(lr=0.01)` |
| Steps | `500` |
| Batch | `1` |
| Seed | `Random.seed!(42)` before every parameter init |
| Device | `CUDA.functional() ? GPU : CPU` (identical MSE to `atol=1e-6`) |
| Metric | `final_mse = loss_function(trained_basis, fixture)` after last step |
| Baseline bar | `min(MSE)` over the four ParametricDFT baselines, seeded by `make baseline` |
| Acceptance | `final_mse < best_so_far * (1 - 0.01)` (≥1% relative improvement) |

### Autoresearch loop (`program.md`)

```
1. Setup
   a. Read program.md, .claude/CLAUDE.md, docs/superpowers/specs/.
   b. `make init` (or `make init-fresh` if Project.toml changed).
   c. `julia --project=. prepare.jl` → must print "frozen surface OK".

2. Seed leaderboard (once per branch)
   a. `make baseline` → evaluates QFT/EntangledQFT/TEBD/MERA on fixture,
      appends rows to results.tsv, writes best.tsv.

3. Iterate (until user Ctrl+C)
   a. Read results.tsv + best.tsv. Form a hypothesis.
   b. Add src/bases/<slug>.jl: `struct <Name>Basis <: AbstractSparseBasis ... end`
      with forward_transform, inverse_transform, image_size, num_parameters,
      basis_hash. Register in src/bases/registry.jl.
   c. Add a conformance @testset in test/bases_tests.jl.
   d. `git commit -am "trial: <Name>Basis — <one-line hypothesis>"`.
   e. `make trial NAME=<Name>Basis` → prepare + train + evaluate + append row;
      exits 0 if accepted, 1 if rejected.
   f. If accepted: update best.tsv, commit it, continue.
      If rejected: `git reset --hard HEAD~1`, then commit a dropped-trial note
      on a separate commit (survives the reset), continue.

4. Kill switch: Ctrl+C. Never edit frozen files — CI will reject.
```

### Results schema

`results.tsv` — tab-separated, header on line 1:

```
timestamp  branch  commit_sha  basis_name  basis_hash  num_parameters
  final_mse  probe_mse  train_steps  train_wallclock_ms  transform_time_ms
  device  status  notes
```

- `basis_hash` — `ParametricDFT.basis_hash(basis)` (params + struct).
- `probe_mse` — recomputed on every row so old runs can be replayed and
  cross-validated.
- `status` ∈ `{baseline, kept, dropped}`.
- `notes` — short free-form (hypothesis + observation).

`best.tsv` — single data row mirroring the leading `results.tsv` entry.
Atomically replaced by `make trial` on acceptance.

### Makefile targets

| Target | Behavior |
|---|---|
| `make init` | `Pkg.instantiate()` |
| `make init-fresh` | resolve + update Manifest after Project.toml changes |
| `make test` | frozen-surface verify + `runtests.jl` |
| `make baseline` | evaluates the four ParametricDFT baselines on fixture; seeds `results.tsv` + `best.tsv`; idempotent (skips if already seeded on this branch) |
| `make trial NAME=<BasisName>` | prepare + train + evaluate; append row; exit 0/1 per acceptance |
| `make verify` | `prepare.jl` only |

### Testing

1. **Frozen-surface verify (`prepare.jl`).** Reads `frozen-manifest.toml`,
   recomputes SHA256 of each listed file. Verifies `Manifest.toml` hash matches
   manifest entry. In CI, additionally verifies
   `sha256(salt || concat(file_shas)) == [secret].manifest_sha`.

2. **Identity probe (`test/harness_tests.jl`).** Constructs `QFTBasis(9, 9)`,
   runs frozen eval on fixture, asserts
   `probe_mse ≈ manifest.probe.qft_identity_mse atol=1e-10`.

3. **Interface conformance (`test/harness_tests.jl`, parametrized over
   `registry.jl`).** For each registered basis: `forward ∘ inverse ≈ I` on
   random input with seed 42, `image_size == (512, 512)`, `num_parameters ≥ 0`,
   `basis_hash` is deterministic.

4. **Agent's basis tests (`test/bases_tests.jl`).** Per-basis `@testset` the
   agent adds — unitarity, interface checks. Required by
   `.claude/rules/julia-conventions.md` (">95% test coverage for new code").

### CI

- **`.github/workflows/CI.yml`** — matrix on Julia 1.10 (LTS) + 1.11,
  ubuntu-latest, CPU only. Runs `make init && make test`. Required for all PRs.

- **`.github/workflows/basis-freeze.yml`** — runs `prepare.jl` with
  `BASIS_FREEZE_SALT` injected from repo secrets. Required for PRs to `main`.

- **Branch protection on `main`** — PR required; `basis-freeze/verify` and
  `CI` must pass; no force-push; code-owner review optional for single-user
  repo.

- **GPU tests opt-in** — only run when commit message contains `[gpu]`
  (mirrors ParametricDFT.jl convention).

## Deliverables

The implementation session will produce, in order:

1. Write `.claude/CLAUDE.md` with project overview, frozen-surface rules,
   runbook pointer, pointer to this spec.
2. Retitle `.claude/rules/julia-conventions.md` references from ParametricDFT
   to AutoDFT where needed; content otherwise inherited verbatim.
3. Scaffold the full layout above with real (not placeholder) contents:
   harness, fixture generator, manifest builder, Makefile, workflows,
   `program.md`, an `src/bases/example_basis.jl` template, README.
4. `git add` + initial commit; pin `ParametricDFT.jl` in `Project.toml` at a
   specific commit SHA (latest `main` at scaffolding time).
5. **(Pause for confirmation)** `gh repo create zazabap/AutoDFT.jl --public
   --source=. --push`.
6. If ParametricDFT.jl installs cleanly locally, run `make baseline` to seed
   `results.tsv` + `best.tsv`; otherwise leave them empty with a
   `program.md` note instructing the first session to run it.

## Open questions

None at design time.

## Changelog

- 2026-04-19: Initial design approved via brainstorming skill.
