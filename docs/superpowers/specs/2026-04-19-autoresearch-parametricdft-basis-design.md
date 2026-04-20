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

## Addendum (2026-04-19, post-scaffold): Walsh-Hadamard collapse and DFT gap

After seeding baselines and running 5 trials (branch `autoresearch/initial`,
commits `20f47e1..3acae5e`), an empirical finding surfaced that sharpens this
spec:

**All four ParametricDFT "baselines" reduce to the same compressor under the
top-k MSE metric.** Each of `QFTBasis`, `EntangledQFTBasis`, `TEBDBasis`,
`MERABasis` has circuit form `H⊗ⁿ · D · H⊗ⁿ` with `D` a diagonal unitary of
learnable phases. Top-k magnitude selection is invariant under any diagonal
unitary (`|D c| = |c|`), and `D†` cancels `D` on the retained coefficients
during inverse. The round-trip therefore computes a Walsh-Hadamard transform
regardless of the phase values, giving MSE ≈ 2696.07 on the fixture for QFT,
EntangledQFT, TEBD, MERA, and every phase-only variant the agent tried.

**What the fixture actually rewards.** The fixture is band-limited (Gaussian
random field, low-pass bandwidth 32); ~3208 of its 262_144 Fourier
coefficients are non-zero. A standard 2D DFT therefore achieves near-perfect
reconstruction at `k = 26_214` (MSE ≈ 2.6e-26 in numpy). So the true
achievable MSE on this fixture is ~0, not ~2700. The 2700 gap is not an
architectural ceiling — it's that none of the existing baselines implement a
proper DFT.

**Implication: Yao's `qft_circuit` is not the standard DFT on this einsum
pipeline.** Empirically (verified by the `BitReversedQFTBasis` trial),
`|optcode(QFTBasis.tensors..., x)|` does not match `|fft(x)|` even after
bit-reversal SWAPs. A winning basis likely needs to construct the DFT
tensor network manually (specify the CPHASE phases as `exp(-2πi jk/N)`
directly) rather than rely on `Yao.qft_circuit`.

**Revised acceptance interpretation.** The original acceptance rule
(`final_mse < best_so_far * 0.99`) remains correct — but the target
is much lower than TEBD's 2696.07. A correctly-constructed DFT basis
should drive MSE to effectively zero on this fixture; the search space
for qualitatively different bases is broader than just "variants of
phase-gate circuits."

**Baseline decision, revisited.** Because the four existing baselines are
equivalent under this metric, there's a case for collapsing `BASELINES` to a
single Walsh-Hadamard reference plus an explicit "classical FFT" reference
(for the theoretical floor). That would be a spec change; the current
implementation keeps the three non-MERA baselines for continuity with
ParametricDFT's existing types.

## Addendum B (2026-04-20, post-DFT-win): sign convention + exhaustion

**DFTBasis landed on 2026-04-19 (commit `6f14daa`) at MSE 3.54e-24 — machine
precision, 26 orders of magnitude below TEBD.** The fix was an input-leg
permutation on the einsum (`qubit_perm = vcat(reverse(1:m), reverse(m+1:m+n))`):
the QFT tensors were already correct, only the bits were being fed to qubits
in reversed significance order.

**Numerical validation (m=n=3, 64-dim transforms):**

| Statement | Relative error |
|---|---:|
| `F_qft = F_dft · P` where P is 2D bit-reversal | 2.2e-16 |
| `DFT(bit-reverse(img)) = QFT(img)` pointwise | 2.5e-16 |
| Both unitary: `FF† = N·I` | 2.2e-16 |
| `F_dft = conj(F_std)` where F_std is textbook 2D DFT | 7.9e-15 |

So DFTBasis doesn't implement the textbook *forward* DFT (`exp(-2πi kj/N)`) —
it implements the *inverse / conjugate* variant (`exp(+2πi kj/N)`). This is
irrelevant for compression because top-k magnitude selection is identical
under global conjugation (`|conj(F·x)| = |F·x|`), but documenting it for
precision: "DFTBasis *is the textbook inverse DFT*, not the forward DFT."
The earlier addendum language "true 2D DFT" stands in the loose sense that
any bijective unitary of the 2D-DFT family saturates compression of this
band-limited fixture.

**Search-space exhaustion.** With machine-precision MSE achieved, the
acceptance rule (`final_mse < best * 0.99`) makes further trials on this
fixture unsalvageable: no basis can meaningfully improve on ~3.5e-24, so
every new attempt is dropped at the `kept` gate regardless of its
mathematical merit. The productive next directions are spec changes, not
more basis variants:

1. **Harder fixture** — replace the band-limited Gaussian field with a
   natural image (e.g., a photograph or MNIST sample upscaled to 512×512).
   Non-band-limited content means DFT's compression is no longer perfect,
   and bases that exploit spatial structure (wavelets, MERA with proper
   padding, hybrid transforms) could beat pure DFT.
2. **Tighter sparsity** — drop k from 26_214 (10%) to, say, 1% or 0.1%.
   At extreme sparsity, DFT no longer captures all the energy, and
   alternative bases that concentrate energy differently have a chance.
3. **Multi-fixture evaluation** — evaluate each basis on a *set* of images
   and rank by average MSE. Rewards bases that generalize, not just
   overfit one fixture.

All three are harness-update PRs (edit frozen files, rehash, rotate salt).
Until one is landed, the autoresearch loop has effectively converged.

## Addendum C (2026-04-20, post-multi-fixture): two fixtures + no-op-training bug

The multi-fixture eval protocol landed on branch `harness/multi-fixture`
(fork point `807ace7`). `evaluate_basis` now averages `loss_function(…)`
across every entry in `load_fixtures()`, and the default fixture set is
two images:

1. **`data/fixture_512.bin`** — the original band-limited Gaussian random
   field (seed 42, low-pass bandwidth 32). DFT-friendly.
2. **`data/fixture2_512.bin`** — a piecewise-smooth image (seed 43):
   overlapping rectangles, a diagonal stripe, four Gaussian bumps, and a
   low-amplitude noise floor. Non-band-limited; DFT cannot achieve
   near-zero MSE at k=26_214 because sharp edges spread energy over the
   full Fourier spectrum.

**Results (initial trials on `harness/multi-fixture`):**

| basis | MSE (mean over 2 fixtures) | params | note |
|---|---:|---:|---|
| QFT          | 3891.81 | 360    | baseline (Walsh-Hadamard floor) |
| EntangledQFT | 3891.81 | 396    | baseline (ties QFT — phases invariant) |
| TEBD         | 2260.11 | 144    | baseline (local gates help on edges) |
| DFTBasis     |  837.76 | 360    | kept — DFT from main branch, works without additional change |
| DCTBasis     |  721.57 | 524288 | kept — even-symmetric extension → smaller Gibbs tail on fixture 2 |
| **BlockDCTBasis (32×32)** | **641.91** | 524288 | **kept, leader** — JPEG-style block-diagonal DCT localizes the transform |

BlockDCTBasis is **71.6% better than TEBD** and 23.3% better than DFT. The
progression DFT → DCT → BlockDCT echoes the historical progression that
led to JPEG: a global orthogonal transform → a global even-symmetric
transform → a block-local version of the latter.

### No-op-training bug (surfaced by the multi-fixture session)

Empirically confirmed: **`train_basis` is currently a no-op under this
harness's default config.** `_train_basis_core` in ParametricDFT tracks
`best_tensors` via the check

```julia
if val_loss < best_val_loss
    best_tensors = copy.(current_tensors)
    best_val_loss = val_loss
end
```

With the frozen config `validation_split = 0.0`, the validation set is
empty; `val_loss` is computed from an empty `validation_data` array and
evaluates to `Inf` on every epoch. `best_val_loss` starts at `Inf`, and
`Inf < Inf` is always false, so `best_tensors` *never updates from the
initial tensors*. The final basis returned is the initial one, regardless
of how many Adam steps were taken or whether they reduced training loss.

**Empirical confirmation:**
- QFTBasis with zero-initialized phases: `final_mse` exactly matches the
  `run_probe()` value (no phase learning happened).
- DFTEntangledBasis with zero-initialized entangling phases: after 500
  Adam steps, phases are still zero. Hence dropped.

**Implications for this leaderboard:**
- Every row in `results.tsv` (both baseline and trial) reflects the
  *initial* basis configuration, not a trained one. The leaderboard is
  effectively an "initialization quality" benchmark.
- Parametric bases with learnable phases (EntangledQFT, TEBD, DeepTEBD,
  HadSandwichTEBD, DFTEntangled) are functionally identical to their
  zero-init forms under this harness.
- BlockDCTBasis's win is real — it's a fixed (non-parametric) transform
  whose initialization is exactly what gets evaluated.
- The 500-step training budget is wasted wall-clock per trial.

**Fix paths (not applied in this PR):**
1. **Upstream fix.** Modify `_train_basis_core` to update `best_tensors`
   based on `train_loss` when `validation_split == 0.0`. This is the
   right fix and should land in ParametricDFT.jl upstream. Would also
   require bumping the pinned SHA in our `Project.toml` once merged.
2. **Local workaround.** Set `validation_split = 0.1` in the harness.
   With only 2 fixtures this leaves 0 training images after the split —
   fragile. Only viable if the fixture count grows.
3. **Reimplement train logic.** Bypass `train_basis` entirely in
   `src/harness/train.jl` and roll our own loop that snapshots the
   minimum-training-loss tensors. Works but is a significant harness
   change; would need recomputing probe and rehashing.

**Recommendation.** Until the bug is fixed, treat the search space as
"fixed-initialization bases." BlockDCTBasis at 642 is genuinely the best
FIXED transform we've found; to beat it meaningfully would require either
(a) a smarter fixed transform (e.g., per-block learned basis via a
pre-training step outside the harness), or (b) the harness bug getting
fixed so training actually works.

## Open questions

- Should upstream `_train_basis_core` be patched to update
  `best_tensors` when `validation_split == 0`? This is the cleanest path
  and benefits everyone using `ParametricDFT.train_basis`.
- Is the DFT-vs-QFT discrepancy fixable in upstream `qft_code` (change
  the einsum input-leg convention), so downstream callers don't have to
  permute?
- Should the fixture set grow further (e.g., 4-8 images spanning text,
  photos, medical scans) once the training bug is fixed so that
  genuinely learnable bases have room to differentiate?

## Changelog

- 2026-04-19: Initial design approved via brainstorming skill.
- 2026-04-19 (later): Addendum A on Walsh-Hadamard collapse and DFT gap,
  based on 5 trials on branch `autoresearch/initial`.
- 2026-04-20: Addendum B — sign convention clarification (DFTBasis is the
  inverse-DFT, not forward), numerical validation at m=n=3, and search-
  space-exhaustion analysis proposing three productive spec changes.
- 2026-04-20 (multi-fixture): Addendum C — second fixture introduced,
  eval protocol now averages across all fixtures. Surfaces the no-op-
  training bug in `_train_basis_core` (with `validation_split=0`, final
  basis = initial basis). BlockDCTBasis(32×32) leads at MSE 641.91 on
  fixed-initialization comparison.
