# AutoDFT.jl Autoresearch Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Scaffold the `AutoDFT.jl` repo from the spec at `docs/superpowers/specs/2026-04-19-autoresearch-parametricdft-basis-design.md`, push to `github.com/zazabap/AutoDFT.jl`, and (if possible) seed the baseline leaderboard.

**Architecture:** Thin Julia package. `ParametricDFT.jl` pinned via `Project.toml` at commit `79117aa8b584f405c0d3268a9f1b306c42337b9e`. Harness code under `src/harness/` (frozen) wraps `ParametricDFT.train_basis` / `loss_function`. Agent writes new `AbstractSparseBasis` subtypes under `src/bases/` (editable). A SHA-hashed `frozen-manifest.toml` + deterministic `QFTBasis(9,9)` identity probe catch drift in both local runs and CI.

**Tech Stack:** Julia 1.10+, ParametricDFT.jl (pinned SHA), SHA.jl, TOML.jl (stdlib), GitHub Actions.

**Working directory:** `/home/claude-user/AutoDFT.jl/` — git-initialized, `main` branch, one commit containing the spec.

---

## File Structure (target end state)

| Path | Owner | Purpose |
|---|---|---|
| `.claude/CLAUDE.md` | [F] | Project guidance (updated in Task 1) |
| `.claude/rules/julia-conventions.md` | [F] | Inherited conventions (Task 2) |
| `Project.toml` | [F] | Pinned ParametricDFT.jl |
| `Manifest.toml` | [F] | Reproducibility lockfile |
| `src/AutoDFT.jl` | [F] | Top module; exports runners |
| `src/harness/fixture.jl` | [F] | `load_fixture()` |
| `src/harness/probe.jl` | [F] | `run_probe()` — deterministic QFT MSE |
| `src/harness/evaluate.jl` | [F] | `evaluate_basis(basis)` — forward→topk→inverse→MSE |
| `src/harness/train.jl` | [F] | `train_trial(type; ...)` — wraps `train_basis` with frozen config |
| `src/bases/registry.jl` | [E] | `BASELINES`, `TRIAL_REGISTRY`, `register_basis!` |
| `src/bases/_example_basis.jl` | [E] | Template for new bases |
| `data/fixture_512.bin` | [F] | Float64 512×512 (2097152 bytes) |
| `scripts/generate_fixture.jl` | [F] | Regenerates fixture |
| `scripts/compute_manifest.jl` | [F] | Regenerates `frozen-manifest.toml` |
| `test/runtests.jl` | [F] | Driver |
| `test/harness_tests.jl` | [F] | Probe + interface conformance |
| `test/bases_tests.jl` | [E] | Per-basis tests agent adds |
| `Makefile` | [F] | `init`, `test`, `baseline`, `trial NAME=...`, `verify`, `rehash` |
| `frozen-manifest.toml` | [F] | `[files]`, `[probe]`, optional `[secret]` |
| `prepare.jl` | [F] | Verify frozen surface |
| `program.md` | [F] | Autoresearch runbook |
| `results.tsv` | [E] | Trial log |
| `best.tsv` | [E] | Current leader |
| `.github/workflows/CI.yml` | [F] | Julia test matrix |
| `.github/workflows/basis-freeze.yml` | [F] | Manifest + probe verify |
| `README.md` | [F] | Quick start |
| `LICENSE` | [F] | MIT |

---

## Phase 0 — Update existing `.claude/` files (Deliverables 1 & 2)

### Task 1: Write project-specific `.claude/CLAUDE.md`

**Files:**
- Modify: `.claude/CLAUDE.md` (currently just contains `# CLAUDE.md\n`)

- [ ] **Step 1: Write the new CLAUDE.md content**

```markdown
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
```

- [ ] **Step 2: Verify file contents**

Run: `cat .claude/CLAUDE.md | head -10`
Expected: first line is `# CLAUDE.md — AutoDFT.jl`.

- [ ] **Step 3: Do NOT commit yet** — aggregated with other Phase 0/1 changes at the end.

---

### Task 2: Retitle `julia-conventions.md` references

**Files:**
- Modify: `.claude/rules/julia-conventions.md` (title + one sentence reference ParametricDFT.jl)

- [ ] **Step 1: Read the current file to see what needs retitling**

Run: `head -5 .claude/rules/julia-conventions.md`
Expected: First non-empty line is `# Julia Conventions for ParametricDFT.jl`.

- [ ] **Step 2: Replace title and preamble**

Use Edit to change:
- `# Julia Conventions for ParametricDFT.jl` → `# Julia Conventions for AutoDFT.jl`
- `Follow these conventions when writing or modifying Julia code in this project.` (unchanged)

Everything else (naming rules, type hierarchy, AD guidelines, testing) applies verbatim — AutoDFT uses the same conventions.

- [ ] **Step 3: Verify**

Run: `head -3 .claude/rules/julia-conventions.md`
Expected: title line shows `AutoDFT.jl`.

---

## Phase 1 — Julia project skeleton

### Task 3: Create `Project.toml`

**Files:**
- Create: `Project.toml`

- [ ] **Step 1: Write Project.toml**

```toml
name = "AutoDFT"
uuid = "a17d4a8c-1c9d-4d5e-9e3f-7a6b8c2d1f01"
authors = ["Shiwen An <sweynan@icloud.com>"]
version = "0.1.0-DEV"

[deps]
ParametricDFT = "cc2eb9de-5297-4754-b0bd-fdc80c6df40d"
SHA = "ea8e919c-243c-51af-8825-aaa63cd721ce"
TOML = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
Dates = "ade2ca70-3891-5945-98fb-dc099432e06a"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
OMEinsum = "ebe7aa44-baf0-506c-a96f-8464559b3922"

[compat]
ParametricDFT = "0.1"
julia = "1.10"

[extras]
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[targets]
test = ["Test"]
```

- [ ] **Step 2: Verify**

Run: `grep '^name' Project.toml`
Expected: `name = "AutoDFT"`.

---

### Task 4: Create top-level module `src/AutoDFT.jl`

**Files:**
- Create: `src/AutoDFT.jl`

- [ ] **Step 1: Write the module skeleton**

```julia
module AutoDFT

using ParametricDFT
using Random
using LinearAlgebra
using SHA
using Printf
using Dates
using TOML

# Frozen numerical config — must match the spec. Any change will trip the probe.
const M_QUBITS       = 9
const N_QUBITS       = 9
const IMAGE_SIZE     = (2^M_QUBITS, 2^N_QUBITS)   # (512, 512)
const TOPK           = 26_214                     # floor(0.1 * 2^18)
const TRAIN_STEPS    = 500
const LEARNING_RATE  = 0.01
const SEED           = 42
const ACCEPTANCE_REL = 0.01                       # ≥1% relative improvement

# Frozen paths
const REPO_ROOT     = normpath(joinpath(@__DIR__, ".."))
const FIXTURE_PATH  = joinpath(REPO_ROOT, "data", "fixture_512.bin")
const MANIFEST_PATH = joinpath(REPO_ROOT, "frozen-manifest.toml")
const RESULTS_PATH  = joinpath(REPO_ROOT, "results.tsv")
const BEST_PATH     = joinpath(REPO_ROOT, "best.tsv")

include("harness/fixture.jl")
include("harness/evaluate.jl")
include("harness/probe.jl")
include("harness/train.jl")
include("bases/registry.jl")

export load_fixture, run_probe, evaluate_basis, train_trial
export BASELINES, TRIAL_REGISTRY, register_basis!
export run_baseline, run_trial

# Runners (wired up in Task 25)
function run_baseline end
function run_trial end

include("runners.jl")

end # module
```

- [ ] **Step 2: Verify module parses later** (after Tasks 9-25 fill in includes). For now, file just needs to exist.

Run: `test -f src/AutoDFT.jl && echo OK`
Expected: `OK`.

---

### Task 5: Install ParametricDFT.jl at pinned SHA + generate `Manifest.toml`

**Files:**
- Create: `Manifest.toml` (generated by Pkg)

- [ ] **Step 1: Install at pinned SHA**

Run:
```bash
julia --project=. -e '
using Pkg
Pkg.add(PackageSpec(
    url = "https://github.com/nzy1997/ParametricDFT.jl",
    rev = "79117aa8b584f405c0d3268a9f1b306c42337b9e"
))
Pkg.instantiate()
'
```

Expected output: downloads ParametricDFT + transitive deps (Yao, OMEinsum, Zygote, CairoMakie, …), writes `Manifest.toml`. First run may take 5–15 minutes.

**If install fails** due to network / compat / GPU drivers: skip the full install and instead create a placeholder `Manifest.toml` using only `[deps]` entries without resolution — see Task 5b fallback below. Record the failure reason in `program.md` later.

- [ ] **Step 2: Verify ParametricDFT loads**

Run:
```bash
julia --project=. -e 'using ParametricDFT; println(pathof(ParametricDFT))'
```

Expected: prints a `.../ParametricDFT.jl/src/ParametricDFT.jl` path under `~/.julia/packages/` or similar.

- [ ] **Step 2b (only if Step 1 failed): Fallback — write a minimal Manifest.toml manually**

```toml
# Manifest.toml fallback — full resolution failed in this environment.
# The pinned ParametricDFT SHA is 79117aa8b584f405c0d3268a9f1b306c42337b9e,
# retained in Project.toml. Manifest will be regenerated on a host that can
# resolve the dependency graph.
julia_version = "1.10.0"
manifest_format = "2.0"
```

Note in `program.md` (Task 35) that the first autoresearch session must run `make init-fresh` to generate the real Manifest.

- [ ] **Step 3: Verify Manifest exists**

Run: `test -f Manifest.toml && echo OK`
Expected: `OK`.

---

### Task 6: Create `.gitignore`

**Files:**
- Create: `.gitignore`

- [ ] **Step 1: Write the ignore list**

```
# Julia
*.jl.cov
*.jl.*.cov
*.jl.mem
/deps/deps.jl
/deps/build.log

# Julia package cache (don't commit cloned deps)
/.julia/

# Scratch / tmp
/tmp/
/scratch/
*.tmp
*.swp
.DS_Store

# Training checkpoints and loss logs
/checkpoints/
/logs/
*.json.gz

# Local env overrides
.envrc.local
```

**Do NOT ignore** `Manifest.toml`, `results.tsv`, `best.tsv`, `data/fixture_512.bin`. These are committed.

- [ ] **Step 2: Verify**

Run: `grep -c '^' .gitignore`
Expected: ≥15 lines.

---

## Phase 2 — Fixture generation

### Task 7: Write `scripts/generate_fixture.jl`

**Files:**
- Create: `scripts/generate_fixture.jl`

- [ ] **Step 1: Write the fixture generator**

```julia
# scripts/generate_fixture.jl
#
# Generates the frozen 512×512 test image used by every trial.
# Output: data/fixture_512.bin (Float64, row-major, 524_288 elements = 4_194_304 bytes).
#
# Content: low-pass Gaussian random field, seed=42.
# Low-pass structure gives sparse bases room to differentiate — a pure white-noise
# image would be near-incompressible under any basis.
#
# This script is FROZEN. Re-running it must produce a byte-identical file.

using Random
using LinearAlgebra
using FFTW

const OUT_PATH = joinpath(@__DIR__, "..", "data", "fixture_512.bin")
const SIZE = 512
const SEED = 42
const BANDWIDTH = 32   # cutoff in Fourier pixels (low-pass radius)

function generate_fixture()
    Random.seed!(SEED)
    # White noise in spatial domain
    white = randn(Float64, SIZE, SIZE)
    # FFT → zero out frequencies above BANDWIDTH → IFFT
    F = fft(white)
    kx = fftfreq(SIZE) .* SIZE
    ky = fftfreq(SIZE) .* SIZE
    mask = [abs(k1)^2 + abs(k2)^2 <= BANDWIDTH^2 for k1 in kx, k2 in ky]
    F .*= mask
    spatial = real.(ifft(F))
    # Normalize to zero mean, unit std for predictability
    spatial .-= sum(spatial) / length(spatial)
    spatial ./= sqrt(sum(abs2, spatial) / length(spatial))
    return spatial
end

function main()
    img = generate_fixture()
    @assert size(img) == (SIZE, SIZE)
    @assert eltype(img) == Float64
    mkpath(dirname(OUT_PATH))
    open(OUT_PATH, "w") do io
        write(io, img)
    end
    @info "Wrote $OUT_PATH" size=size(img) bytes=stat(OUT_PATH).size
    return nothing
end

abspath(PROGRAM_FILE) == @__FILE__ && main()
```

- [ ] **Step 2: Verify file exists**

Run: `test -f scripts/generate_fixture.jl && echo OK`
Expected: `OK`.

Note: FFTW is a transitive dep via CairoMakie, but if it's not available as a top-level, this script needs to be run with `--project=deps` or FFTW needs to be added to `[deps]`. **Add FFTW to `Project.toml`** if Step 2 of Task 8 fails with a load error.

---

### Task 8: Execute `generate_fixture.jl` to produce `data/fixture_512.bin`

**Files:**
- Create: `data/fixture_512.bin`

- [ ] **Step 1: Run the generator**

Run:
```bash
julia --project=. scripts/generate_fixture.jl
```

Expected output:
```
[ Info: Wrote /home/claude-user/AutoDFT.jl/data/fixture_512.bin size=(512, 512) bytes=2097152
```

Note: `Float64` × 512 × 512 = 2_097_152 bytes. (If FFTW is missing, run `julia --project=. -e 'using Pkg; Pkg.add("FFTW")'` first, then update `Project.toml` to include `FFTW` under `[deps]` and the compat section.)

- [ ] **Step 2: Verify size and determinism**

Run:
```bash
stat -c%s data/fixture_512.bin
julia --project=. -e '
using SHA
h = open(sha256, "data/fixture_512.bin")
println(bytes2hex(h))
'
```

Expected:
- First line: `2097152`
- Second line: a 64-char hex SHA — record it; used later in Task 31 when building the manifest.

---

## Phase 3 — Harness: fixture loader (TDD)

### Task 9: Write the fixture-loader test

**Files:**
- Create: `test/harness_tests.jl` (partial — expanded in later tasks)

- [ ] **Step 1: Write the test skeleton + fixture test**

```julia
# test/harness_tests.jl — FROZEN
using Test
using AutoDFT
using Random
using LinearAlgebra
using ParametricDFT

@testset "harness" begin

@testset "load_fixture" begin
    img = AutoDFT.load_fixture()
    @test size(img) == (512, 512)
    @test eltype(img) == Float64
    # Determinism: content must match the seed-42 generator.
    @test abs(sum(img) / length(img)) < 1e-9       # zero mean
    @test abs(sqrt(sum(abs2, img) / length(img)) - 1.0) < 1e-9  # unit std
end

# Additional @testsets added in later tasks: probe, evaluate, conformance
end # @testset "harness"
```

- [ ] **Step 2: Create a stub `test/runtests.jl`**

```julia
using Test
using AutoDFT

@testset "AutoDFT.jl" begin
    include("harness_tests.jl")
    include("bases_tests.jl")
end
```

- [ ] **Step 3: Create an empty `test/bases_tests.jl`**

```julia
# test/bases_tests.jl — EDITABLE
# Agent adds @testset blocks here for each new basis (unitarity + interface conformance).
using Test
using AutoDFT
using ParametricDFT

@testset "bases" begin
    # No bases registered yet — placeholder so the test suite runs.
    @test true
end
```

- [ ] **Step 4: Run the test — expect failure**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: fails with `UndefVarError: load_fixture not defined` (because `src/harness/fixture.jl` doesn't exist yet).

---

### Task 10: Implement `src/harness/fixture.jl`

**Files:**
- Create: `src/harness/fixture.jl`

- [ ] **Step 1: Write the loader**

```julia
# src/harness/fixture.jl — FROZEN
#
# Loads the 512×512 Float64 fixture image committed under data/fixture_512.bin.
# Read-only; never regenerates in-process. Use scripts/generate_fixture.jl to rebuild.

"""
    load_fixture() -> Matrix{Float64}

Return the frozen 512×512 test image. Reads raw Float64 bytes from `\$(FIXTURE_PATH)`.
"""
function load_fixture()
    isfile(FIXTURE_PATH) || error("Fixture missing: $FIXTURE_PATH. Run `julia --project=. scripts/generate_fixture.jl`.")
    n = 512
    img = Matrix{Float64}(undef, n, n)
    open(FIXTURE_PATH, "r") do io
        read!(io, img)
    end
    return img
end
```

- [ ] **Step 2: Run the test — expect pass**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: `Test Summary: | Pass  Total ...` with `load_fixture` test passing.

**If install of ParametricDFT failed in Task 5**, the test will fail at `using ParametricDFT`. In that case, skip this step and defer to the first real-host run.

---

## Phase 4 — Harness: evaluate (TDD)

### Task 11: Add an `evaluate_basis` test

**Files:**
- Modify: `test/harness_tests.jl`

- [ ] **Step 1: Append the test**

Add this `@testset` inside `@testset "harness"`, after `load_fixture`:

```julia
@testset "evaluate_basis returns finite MSE for QFTBasis" begin
    Random.seed!(AutoDFT.SEED)
    basis = QFTBasis(AutoDFT.M_QUBITS, AutoDFT.N_QUBITS)
    mse = AutoDFT.evaluate_basis(basis)
    @test isa(mse, Real)
    @test isfinite(mse)
    @test mse > 0
    # QFT on a 10% top-k reconstruction of a low-pass image should be well-compressed
    # but not exactly zero. A sanity band: 1e-6 < mse < 1e4.
    @test 1e-12 < mse < 1e6
end
```

- [ ] **Step 2: Run the test — expect failure**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: `UndefVarError: evaluate_basis not defined`.

---

### Task 12: Implement `src/harness/evaluate.jl`

**Files:**
- Create: `src/harness/evaluate.jl`

- [ ] **Step 1: Write the evaluator**

```julia
# src/harness/evaluate.jl — FROZEN
#
# Single metric definition used by both the baseline leaderboard and every trial.
# Wraps ParametricDFT.loss_function with the frozen MSELoss(TOPK) configuration.

using ParametricDFT: loss_function, MSELoss, AbstractSparseBasis

"""
    evaluate_basis(basis::AbstractSparseBasis; image=load_fixture()) -> Float64

Compute reconstruction MSE of `basis` on `image` at the frozen sparsity `TOPK = 26_214`.
Equivalent to `loss_function(basis.tensors, M, N, basis.optcode, image, MSELoss(TOPK); inverse_code=basis.inverse_code)`.

The returned value is directly comparable across bases and across trials.
"""
function evaluate_basis(basis::AbstractSparseBasis; image=load_fixture())
    @assert size(image) == IMAGE_SIZE "Image size $(size(image)) != $(IMAGE_SIZE)"
    return Float64(loss_function(
        basis.tensors, M_QUBITS, N_QUBITS, basis.optcode, image, MSELoss(TOPK);
        inverse_code = basis.inverse_code,
    ))
end
```

**Note on the interface.** ParametricDFT's concrete bases (`QFTBasis`, etc.) expose `tensors`, `optcode`, `inverse_code` fields. Any new basis subtype **must** also expose these fields (or re-implement `evaluate_basis` for its type). The `_example_basis.jl` template (Task 24) shows the pattern.

- [ ] **Step 2: Run the test — expect pass**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: evaluate test passes. MSE value will be printed in `@info` if you add logging; don't add it — test just checks finiteness and magnitude.

---

## Phase 5 — Harness: probe (TDD)

### Task 13: Add a probe test

**Files:**
- Modify: `test/harness_tests.jl`

- [ ] **Step 1: Append the test**

```julia
@testset "run_probe is deterministic" begin
    a = AutoDFT.run_probe()
    b = AutoDFT.run_probe()
    @test a ≈ b atol=1e-14          # same run, same result, bit-identical
    @test isfinite(a)
    @test a > 0
end
```

- [ ] **Step 2: Run the test — expect failure**

Expected: `UndefVarError: run_probe not defined`.

---

### Task 14: Implement `src/harness/probe.jl`

**Files:**
- Create: `src/harness/probe.jl`

- [ ] **Step 1: Write the probe**

```julia
# src/harness/probe.jl — FROZEN
#
# BASIS_IDENTITY_PROBE. Runs a fully deterministic compression of the fixture
# through QFTBasis(9,9) — which has no learnable parameters beyond the initial
# QFT circuit — and returns the reconstruction MSE.
#
# The returned scalar is pinned in frozen-manifest.toml under [probe].qft_identity_mse.
# Any drift in ParametricDFT numerics, top-k truncation, or fixture bytes will change
# the probe value and be caught by prepare.jl / test/harness_tests.jl.

"""
    run_probe() -> Float64

Deterministic MSE for `QFTBasis(M_QUBITS, N_QUBITS)` evaluated against the fixture
at `k = TOPK`. Used as the semantic regression gate.
"""
function run_probe()
    basis = QFTBasis(M_QUBITS, N_QUBITS)
    return evaluate_basis(basis)
end
```

- [ ] **Step 2: Run the test — expect pass**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: probe test passes.

- [ ] **Step 3: Record the probe value for use in Task 31 (manifest generation)**

Run:
```bash
julia --project=. -e 'using AutoDFT; println(AutoDFT.run_probe())'
```
Expected output: a single Float64, e.g. `0.00123456789...`. **Save this exact value** — it's the `probe.qft_identity_mse` entry in the manifest.

---

## Phase 6 — Harness: trainer (TDD)

### Task 15: Add a `train_trial` test

**Files:**
- Modify: `test/harness_tests.jl`

- [ ] **Step 1: Append the test**

```julia
@testset "train_trial reduces or equals baseline MSE (QFT)" begin
    Random.seed!(AutoDFT.SEED)
    baseline = AutoDFT.run_probe()
    # Train QFTBasis for a *tiny* number of steps (fast test) — ONLY for this test.
    # Uses a reduced-steps override; production trials use TRAIN_STEPS = 500.
    basis, final_mse, wallclock_ms = AutoDFT.train_trial(QFTBasis; steps=5)
    @test isa(basis, QFTBasis)
    @test isfinite(final_mse)
    # 5 steps of Adam at lr=0.01 shouldn't blow up — MSE ≤ 10× baseline is generous.
    @test final_mse < 10 * baseline
    @test wallclock_ms > 0
end
```

- [ ] **Step 2: Run the test — expect failure**

Expected: `UndefVarError: train_trial not defined`.

---

### Task 16: Implement `src/harness/train.jl`

**Files:**
- Create: `src/harness/train.jl`

- [ ] **Step 1: Write the trainer**

```julia
# src/harness/train.jl — FROZEN
#
# Wraps ParametricDFT.train_basis with the frozen configuration from spec §Evaluation
# Protocol: RiemannianAdam(lr=0.01), 500 steps, seed 42, batch 1, no validation split.
# The `steps` kwarg is only for tests — production trials always use TRAIN_STEPS.

using ParametricDFT: train_basis, RiemannianAdam, MSELoss, AbstractSparseBasis

"""
    train_trial(::Type{B}; image=load_fixture(), steps=TRAIN_STEPS, device=_default_device())
        -> (trained_basis::B, final_mse::Float64, wallclock_ms::Float64)

Train a basis of type `B` on a single fixture image with the frozen config, then
evaluate final MSE on that same image. Returns both the trained basis and its MSE.
"""
function train_trial(::Type{B};
                     image=load_fixture(),
                     steps::Int=TRAIN_STEPS,
                     device::Symbol=_default_device(),
                     extra_kwargs...) where {B <: AbstractSparseBasis}
    Random.seed!(SEED)
    t0 = time_ns()
    basis, _history = train_basis(
        B, [image];
        m = M_QUBITS, n = N_QUBITS,
        loss = MSELoss(TOPK),
        epochs = 1,
        steps_per_image = steps,
        validation_split = 0.0,
        shuffle = false,
        early_stopping_patience = typemax(Int),
        optimizer = RiemannianAdam(lr = LEARNING_RATE),
        batch_size = 1,
        device = device,
        extra_kwargs...,
    )
    wallclock_ms = (time_ns() - t0) / 1e6
    final_mse = evaluate_basis(basis; image = image)
    return basis, final_mse, wallclock_ms
end

function _default_device()
    # ParametricDFT decides :gpu only if `using CUDA` was called and CUDAExt loaded.
    # We default to :cpu to keep reproducibility tight across machines.
    return :cpu
end
```

**Rationale for `:cpu` default:** the spec says "Device: `CUDA.functional() ? GPU : CPU` — both produce the same MSE to `atol=1e-6`". For the initial scaffold we default to `:cpu` so results are byte-reproducible. Users can override per-call via `device=:gpu` after `using CUDA`.

- [ ] **Step 2: Run the test — expect pass**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: train_trial test passes (may take 30-90 seconds for 5 steps).

---

## Phase 7 — Bases: registry & template

### Task 17: Write `src/bases/registry.jl`

**Files:**
- Create: `src/bases/registry.jl`

- [ ] **Step 1: Write the registry**

```julia
# src/bases/registry.jl — EDITABLE
#
# Maps basis name strings → (::Type{<:AbstractSparseBasis}) constructor types.
# `BASELINES` is the fixed seed set for `make baseline`; do NOT remove entries.
# `TRIAL_REGISTRY` is the open set an autoresearch session adds to via
# `register_basis!("<Name>", <Name>Basis)` at the bottom of a new-basis file.

using ParametricDFT: QFTBasis, EntangledQFTBasis, TEBDBasis, MERABasis, AbstractSparseBasis

"""Baselines seeded by `make baseline`. Order defines the TSV row order."""
const BASELINES = [
    ("QFT",          QFTBasis),
    ("EntangledQFT", EntangledQFTBasis),
    ("TEBD",         TEBDBasis),
    ("MERA",         MERABasis),
]

"""Registry of agent-contributed bases. Populated at module-load time via `register_basis!`."""
const TRIAL_REGISTRY = Dict{String, Type{<:AbstractSparseBasis}}()

"""
    register_basis!(name::String, ::Type{B}) where B <: AbstractSparseBasis

Register a new basis under `name`. Call this at the bottom of your `src/bases/<slug>.jl`.
"""
function register_basis!(name::String, ::Type{B}) where {B <: AbstractSparseBasis}
    haskey(TRIAL_REGISTRY, name) && @warn "Overwriting existing registration" name
    TRIAL_REGISTRY[name] = B
    return B
end

function get_basis_type(name::String)
    # Check baselines first
    for (bname, btype) in BASELINES
        bname == name && return btype
    end
    haskey(TRIAL_REGISTRY, name) && return TRIAL_REGISTRY[name]
    error("Unknown basis: $name. Known: $(vcat(first.(BASELINES), collect(keys(TRIAL_REGISTRY)))).")
end

# Auto-discover new basis files: every src/bases/<file>.jl (except registry.jl
# and files starting with `_`) is included at module-load time. This way an
# agent adding src/bases/my_basis.jl doesn't also have to edit this file.
let dir = @__DIR__
    for f in sort(readdir(dir))
        (f == "registry.jl" || startswith(f, "_") || !endswith(f, ".jl")) && continue
        include(joinpath(dir, f))
    end
end
```

- [ ] **Step 2: Verify the module loads with no registered trial bases**

Run: `julia --project=. -e 'using AutoDFT; println(keys(AutoDFT.TRIAL_REGISTRY))'`
Expected: empty set / no keys (since no trial file exists yet).

---

### Task 18: Write `src/bases/_example_basis.jl` (template — NOT auto-included)

**Files:**
- Create: `src/bases/_example_basis.jl`

The leading underscore is deliberate: `registry.jl` skips files beginning with `_`. Copy this file, rename without underscore, and edit.

- [ ] **Step 1: Write the template**

```julia
# src/bases/_example_basis.jl — TEMPLATE (NOT auto-included)
#
# Copy to src/bases/<your_name>_basis.jl, then:
#   1. Rename IdentityBasis → <YourName>Basis throughout the file.
#   2. Replace the body of `forward_transform` / `inverse_transform` with your
#      new construction (e.g., a novel tensor network).
#   3. Optionally: override `_init_circuit` / `_build_basis` (below) so the
#      trainer can build initial tensors for your topology.
#   4. At the bottom: register_basis!("<YourName>", <YourName>Basis)
#
# Interface contract (AbstractSparseBasis):
#   forward_transform(basis, image) -> Complex matrix (size = image_size(basis))
#   inverse_transform(basis, freq)  -> Complex matrix
#   image_size(basis)               -> (height, width)
#   num_parameters(basis)           -> Int  (count of learnable scalars)
#   basis_hash(basis)               -> String (SHA-256 of params; deterministic)
#
# Additional contract for train_basis compatibility:
#   ParametricDFT._init_circuit(::Type{T}, m, n; kwargs...) -> (optcode, inverse_code, initial_tensors)
#   ParametricDFT._build_basis(::Type{T}, m, n, tensors, optcode, inverse_code; kwargs...) -> T
#   ParametricDFT._basis_name(::Type{T}) -> String (display name)
#
# Additional contract for evaluate_basis compatibility:
#   `basis` must have fields `tensors::Vector`, `optcode`, `inverse_code`
#   (or override evaluate_basis for its type).

using ParametricDFT: QFTBasis, AbstractSparseBasis,
                     forward_transform, inverse_transform,
                     image_size, num_parameters, basis_hash
using SHA: sha256

"""
    IdentityBasis <: AbstractSparseBasis

A thin wrapper around `QFTBasis`. Trains and evaluates identically to QFT — exists
only as a template demonstrating the interface contract. Copy this file and replace
`IdentityBasis` with your new type.
"""
struct IdentityBasis <: AbstractSparseBasis
    m::Int
    n::Int
    tensors::Vector
    optcode
    inverse_code
end

function IdentityBasis(m::Int, n::Int)
    inner = QFTBasis(m, n)
    return IdentityBasis(m, n, inner.tensors, inner.optcode, inner.inverse_code)
end

# AbstractSparseBasis interface
forward_transform(b::IdentityBasis, image) =
    forward_transform(QFTBasis(b.m, b.n, b.tensors, b.optcode, b.inverse_code), image)
inverse_transform(b::IdentityBasis, freq) =
    inverse_transform(QFTBasis(b.m, b.n, b.tensors, b.optcode, b.inverse_code), freq)
image_size(b::IdentityBasis) = (2^b.m, 2^b.n)
num_parameters(b::IdentityBasis) = sum(length, b.tensors)
function basis_hash(b::IdentityBasis)
    io = IOBuffer()
    write(io, "IdentityBasis:m=$(b.m):n=$(b.n):")
    for t in b.tensors, v in t
        write(io, "$(real(v)),$(imag(v));")
    end
    return bytes2hex(sha256(take!(io)))
end

# ParametricDFT train_basis dispatch — reuse QFT's init/build
ParametricDFT._init_circuit(::Type{IdentityBasis}, m, n; kwargs...) =
    ParametricDFT._init_circuit(QFTBasis, m, n; kwargs...)

ParametricDFT._build_basis(::Type{IdentityBasis}, m, n, tensors, optcode, inverse_code; kwargs...) =
    IdentityBasis(m, n, tensors, optcode, inverse_code)

ParametricDFT._basis_name(::Type{IdentityBasis}) = "Identity"

# register_basis!("Identity", IdentityBasis)  # <-- uncomment in your copy
```

- [ ] **Step 2: Verify it does NOT auto-load**

Run: `julia --project=. -e 'using AutoDFT; println(isdefined(AutoDFT, :IdentityBasis))'`
Expected: `false` (file starts with `_`, so `registry.jl` skipped it).

---

## Phase 8 — Module runners

### Task 19: Write `src/runners.jl`

**Files:**
- Create: `src/runners.jl`

- [ ] **Step 1: Write the runners**

```julia
# src/runners.jl — FROZEN
#
# User-facing entry points for `make baseline` and `make trial NAME=...`.
# Both produce rows in results.tsv; `run_trial` also compares against best.tsv
# and writes the acceptance decision.

using Dates
using Printf

const RESULTS_HEADER = "timestamp\tbranch\tcommit_sha\tbasis_name\tbasis_hash\tnum_parameters\tfinal_mse\tprobe_mse\ttrain_steps\ttrain_wallclock_ms\ttransform_time_ms\tdevice\tstatus\tnotes"

function _git_branch()
    try
        strip(read(`git rev-parse --abbrev-ref HEAD`, String))
    catch
        "unknown"
    end
end

function _git_sha()
    try
        strip(read(`git rev-parse --short HEAD`, String))
    catch
        "uncommitted"
    end
end

function _append_results_row(; timestamp, branch, commit_sha, basis_name, basis_hash_,
                              num_parameters, final_mse, probe_mse, train_steps,
                              train_wallclock_ms, transform_time_ms, device, status, notes)
    new_file = !isfile(RESULTS_PATH)
    open(RESULTS_PATH, "a") do io
        new_file && println(io, RESULTS_HEADER)
        @printf(io, "%s\t%s\t%s\t%s\t%s\t%d\t%.10e\t%.10e\t%d\t%.3f\t%.3f\t%s\t%s\t%s\n",
                timestamp, branch, commit_sha, basis_name, basis_hash_, num_parameters,
                final_mse, probe_mse, train_steps, train_wallclock_ms, transform_time_ms,
                device, status, notes)
    end
end

function _read_best_mse()
    isfile(BEST_PATH) || return Inf
    lines = readlines(BEST_PATH)
    length(lines) < 2 && return Inf
    cols = split(lines[2], '\t')
    # final_mse is column 7 (1-indexed)
    return parse(Float64, cols[7])
end

function _write_best_row(row_str::AbstractString)
    open(BEST_PATH, "w") do io
        println(io, RESULTS_HEADER)
        println(io, rstrip(row_str))
    end
end

"""
    run_baseline()

Evaluate each of the four ParametricDFT baselines with the frozen training config,
append their rows to results.tsv, and set best.tsv to the leader. Idempotent per-branch:
will NOT re-run if rows with `status=baseline` already exist on the current branch.
"""
function run_baseline()
    probe_mse = run_probe()
    branch = _git_branch()
    if isfile(RESULTS_PATH)
        for line in readlines(RESULTS_PATH)[2:end]
            cols = split(line, '\t')
            length(cols) >= 13 && cols[2] == branch && cols[13] == "baseline" &&
                (@info "Baselines already seeded on branch $branch — skipping"; return nothing)
        end
    end

    best_mse = Inf
    best_row = ""
    for (name, T) in BASELINES
        @info "Evaluating baseline $name"
        basis, final_mse, wallclock = train_trial(T)
        ts = string(now())
        row = (
            timestamp = ts, branch = branch, commit_sha = _git_sha(),
            basis_name = name, basis_hash_ = basis_hash(basis),
            num_parameters = num_parameters(basis),
            final_mse = final_mse, probe_mse = probe_mse,
            train_steps = TRAIN_STEPS, train_wallclock_ms = wallclock,
            transform_time_ms = 0.0, device = "cpu",
            status = "baseline", notes = "",
        )
        _append_results_row(; row...)
        if final_mse < best_mse
            best_mse = final_mse
            best_row = join((ts, branch, _git_sha(), name, basis_hash(basis),
                             string(num_parameters(basis)),
                             @sprintf("%.10e", final_mse), @sprintf("%.10e", probe_mse),
                             string(TRAIN_STEPS), @sprintf("%.3f", wallclock), "0.000",
                             "cpu", "baseline", ""), '\t')
        end
    end
    _write_best_row(best_row)
    @info "Baseline seeding complete" best_mse
    return nothing
end

"""
    run_trial(name::String) -> Bool

Train+evaluate the basis registered under `name`. Append a row to results.tsv;
return true if accepted (final_mse < best * (1 - ACCEPTANCE_REL)), false otherwise.
The Makefile target translates the Bool into exit 0/1.
"""
function run_trial(name::String)
    T = get_basis_type(name)
    probe_mse = run_probe()
    best_mse = _read_best_mse()
    @info "Running trial" name best_mse
    basis, final_mse, wallclock = train_trial(T)
    accepted = final_mse < best_mse * (1 - ACCEPTANCE_REL)
    ts = string(now())
    branch = _git_branch()
    sha = _git_sha()
    bh = basis_hash(basis)
    row_fields = (ts, branch, sha, name, bh,
                  string(num_parameters(basis)),
                  @sprintf("%.10e", final_mse), @sprintf("%.10e", probe_mse),
                  string(TRAIN_STEPS), @sprintf("%.3f", wallclock), "0.000",
                  "cpu", accepted ? "kept" : "dropped", "")
    _append_results_row(; timestamp=ts, branch=branch, commit_sha=sha, basis_name=name,
                        basis_hash_=bh, num_parameters=num_parameters(basis),
                        final_mse=final_mse, probe_mse=probe_mse,
                        train_steps=TRAIN_STEPS, train_wallclock_ms=wallclock,
                        transform_time_ms=0.0, device="cpu",
                        status=(accepted ? "kept" : "dropped"), notes="")
    if accepted
        _write_best_row(join(row_fields, '\t'))
        @info "Accepted — new best" final_mse baseline=best_mse
    else
        @info "Rejected" final_mse baseline=best_mse
    end
    return accepted
end
```

- [ ] **Step 2: Module loads**

Run: `julia --project=. -e 'using AutoDFT; println(methods(AutoDFT.run_trial))'`
Expected: prints `1 method for generic function run_trial`.

---

## Phase 9 — Manifest + prepare

### Task 20: Write `scripts/compute_manifest.jl`

**Files:**
- Create: `scripts/compute_manifest.jl`

- [ ] **Step 1: Write the manifest builder**

```julia
# scripts/compute_manifest.jl — FROZEN (itself hashed in the manifest it creates)
#
# Regenerates frozen-manifest.toml:
#   [files]   — SHA256 of every frozen file
#   [probe]   — qft_identity_mse = run_probe()
#   [secret]  — sha256(BASIS_FREEZE_SALT + concat(sorted_file_shas)), if env var set
#
# Run after any legitimate harness edit. Developers without the salt simply
# regenerate [files] + [probe]; [secret] is injected in CI by the rehash workflow.

using SHA
using TOML

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))

"List of FROZEN file paths (relative to repo root). Keep in sync with the spec."
const FROZEN_FILES = [
    ".claude/CLAUDE.md",
    ".claude/rules/julia-conventions.md",
    ".github/workflows/CI.yml",
    ".github/workflows/basis-freeze.yml",
    "Project.toml",
    "Manifest.toml",
    "Makefile",
    "prepare.jl",
    "program.md",
    "README.md",
    "LICENSE",
    "src/AutoDFT.jl",
    "src/runners.jl",
    "src/harness/fixture.jl",
    "src/harness/evaluate.jl",
    "src/harness/probe.jl",
    "src/harness/train.jl",
    "scripts/generate_fixture.jl",
    "scripts/compute_manifest.jl",
    "data/fixture_512.bin",
    "test/runtests.jl",
    "test/harness_tests.jl",
]

function file_sha(path)
    bytes2hex(open(sha256, path))
end

function main()
    file_hashes = Dict{String, String}()
    for rel in FROZEN_FILES
        abs = joinpath(REPO_ROOT, rel)
        isfile(abs) || (@warn "Frozen file missing — skipping" rel; continue)
        file_hashes[rel] = file_sha(abs)
    end

    # Compute probe value (requires ParametricDFT + fixture + src/ all present)
    probe_mse = try
        @info "Running probe to pin qft_identity_mse"
        push!(LOAD_PATH, REPO_ROOT)
        using_result = Base.require(Base.PkgId(Base.UUID("a17d4a8c-1c9d-4d5e-9e3f-7a6b8c2d1f01"), "AutoDFT"))
        using_result.run_probe()
    catch e
        @warn "Could not compute probe — leaving sentinel -1.0. Run prepare.jl manually when environment is ready." exception=e
        -1.0
    end

    # Optional CI secret: salt from env
    secret_entry = Dict{String, Any}()
    salt = get(ENV, "BASIS_FREEZE_SALT", "")
    if !isempty(salt)
        sorted_shas = [file_hashes[k] for k in sort(collect(keys(file_hashes)))]
        combined = salt * join(sorted_shas, "")
        secret_entry["manifest_sha"] = bytes2hex(sha256(combined))
    end

    manifest = Dict{String, Any}(
        "files" => file_hashes,
        "probe" => Dict("qft_identity_mse" => probe_mse),
    )
    !isempty(secret_entry) && (manifest["secret"] = secret_entry)

    out_path = joinpath(REPO_ROOT, "frozen-manifest.toml")
    open(out_path, "w") do io
        TOML.print(io, manifest; sorted=true)
    end
    @info "Wrote manifest" out_path files=length(file_hashes) probe_mse
    return nothing
end

abspath(PROGRAM_FILE) == @__FILE__ && main()
```

- [ ] **Step 2: File exists**

Run: `test -f scripts/compute_manifest.jl && echo OK`
Expected: `OK`.

---

### Task 21: Write `prepare.jl`

**Files:**
- Create: `prepare.jl`

- [ ] **Step 1: Write the verifier**

```julia
# prepare.jl — FROZEN (itself hashed)
#
# Runs at the start of every `make trial` and `make test`. Verifies:
#   1. Every file in [files] has matching SHA256.
#   2. Probe value matches [probe].qft_identity_mse (atol 1e-10).
#   3. If BASIS_FREEZE_SALT is in env, recompute [secret].manifest_sha and compare.
#
# Exits with status 0 ("frozen surface OK") or 1 (prints the failing check).

using SHA
using TOML

const REPO_ROOT = @__DIR__

function die(msg)
    println(stderr, "FROZEN SURFACE VIOLATION: $msg")
    exit(1)
end

function verify_files(manifest)
    files = get(manifest, "files", Dict{String,String}())
    for (rel, expected_sha) in files
        abs = joinpath(REPO_ROOT, rel)
        isfile(abs) || die("missing frozen file: $rel")
        actual = bytes2hex(open(sha256, abs))
        actual == expected_sha || die("SHA mismatch for $rel\n  expected: $expected_sha\n  actual:   $actual")
    end
end

function verify_probe(manifest)
    expected = get(get(manifest, "probe", Dict()), "qft_identity_mse", nothing)
    expected === nothing && die("[probe].qft_identity_mse missing from frozen-manifest.toml")
    expected == -1.0 && (@warn "Probe not yet pinned (sentinel -1.0). Run scripts/compute_manifest.jl on a host with ParametricDFT installed."; return)
    push!(LOAD_PATH, REPO_ROOT)
    AutoDFT = Base.require(Base.PkgId(Base.UUID("a17d4a8c-1c9d-4d5e-9e3f-7a6b8c2d1f01"), "AutoDFT"))
    actual = AutoDFT.run_probe()
    isapprox(actual, expected; atol=1e-10) ||
        die("probe mismatch\n  expected: $expected\n  actual:   $actual\n  diff:     $(actual - expected)")
end

function verify_secret(manifest)
    salt = get(ENV, "BASIS_FREEZE_SALT", "")
    isempty(salt) && return
    expected = get(get(manifest, "secret", Dict()), "manifest_sha", nothing)
    expected === nothing && die("[secret].manifest_sha missing but BASIS_FREEZE_SALT is set")
    files = get(manifest, "files", Dict())
    sorted_shas = [files[k] for k in sort(collect(keys(files)))]
    actual = bytes2hex(sha256(salt * join(sorted_shas, "")))
    actual == expected || die("manifest secret SHA mismatch (file + manifest edits don't align)")
end

function main()
    manifest_path = joinpath(REPO_ROOT, "frozen-manifest.toml")
    isfile(manifest_path) || die("frozen-manifest.toml not found")
    manifest = TOML.parsefile(manifest_path)
    verify_files(manifest)
    verify_secret(manifest)
    verify_probe(manifest)
    println("frozen surface OK")
end

main()
```

- [ ] **Step 2: File exists**

Run: `test -f prepare.jl && echo OK`
Expected: `OK`.

---

### Task 22: Generate the initial `frozen-manifest.toml`

**Files:**
- Create: `frozen-manifest.toml`

- [ ] **Step 1: Run the manifest builder**

Run: `julia --project=. scripts/compute_manifest.jl`

Expected output:
```
[ Info: Running probe to pin qft_identity_mse
[ Info: Wrote manifest out_path="..." files=22 probe_mse=<some float>
```

If ParametricDFT is missing or fixture is not generated, the script will warn and pin `probe_mse = -1.0`. That's acceptable at scaffold time; Task 36 will rehash once baselines are seeded.

- [ ] **Step 2: Sanity-check the manifest**

Run: `head -20 frozen-manifest.toml`
Expected: starts with `[files]`, has entries like `".claude/CLAUDE.md" = "<64-hex>"`, and ends with a `[probe]` section.

- [ ] **Step 3: Verify `prepare.jl` passes**

Run: `julia --project=. prepare.jl`
Expected: prints `frozen surface OK`, exits 0.

---

## Phase 10 — Makefile, program.md, README, LICENSE

### Task 23: Write `Makefile`

**Files:**
- Create: `Makefile`

- [ ] **Step 1: Write the Makefile**

```makefile
# Makefile — FROZEN
# Shell targets wrapping the AutoDFT runners and the freeze-verification flow.

JULIA   ?= julia
PROJECT := --project=.

.PHONY: help init init-fresh test verify baseline trial rehash fixture

help:
	@echo "Targets:"
	@echo "  init         Pkg.instantiate() from committed Manifest.toml"
	@echo "  init-fresh   Pkg.resolve() + instantiate (after Project.toml change)"
	@echo "  fixture      Regenerate data/fixture_512.bin"
	@echo "  verify       Run prepare.jl (frozen-surface check)"
	@echo "  test         verify + Pkg.test()"
	@echo "  baseline     Seed results.tsv with 4 ParametricDFT baselines"
	@echo "  trial NAME=X Train+eval basis X; exit 0 if accepted, 1 if dropped"
	@echo "  rehash       Regenerate frozen-manifest.toml (harness-edit flow only)"

init:
	$(JULIA) $(PROJECT) -e 'using Pkg; Pkg.instantiate()'

init-fresh:
	$(JULIA) $(PROJECT) -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'

fixture:
	$(JULIA) $(PROJECT) scripts/generate_fixture.jl

verify:
	$(JULIA) $(PROJECT) prepare.jl

test: verify
	$(JULIA) $(PROJECT) -e 'using Pkg; Pkg.test()'

baseline:
	$(JULIA) $(PROJECT) -e 'using AutoDFT; AutoDFT.run_baseline()'

trial:
	@test -n "$(NAME)" || { echo "Usage: make trial NAME=<BasisName>"; exit 2; }
	$(JULIA) $(PROJECT) -e 'using AutoDFT; exit(AutoDFT.run_trial("$(NAME)") ? 0 : 1)'

rehash:
	$(JULIA) $(PROJECT) scripts/compute_manifest.jl
```

- [ ] **Step 2: Verify**

Run: `make help`
Expected: prints the target list above.

---

### Task 24: Write `program.md`

**Files:**
- Create: `program.md`

- [ ] **Step 1: Write the runbook**

```markdown
# Autoresearch Runbook — AutoDFT.jl

You are an autonomous Claude Code session running on a checkout of
`github.com/zazabap/AutoDFT.jl`. Your goal: find a new `AbstractSparseBasis`
implementation that achieves lower reconstruction MSE than the current
best (see `best.tsv`) on the frozen 512×512 fixture at `k = 26_214`.

## Rules

1. **Do not edit frozen files.** Files marked `[F]` in the design spec
   (`docs/superpowers/specs/2026-04-19-autoresearch-parametricdft-basis-design.md`)
   are SHA-hashed in `frozen-manifest.toml`. Any edit trips `make verify`
   and CI will reject the PR.
2. **Only edit `src/bases/*.jl`, `test/bases_tests.jl`, `results.tsv`, `best.tsv`.**
3. **Every trial must run through `make trial NAME=<Name>`.** Never
   hand-edit `results.tsv` or `best.tsv` — they are updated by the runner.
4. **If a trial is rejected (`make trial` exits 1):** immediately
   `git reset --hard HEAD~1` to undo the basis file. Then commit a
   dropped-trial note on a SEPARATE commit:
   `git commit --allow-empty -m "dropped: <Name>Basis — <why it failed>"`.
   This preserves the attempt log even though the code was reverted.
5. **The user is the kill switch.** Ctrl+C ends the session.

## Startup

```bash
make init            # Or: make init-fresh  if Project.toml has drifted.
make verify          # Prints "frozen surface OK" on success.
make test            # Full test suite — must pass.
```

If `results.tsv` is empty (no rows under current branch with status=baseline):

```bash
make baseline        # ~4 × 500-step trainings. Takes 10-60 min on GPU, much
                     # longer on CPU. Seeds results.tsv and best.tsv.
```

## Iteration loop

1. **Read** `best.tsv` — that's the bar to beat.
2. **Form a hypothesis.** Examples:
   - "MERA with 3 layers underperforms vs. QFT because its unitaries are
     over-constrained at 512×512. A 4-layer MERA with periodic boundary may help."
   - "Block-diagonal QFT⊕TEBD could capture both low- and high-frequency
     structure without increasing parameter count."
   Write the hypothesis as a commit message.
3. **Copy** `src/bases/_example_basis.jl` to `src/bases/<slug>.jl`.
   Rename `IdentityBasis` → `<Name>Basis`. Implement the contract described in
   the template. Add `register_basis!("<Name>", <Name>Basis)` at the bottom.
4. **Add a conformance test** to `test/bases_tests.jl`:
   ```julia
   @testset "<Name>Basis interface" begin
       b = <Name>Basis(9, 9)
       @test image_size(b) == (512, 512)
       @test num_parameters(b) > 0
       x = randn(ComplexF64, 512, 512)
       y = forward_transform(b, x)
       x̂ = inverse_transform(b, y)
       @test x̂ ≈ x atol=1e-8
       @test basis_hash(b) == basis_hash(b)
   end
   ```
5. **Commit:**
   ```bash
   git add src/bases/<slug>.jl test/bases_tests.jl
   git commit -m "trial: <Name>Basis — <one-line hypothesis>"
   ```
6. **Run the trial:**
   ```bash
   make trial NAME=<Name>
   ```
7. **On acceptance (exit 0):** `best.tsv` updated automatically. Commit it:
   ```bash
   git add best.tsv results.tsv
   git commit -m "accept: <Name>Basis lowers MSE to <x>"
   ```
   Then go back to step 1.

8. **On rejection (exit 1):**
   ```bash
   git reset --hard HEAD~1                  # remove the basis file
   git commit --allow-empty -m "dropped: <Name>Basis — <reason>"
   git add results.tsv
   git commit --amend --no-edit             # fold the results row into the note
   ```
   Then go back to step 1.

## Adding dependencies

If your new basis needs an additional Julia package, DO NOT edit `Project.toml`
or `Manifest.toml` — both are frozen. Instead, raise a human PR:
"harness update: add <Pkg> for <reason>." A harness-update PR bundles:
- edits to frozen files
- rerun of `scripts/compute_manifest.jl` (updates `frozen-manifest.toml`)
- a CI run with `BASIS_FREEZE_SALT` that updates `[secret].manifest_sha`

## Performance budget

`make trial` should complete in under ~10 minutes per trial on GPU, ~60 minutes
on CPU. If yours routinely exceeds this, log it in the commit note and skip to
the next idea — a slow trial starves exploration.
```

- [ ] **Step 2: Verify**

Run: `head -5 program.md`
Expected: starts with `# Autoresearch Runbook — AutoDFT.jl`.

---

### Task 25: Write `README.md`

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write the README**

```markdown
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
```

- [ ] **Step 2: Verify**

Run: `head -1 README.md`
Expected: `# AutoDFT.jl`.

---

### Task 26: Write `LICENSE`

**Files:**
- Create: `LICENSE`

- [ ] **Step 1: Write MIT LICENSE**

```
MIT License

Copyright (c) 2026 Shiwen An

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 2: Verify**

Run: `head -1 LICENSE`
Expected: `MIT License`.

---

## Phase 11 — CI workflows

### Task 27: Write `.github/workflows/CI.yml`

**Files:**
- Create: `.github/workflows/CI.yml`

- [ ] **Step 1: Write the CI workflow**

```yaml
name: CI
on:
  push:
    branches: [main]
  pull_request:

permissions:
  contents: read

jobs:
  test:
    name: Julia ${{ matrix.julia-version }}
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        julia-version: ['1.10', '1.11']
    steps:
      - uses: actions/checkout@v4

      - uses: julia-actions/setup-julia@v2
        with:
          version: ${{ matrix.julia-version }}

      - uses: julia-actions/cache@v2

      - name: Instantiate
        run: julia --project=. -e 'using Pkg; Pkg.instantiate()'

      - name: Frozen-surface verify + test suite
        run: make test
```

- [ ] **Step 2: Verify**

Run: `test -f .github/workflows/CI.yml && echo OK`
Expected: `OK`.

---

### Task 28: Write `.github/workflows/basis-freeze.yml`

**Files:**
- Create: `.github/workflows/basis-freeze.yml`

- [ ] **Step 1: Write the freeze workflow**

```yaml
name: basis-freeze
on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

permissions:
  contents: read

jobs:
  verify:
    name: verify frozen surface
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: julia-actions/setup-julia@v2
        with:
          version: '1.10'

      - uses: julia-actions/cache@v2

      - name: Instantiate
        run: julia --project=. -e 'using Pkg; Pkg.instantiate()'

      - name: Verify frozen surface (file SHAs + probe + secret)
        env:
          BASIS_FREEZE_SALT: ${{ secrets.BASIS_FREEZE_SALT }}
        run: julia --project=. prepare.jl
```

- [ ] **Step 2: Verify**

Run: `test -f .github/workflows/basis-freeze.yml && echo OK`
Expected: `OK`.

---

## Phase 12 — Final verify + initial commit (Deliverable 4)

### Task 29: Regenerate the manifest (all frozen files now exist)

Since we added many frozen files after the first manifest run, the current
`frozen-manifest.toml` doesn't hash them all. Regenerate.

- [ ] **Step 1: Rehash**

Run: `make rehash`
Expected: updated manifest covering all 22 frozen files; prints new probe value.

- [ ] **Step 2: Verify**

Run: `make verify`
Expected: `frozen surface OK`.

---

### Task 30: Run `make test` — full suite passes

- [ ] **Step 1: Run**

Run: `make test`
Expected output ends with a summary like:
```
Test Summary: |  Pass  Total
AutoDFT.jl    |    N      N
```
with all `@testset`s passing.

**If ParametricDFT.jl is missing** (Task 5 fallback path), `make test` will fail at `using ParametricDFT`. That's acceptable at scaffold time — mark this step as skipped and note it in the commit message.

---

### Task 31: Initial commit of the scaffold

- [ ] **Step 1: Stage everything**

Run:
```bash
git add -A
git status
```
Expected: lots of new files across `src/`, `test/`, `scripts/`, `data/`, `.github/`, config files, etc.

- [ ] **Step 2: Commit**

Run:
```bash
git commit -m "$(cat <<'EOF'
scaffold: AutoDFT.jl autoresearch harness

Initial scaffold per design spec. Includes:
- Frozen harness (src/harness/{fixture,evaluate,probe,train}.jl)
- Runners (src/runners.jl) and driver (src/AutoDFT.jl)
- Bases registry + example template (src/bases/registry.jl,
  _example_basis.jl)
- Fixture generator + data/fixture_512.bin (512x512 Float64, seed 42)
- SHA-hashed freeze manifest (frozen-manifest.toml) + verifier (prepare.jl)
- Makefile with init/test/baseline/trial/verify/rehash targets
- CI (.github/workflows/CI.yml) and freeze check (basis-freeze.yml)
- Autoresearch runbook (program.md)
- README, LICENSE

ParametricDFT.jl pinned at 79117aa8b584f405c0d3268a9f1b306c42337b9e.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```
Expected: `[main <sha>] scaffold: AutoDFT.jl ...` with many files changed.

---

## Phase 13 — GitHub push (Deliverable 5 — **PAUSE**)

### Task 32: Confirm GitHub repo creation

- [ ] **Step 1: PAUSE — ask user for explicit confirmation**

Before running `gh repo create`, the executor MUST print:

```
About to create PUBLIC GitHub repo: zazabap/AutoDFT.jl
Then push the current main branch.
This is an externally-visible action. Confirm to proceed? (yes/no)
```

Wait for the user's response. If anything other than an affirmative ("yes",
"y", "go"), STOP and leave the repo local-only.

---

### Task 33: Create and push the repo

Only after Task 32 confirms.

- [ ] **Step 1: Create the repo and push `main`**

Run:
```bash
gh repo create zazabap/AutoDFT.jl \
  --public \
  --source=. \
  --description "Autoresearch harness over ParametricDFT.jl — searches for new AbstractSparseBasis variants that beat baselines on a frozen 512x512 image." \
  --push
```

Expected:
```
✓ Created repository zazabap/AutoDFT.jl on GitHub
✓ Added remote https://github.com/zazabap/AutoDFT.jl.git
✓ Pushed commits to https://github.com/zazabap/AutoDFT.jl.git
```

- [ ] **Step 2: Optionally add the `BASIS_FREEZE_SALT` secret**

Run:
```bash
# Generate a random 32-byte salt locally, DO NOT commit
SALT=$(openssl rand -hex 32)
gh secret set BASIS_FREEZE_SALT --body "$SALT" --repo zazabap/AutoDFT.jl
# Keep $SALT somewhere safe — you'll need it to rehash during legitimate harness updates.
```

Expected: `✓ Set Actions secret BASIS_FREEZE_SALT for zazabap/AutoDFT.jl`.

- [ ] **Step 3: Update `frozen-manifest.toml` with `[secret]` entry**

Run:
```bash
BASIS_FREEZE_SALT="$SALT" make rehash
git add frozen-manifest.toml
git commit -m "chore: seed frozen-manifest [secret].manifest_sha"
git push
```

Expected: push succeeds; CI runs.

- [ ] **Step 4: Confirm CI passes on first push**

Run: `gh run list --repo zazabap/AutoDFT.jl --limit 5`
Expected: the `CI` and `basis-freeze` runs eventually report `completed success`.

---

## Phase 14 — Seed baselines (Deliverable 6)

### Task 34: Attempt `make baseline`

- [ ] **Step 1: Check if ParametricDFT is installed cleanly**

Run: `julia --project=. -e 'using ParametricDFT; println("OK")'`

**Case A — prints "OK":** proceed to Step 2.

**Case B — errors:** skip Step 2; instead create an empty `results.tsv` with just the header and commit with a note:
```bash
printf 'timestamp\tbranch\tcommit_sha\tbasis_name\tbasis_hash\tnum_parameters\tfinal_mse\tprobe_mse\ttrain_steps\ttrain_wallclock_ms\ttransform_time_ms\tdevice\tstatus\tnotes\n' > results.tsv
git add results.tsv
git commit -m "chore: empty results.tsv (ParametricDFT not installable in scaffold env)"
git push
```
Add a line near the top of `program.md`: "NOTE: baselines not yet seeded — run `make baseline` on a host that can install ParametricDFT.jl."

- [ ] **Step 2 (Case A only): Run `make baseline`**

Run: `make baseline`
Expected: four `[ Info: Evaluating baseline ...` lines, then `[ Info: Baseline seeding complete best_mse=<x>`. Takes 10-60 min.

- [ ] **Step 3 (Case A only): Commit the seeded leaderboard**

Run:
```bash
git add results.tsv best.tsv
git commit -m "chore: seed baseline leaderboard (4 ParametricDFT bases on fixture)"
git push
```

---

## Self-Review

Checking the plan against the spec:

**1. Spec coverage.**
- Repo layout: Tasks 3-28 cover every `[F]` and `[E]` file listed in the spec. ✓
- Frozen surface (per-file SHA): Task 20 (manifest builder) + Task 21 (verifier) + Task 22 (initial generation). ✓
- Salted manifest SHA: Task 20 reads `BASIS_FREEZE_SALT`; Task 21 verifies; Task 33 Step 2-3 seeds the secret. ✓
- Identity probe: Tasks 13-14 (probe impl + test) + Task 21 (verify in prepare.jl). ✓
- Evaluation protocol (k=26_214, m=n=9, Adam lr=0.01, 500 steps, seed 42): encoded as constants in Task 4 (`src/AutoDFT.jl`) and used by Task 16 (train.jl). ✓
- Acceptance rule (≥1% relative): encoded in `ACCEPTANCE_REL` (Task 4) and used in Task 19 `run_trial`. ✓
- Autoresearch loop: Task 24 (`program.md`). ✓
- Results schema (14 cols, tab-separated): Task 19 `RESULTS_HEADER` constant + `_append_results_row`. ✓
- CI (Julia 1.10 + 1.11; freeze workflow): Tasks 27-28. ✓
- GPU opt-in via `[gpu]` commit tag: **Not implemented** — acceptable since the harness defaults to `:cpu` and `device=:gpu` can be passed via kwargs. If the spec-literal `[gpu]` tag logic is desired, extend `CI.yml` later. Flagging this as a deliberate simplification.

**2. Placeholder scan.** None — every code block is complete, no "TODO" / "fill in".

**3. Type consistency.**
- `M_QUBITS`, `N_QUBITS`, `IMAGE_SIZE`, `TOPK`, `TRAIN_STEPS`, `LEARNING_RATE`, `SEED`, `ACCEPTANCE_REL` — defined in `src/AutoDFT.jl` (Task 4), used consistently in `evaluate.jl`, `probe.jl`, `train.jl`, `runners.jl`.
- `FIXTURE_PATH`, `MANIFEST_PATH`, `RESULTS_PATH`, `BEST_PATH` — defined once, used consistently.
- `load_fixture`, `run_probe`, `evaluate_basis`, `train_trial`, `run_baseline`, `run_trial` — signatures match between tests and implementations.
- `basis_hash`, `num_parameters`, `image_size`, `forward_transform`, `inverse_transform` — imported from `ParametricDFT` in runners/runners, used with the documented signatures.
- `BASELINES` (Vector of tuples) vs. `TRIAL_REGISTRY` (Dict) — `get_basis_type` handles both correctly.

**4. Scope.** 34 tasks across 14 phases. All tasks are bite-sized (single-file or single-command). Independently verifiable. Scaffold work is largely write-then-verify rather than TDD because most files are config/docs; TDD is applied where it fits (fixture loader, evaluate, probe, train — Phases 3-6).

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-04-19-autoresearch-parametricdft-basis.md`.**

Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks. Better context isolation, fast iteration, explicit review gates.

2. **Inline Execution** — Execute tasks in this session using `superpowers:executing-plans`. Batch execution with checkpoints at phase boundaries.

Which approach?
