module AutoDFT

using ParametricDFT
using CUDA     # triggers ParametricDFT.CUDAExt; enables :gpu device when CUDA.functional()
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

# Additional includes added by later groups (harness, bases, runners).
# Do not add `include(...)` statements for files that don't yet exist —
# they're created by subsequent scaffolding groups.
include("harness/fixture.jl")
include("harness/evaluate.jl")
include("harness/probe.jl")
include("harness/train.jl")
include("bases/registry.jl")
include("runners.jl")

export load_fixture, run_probe, evaluate_basis, train_trial
export BASELINES, TRIAL_REGISTRY, register_basis!
export run_baseline, run_trial

# Forward declarations — real implementations added by later groups.
function run_baseline end
function run_trial end

end # module
