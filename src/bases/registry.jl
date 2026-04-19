# src/bases/registry.jl — EDITABLE
#
# Maps basis name strings → (::Type{<:AbstractSparseBasis}) constructor types.
# `BASELINES` is the fixed seed set for `make baseline`.
# `TRIAL_REGISTRY` is the open set an autoresearch session adds to via
# `register_basis!("<Name>", <Name>Basis)` at the bottom of a new-basis file.
#
# Note: MERABasis is intentionally excluded. ParametricDFT's mera_code asserts
# `m` must be a power of 2 (>= 2), which conflicts with the frozen m=n=9 config
# (see `AutoDFT.M_QUBITS`). Re-add it only after that constraint is relaxed or
# the image size is changed.

using ParametricDFT: QFTBasis, EntangledQFTBasis, TEBDBasis, AbstractSparseBasis

"""Baselines seeded by `make baseline`. Order defines the TSV row order."""
const BASELINES = [
    ("QFT",          QFTBasis),
    ("EntangledQFT", EntangledQFTBasis),
    ("TEBD",         TEBDBasis),
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
