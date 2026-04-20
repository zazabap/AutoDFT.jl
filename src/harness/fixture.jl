# src/harness/fixture.jl — FROZEN
#
# Loads the committed 512×512 Float64 fixture images. Read-only; never
# regenerates in-process. Use `scripts/generate_fixture.jl` /
# `scripts/generate_fixture2.jl` to rebuild the raw bytes.

"""
    load_fixture(idx::Integer = 1) -> Matrix{Float64}

Return the `idx`-th frozen 512×512 test image (default: the first, which is the
band-limited Gaussian random field). Reads raw Float64 bytes from
`FIXTURE_PATHS[idx]`.
"""
function load_fixture(idx::Integer = 1)
    1 <= idx <= length(FIXTURE_PATHS) ||
        error("load_fixture: idx=$idx out of range 1..$(length(FIXTURE_PATHS))")
    path = FIXTURE_PATHS[idx]
    isfile(path) || error("Fixture missing: $path. Run the corresponding scripts/generate_fixture*.jl.")
    n = 512
    img = Matrix{Float64}(undef, n, n)
    open(path, "r") do io
        read!(io, img)
    end
    return img
end

"""
    load_fixtures() -> Vector{Matrix{Float64}}

Return all committed fixtures in evaluation order. `evaluate_basis` averages
reconstruction MSE across this set.
"""
function load_fixtures()
    return [load_fixture(i) for i in 1:length(FIXTURE_PATHS)]
end
