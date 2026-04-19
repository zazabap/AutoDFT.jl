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
