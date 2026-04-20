# scripts/generate_fixture2.jl — FROZEN
#
# Generates the SECOND 512×512 test fixture: a piecewise-smooth image with
# sharp edges, deterministic procedural content, seed=43. This complements
# the band-limited Gaussian field (fixture 1, generated with seed 42) by
# providing content that DFT compression cannot perfectly reconstruct at
# any finite sparsity — the sharp edges create broadband Fourier content.
#
# Output: data/fixture2_512.bin (Float64, row-major, 262144 elements).
#
# Components (all deterministic from seed 43):
#   - a central high-contrast rectangle
#   - two overlapping off-centre rectangles at different intensities
#   - a diagonal stripe
#   - four radial Gaussian bumps at random centres / sigmas
#   - low-amplitude Gaussian noise floor
#   - final zero-mean, unit-variance normalization
#
# Re-running this script must produce a byte-identical file.

using Random
using LinearAlgebra

const OUT_PATH = joinpath(@__DIR__, "..", "data", "fixture2_512.bin")
const SIZE     = 512
const SEED     = 43

function generate_fixture2()
    Random.seed!(SEED)
    img = zeros(Float64, SIZE, SIZE)

    # Central rectangle (bright on dark)
    img[90:300, 120:280] .+= 1.2

    # Overlapping mid-right rectangle at a lower intensity
    img[180:390, 230:400] .+= 0.7

    # Small inner hole to create an edge inside the overlap
    img[220:300, 270:340] .-= 0.9

    # Diagonal stripe — a 3-pixel-thick anti-aliased diagonal
    for i in 1:SIZE
        j_centre = 0.7 * i + 60.0
        for dj in -1:1
            j = clamp(round(Int, j_centre + dj), 1, SIZE)
            img[i, j] += 0.4 * (1 - abs(dj) / 2)
        end
    end

    # Four radial Gaussian bumps
    bump_centres = [(100, 420, 35), (420, 120, 45), (450, 460, 25), (60, 60, 18)]
    for (cx, cy, r) in bump_centres
        for i in 1:SIZE, j in 1:SIZE
            d2 = (i - cx)^2 + (j - cy)^2
            img[i, j] += 0.5 * exp(-d2 / (2r^2))
        end
    end

    # Low-amplitude i.i.d. Gaussian noise floor
    img .+= 0.04 .* randn(Float64, SIZE, SIZE)

    # Normalize to zero mean, unit variance (matches fixture 1's convention)
    img .-= sum(img) / length(img)
    img ./= sqrt(sum(abs2, img) / length(img))
    return img
end

function main()
    img = generate_fixture2()
    @assert size(img) == (SIZE, SIZE)
    @assert eltype(img) == Float64
    mkpath(dirname(OUT_PATH))
    open(OUT_PATH, "w") do io
        write(io, img)
    end
    @info "Wrote $OUT_PATH" size = size(img) bytes = stat(OUT_PATH).size
    return nothing
end

(abspath(PROGRAM_FILE) == @__FILE__) && main()
