# scripts/generate_fixture.jl
#
# Generates the frozen 512×512 test image used by every trial.
# Output: data/fixture_512.bin (Float64, row-major, 262144 elements = 2097152 bytes).
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

(abspath(PROGRAM_FILE) == @__FILE__) && main()
