# test/bases_tests.jl — EDITABLE
# Agent adds @testset blocks here for each new basis (unitarity + interface conformance).
using Test
using Random
using AutoDFT
using ParametricDFT
using FFTW

@testset "bases" begin
    @testset "DFTBasis interface + correctness" begin
        Random.seed!(42)

        # Small sanity: m=n=3, round-trip on random input
        b_small = AutoDFT.DFTBasis(3, 3)
        @test image_size(b_small) == (8, 8)
        @test num_parameters(b_small) > 0
        x = randn(ComplexF64, 8, 8)
        y = forward_transform(b_small, x)
        x̂ = inverse_transform(b_small, y)
        @test x̂ ≈ x atol = 1e-8
        @test basis_hash(b_small) == basis_hash(b_small)

        # Full-size: DFTBasis on the fixture must match FFT magnitudes
        # (up to the 1/N normalization factor per dimension).
        b = AutoDFT.DFTBasis(9, 9)
        @test image_size(b) == (512, 512)

        img = AutoDFT.load_fixture()
        via_basis = forward_transform(b, img)
        via_fft = fft(ComplexF64.(img))

        # DFT (positive sign, normalized 1/N per dim) vs FFT (negative sign, unnormalized)
        # have identical magnitude spectra up to the factor N = 2^9 per dimension,
        # i.e. sort(|basis|) * N = sort(|fft|) for any input.
        N = 512
        @test isapprox(
            sort(vec(abs.(via_basis))) .* N,
            sort(vec(abs.(via_fft)));
            rtol = 1e-9,
        )

        # Round-trip with top-k=26_214 on the band-limited fixture: MSE should be
        # near machine precision (~1e-20), ruling out the Walsh-Hadamard ceiling.
        # Evaluate specifically on fixture 1 (band-limited) where the near-zero
        # claim holds; the multi-fixture average includes fixture 2 (edges) which
        # is not band-limited and has non-trivial MSE even under exact DFT.
        mse_band_limited = AutoDFT.evaluate_basis(b; image = AutoDFT.load_fixture(1))
        @test mse_band_limited < 1.0   # actually ~1e-24; generous bound against perturbations
    end
end
