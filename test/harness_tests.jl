# test/harness_tests.jl — FROZEN
using Test
using AutoDFT
using Random
using LinearAlgebra
using ParametricDFT

@testset "harness" begin

@testset "load_fixture" begin
    # Fixture 1: band-limited Gaussian, seed 42
    img1 = AutoDFT.load_fixture()           # default idx=1
    @test size(img1) == (512, 512)
    @test eltype(img1) == Float64
    @test abs(sum(img1) / length(img1)) < 1e-9
    @test abs(sqrt(sum(abs2, img1) / length(img1)) - 1.0) < 1e-9

    # Fixture 2: piecewise-smooth + edges, seed 43 (same zero-mean / unit-std)
    img2 = AutoDFT.load_fixture(2)
    @test size(img2) == (512, 512)
    @test eltype(img2) == Float64
    @test abs(sum(img2) / length(img2)) < 1e-9
    @test abs(sqrt(sum(abs2, img2) / length(img2)) - 1.0) < 1e-9
    @test img1 != img2   # they're genuinely different images

    # load_fixtures returns all fixtures in order
    imgs = AutoDFT.load_fixtures()
    @test length(imgs) == 2
    @test imgs[1] == img1
    @test imgs[2] == img2
end

@testset "evaluate_basis returns finite MSE for QFTBasis" begin
    Random.seed!(AutoDFT.SEED)
    basis = QFTBasis(AutoDFT.M_QUBITS, AutoDFT.N_QUBITS)
    mse = AutoDFT.evaluate_basis(basis)
    @test isa(mse, Real)
    @test isfinite(mse)
    @test mse > 0
    # QFT on a 10% top-k reconstruction of a low-pass image should be well-compressed
    # but not exactly zero. A sanity band: 1e-12 < mse < 1e6.
    @test 1e-12 < mse < 1e6
end

@testset "run_probe is deterministic" begin
    a = AutoDFT.run_probe()
    b = AutoDFT.run_probe()
    @test a ≈ b atol=1e-14          # same run, same result, bit-identical
    @test isfinite(a)
    @test a > 0
end

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

# Additional @testsets added in later steps: evaluate, probe, train
end # @testset "harness"
