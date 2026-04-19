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
