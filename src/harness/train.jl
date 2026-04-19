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
