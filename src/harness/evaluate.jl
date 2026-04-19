# src/harness/evaluate.jl — FROZEN
#
# Single metric definition used by both the baseline leaderboard and every trial.
# Wraps ParametricDFT.loss_function with the frozen MSELoss(TOPK) configuration.

using ParametricDFT: loss_function, MSELoss, AbstractSparseBasis

"""
    evaluate_basis(basis::AbstractSparseBasis; image=load_fixture()) -> Float64

Compute reconstruction MSE of `basis` on `image` at the frozen sparsity `TOPK = 26_214`.
Equivalent to `loss_function(basis.tensors, M, N, basis.optcode, image, MSELoss(TOPK); inverse_code=basis.inverse_code)`.

The returned value is directly comparable across bases and across trials.
"""
function evaluate_basis(basis::AbstractSparseBasis; image=load_fixture())
    @assert size(image) == IMAGE_SIZE "Image size $(size(image)) != $(IMAGE_SIZE)"
    # ParametricDFT.loss_function requires a ComplexF64 image (per-image, non-batched
    # path does not auto-convert; train_basis converts internally). Mirror that conversion
    # here so callers can pass a real-valued fixture directly.
    cimage = Complex{Float64}.(image)
    return Float64(loss_function(
        basis.tensors, M_QUBITS, N_QUBITS, basis.optcode, cimage, MSELoss(TOPK);
        inverse_code = basis.inverse_code,
    ))
end
