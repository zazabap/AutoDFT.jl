# src/harness/evaluate.jl — FROZEN
#
# Single metric definition used by both the baseline leaderboard and every
# trial. Wraps `ParametricDFT.loss_function` with the frozen `MSELoss(TOPK)`
# configuration and averages across ALL frozen fixtures (`load_fixtures()`).
#
# Rewarding average-MSE across multiple fixtures means a winning basis must
# compress well on EACH image type, not just overfit one. See the spec
# addendum (Addendum C) for the multi-fixture eval rationale.

using ParametricDFT: loss_function, MSELoss, AbstractSparseBasis

"""
    evaluate_basis(basis::AbstractSparseBasis; image=nothing) -> Float64

Compute reconstruction MSE of `basis` at the frozen sparsity `TOPK = 26_214`.

- If `image === nothing` (default): average MSE across every fixture in
  `load_fixtures()`. This is the canonical evaluation used by
  `run_baseline` / `run_trial`.
- If `image` is a single `Matrix`: evaluate on that one image only — useful
  for diagnostics and per-fixture reports. Not the acceptance metric.

The returned value is directly comparable across bases and across trials.
"""
function evaluate_basis(basis::AbstractSparseBasis; image=nothing)
    images = image === nothing ? load_fixtures() : (image isa AbstractMatrix ? [image] : image)
    total = 0.0
    for img in images
        @assert size(img) == IMAGE_SIZE "Image size $(size(img)) != $(IMAGE_SIZE)"
        # ParametricDFT.loss_function requires a ComplexF64 image (the per-image,
        # non-batched path does not auto-convert — train_basis converts internally).
        cimg = Complex{Float64}.(img)
        total += Float64(loss_function(
            basis.tensors, M_QUBITS, N_QUBITS, basis.optcode, cimg, MSELoss(TOPK);
            inverse_code = basis.inverse_code,
        ))
    end
    return total / length(images)
end
