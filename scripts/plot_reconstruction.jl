# scripts/plot_reconstruction.jl — EDITABLE
#
# Renders docs/reconstruction.png: a 2×2 grid comparing the fixture against
# what QFTBasis and DFTBasis actually reconstruct at k = TOPK. QFT misses
# badly (MSE ≈ 5636); DFT is pixel-perfect (MSE ≈ 3e-24). The fourth panel
# shows the residual where QFT goes wrong — low-frequency smearing because
# the qubit-ordering mismatch aliases the "low-pass kept coefficients" to
# the wrong frequencies.
#
# Regenerate with:
#   julia --project=. scripts/plot_reconstruction.jl

using AutoDFT
using CairoMakie
using ParametricDFT: QFTBasis, topk_truncate, forward_transform, inverse_transform
using Printf

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
const OUT_PATH  = joinpath(REPO_ROOT, "docs", "reconstruction.png")

# Import DFTBasis after it's loaded via registry.jl auto-discover
function main()
    fixture = AutoDFT.load_fixture()

    qft = QFTBasis(AutoDFT.M_QUBITS, AutoDFT.N_QUBITS)
    dft = Main.AutoDFT.DFTBasis(AutoDFT.M_QUBITS, AutoDFT.N_QUBITS)

    function reconstruct(basis)
        fwd = forward_transform(basis, fixture)
        tru = topk_truncate(fwd, AutoDFT.TOPK)
        inv = inverse_transform(basis, tru)
        return real.(inv)
    end

    qft_rec = reconstruct(qft)
    dft_rec = reconstruct(dft)

    # Harness convention: sum of squared errors across all pixels (matches
    # `ParametricDFT.loss_function(..., MSELoss(k))`). Matches the numbers
    # in README / results.tsv.
    qft_mse = sum(abs2, fixture .- qft_rec)
    dft_mse = sum(abs2, fixture .- dft_rec)
    qft_residual = fixture .- qft_rec

    # Shared color range across image panels so the visual comparison is honest
    vlim = maximum(abs, fixture)
    imgkw = (colormap = :RdBu, colorrange = (-vlim, vlim))

    fig = Figure(size = (1100, 900), fontsize = 14)

    ax1 = Axis(fig[1, 1], title = "Fixture (ground truth)",
               aspect = DataAspect(), yreversed = true)
    hidedecorations!(ax1); hidespines!(ax1)
    heatmap!(ax1, fixture; imgkw...)

    ax2 = Axis(fig[1, 2],
               title = @sprintf("QFT reconstruction  (MSE = %.2f)", qft_mse),
               aspect = DataAspect(), yreversed = true)
    hidedecorations!(ax2); hidespines!(ax2)
    heatmap!(ax2, qft_rec; imgkw...)

    ax3 = Axis(fig[2, 1],
               title = @sprintf("DFT reconstruction  (MSE = %.2e)", dft_mse),
               aspect = DataAspect(), yreversed = true)
    hidedecorations!(ax3); hidespines!(ax3)
    heatmap!(ax3, dft_rec; imgkw...)

    # Residual uses its own color range — the scale is different
    res_lim = maximum(abs, qft_residual)
    ax4 = Axis(fig[2, 2],
               title = "QFT residual (fixture − QFT rec)",
               aspect = DataAspect(), yreversed = true)
    hidedecorations!(ax4); hidespines!(ax4)
    hm4 = heatmap!(ax4, qft_residual;
                   colormap = :RdBu, colorrange = (-res_lim, res_lim))
    Colorbar(fig[2, 3], hm4, width = 12)

    Label(fig[0, 1:2], "QFTBasis vs DFTBasis on the frozen 512×512 fixture, k = 26_214";
          fontsize = 16, font = :bold, halign = :center)

    mkpath(dirname(OUT_PATH))
    save(OUT_PATH, fig)
    @info "Wrote $OUT_PATH" qft_mse dft_mse residual_max = res_lim
    return OUT_PATH
end

(abspath(PROGRAM_FILE) == @__FILE__) && main()
