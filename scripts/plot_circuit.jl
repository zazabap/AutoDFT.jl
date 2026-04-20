# scripts/plot_circuit.jl — EDITABLE
#
# Renders circuit diagrams via `ParametricDFT.plot_circuit`:
#   docs/circuit.png         — QFTBasis(4,4)        pedagogical (= DFTBasis circuit)
#   docs/circuit_full.png    — QFTBasis(9,9)        actual experiment (denser)
#   docs/circuit_tebd.png    — TEBDBasis(4,4)       baseline parent for 4 failed trials
#   docs/circuit_entangled.png — EntangledQFTBasis(4,4)  baseline parent
#
# DFTBasis reuses the QFT tensors unchanged — the only difference is a
# permutation on the einsum's input-leg labels, so its circuit diagram is
# identical to QFT.
#
# Regenerate with:
#   julia --project=. scripts/plot_circuit.jl

using ParametricDFT

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))

const OUT_QFT_PEDA       = joinpath(REPO_ROOT, "docs", "circuit.png")
const OUT_QFT_FULL       = joinpath(REPO_ROOT, "docs", "circuit_full.png")
const OUT_TEBD_PEDA      = joinpath(REPO_ROOT, "docs", "circuit_tebd.png")
const OUT_ENTANGLED_PEDA = joinpath(REPO_ROOT, "docs", "circuit_entangled.png")

function main()
    mkpath(dirname(OUT_QFT_PEDA))

    @info "Rendering QFT 4x4 (= DFTBasis circuit, pedagogical)"
    plot_circuit(QFTBasis(4, 4); output_path = OUT_QFT_PEDA)

    @info "Rendering QFT 9x9 (actual experiment scale)"
    plot_circuit(QFTBasis(9, 9); output_path = OUT_QFT_FULL)

    @info "Rendering TEBD 4x4 (parent for ExtendedTEBD/DeepTEBD/HadSandwichTEBD)"
    plot_circuit(TEBDBasis(4, 4); output_path = OUT_TEBD_PEDA)

    @info "Rendering EntangledQFT 4x4 (baseline)"
    plot_circuit(EntangledQFTBasis(4, 4); output_path = OUT_ENTANGLED_PEDA)

    @info "All circuit diagrams written" dir = dirname(OUT_QFT_PEDA)
    return nothing
end

(abspath(PROGRAM_FILE) == @__FILE__) && main()
