# scripts/plot_circuit.jl — EDITABLE
#
# Renders docs/circuit.png: the 2D QFT tensor-network that DFTBasis realises.
# Uses `ParametricDFT.plot_circuit` against a pedagogical-size basis (m=n=4)
# so the gate structure is legible at README scale. The actual experiment
# uses m=n=9 — same structure, 18 qubits.
#
# DFTBasis reuses these exact tensors; the only difference between QFTBasis
# and DFTBasis is a permutation on the einsum's input-leg labels, so the
# circuit diagram is identical.
#
# Regenerate with:
#   julia --project=. scripts/plot_circuit.jl

using ParametricDFT

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
const OUT_PEDA  = joinpath(REPO_ROOT, "docs", "circuit.png")        # m=n=4 pedagogical
const OUT_FULL  = joinpath(REPO_ROOT, "docs", "circuit_full.png")   # m=n=9 actual

function main()
    mkpath(dirname(OUT_PEDA))

    # Pedagogical diagram (m=n=4, 8 qubits) — readable at README scale
    @info "Rendering pedagogical 4x4 circuit"
    plot_circuit(QFTBasis(4, 4); output_path = OUT_PEDA)

    # Actual experiment (m=n=9, 18 qubits) — same structure, denser
    @info "Rendering actual 9x9 circuit"
    plot_circuit(QFTBasis(9, 9); output_path = OUT_FULL)

    @info "Wrote circuit diagrams" pedagogical = OUT_PEDA actual = OUT_FULL
    return nothing
end

(abspath(PROGRAM_FILE) == @__FILE__) && main()
