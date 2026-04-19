# src/harness/probe.jl — FROZEN
#
# BASIS_IDENTITY_PROBE. Runs a fully deterministic compression of the fixture
# through QFTBasis(9,9) — which has no learnable parameters beyond the initial
# QFT circuit — and returns the reconstruction MSE.
#
# The returned scalar is pinned in frozen-manifest.toml under [probe].qft_identity_mse.
# Any drift in ParametricDFT numerics, top-k truncation, or fixture bytes will change
# the probe value and be caught by prepare.jl / test/harness_tests.jl.

"""
    run_probe() -> Float64

Deterministic MSE for `QFTBasis(M_QUBITS, N_QUBITS)` evaluated against the fixture
at `k = TOPK`. Used as the semantic regression gate.
"""
function run_probe()
    basis = QFTBasis(M_QUBITS, N_QUBITS)
    return evaluate_basis(basis)
end
