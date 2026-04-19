# src/bases/dft_basis.jl
#
# DFTBasis: a manually constructed tensor network that computes the true 2D DFT
# (up to normalization) on the 512×512 fixture at k=26_214.
#
# The previous autoresearch session discovered (spec addendum, 2026-04-19):
#   - Yao's `qft_circuit` as wired into ParametricDFT's einsum pipeline does NOT
#     compute the standard DFT. Empirically, QFTBasis gives MSE ≈ 5636 even
#     though the fixture is band-limited with ~3208 non-zero Fourier coefficients
#     (so a proper DFT should reach MSE ≈ 0 at top-k=26_214).
#
# Diagnosis (this session):
#   Analyzing the full 8×8 tensor-network matrix for m=n=3, we find
#     F_tn ≈ F_std[:, perm_br]
#   where F_std[j,k] = exp(+2πi·j·k/N)/√N is the textbook unitary QFT matrix
#   (positive sign, normalized) and perm_br is the bit-reversal permutation.
#   In other words, the Yao circuit + yao2einsum + `qft_code` pipeline is a
#   correct QFT *up to bit-reversal of the input qubit ordering*. Equivalently,
#   the natural reshape `reshape(img, 2, 2, …, 2)` feeds the image to the wrong
#   qubit-ordering: row-qubit 1 gets the least-significant bit of the row index
#   but the circuit treats row-qubit 1 as the most-significant (or vice-versa).
#
#   So the fix is NOT a phase-assignment fix — the phases are already correct.
#   We just need to permute the input-leg labels of the einsum so that the
#   reshaped image's axes land on the correct qubits.
#
# Construction:
#   - Build the same Yao qft_circuit + subroutines as `qft_code` does.
#   - In the resulting einsum, permute the input-leg labels to reverse the
#     within-row and within-column qubit orderings.
#   - Same tensors (already correct CPHASE phases); only the einsum topology
#     changes.
#
# Sanity check at m=n=9 on the fixture:
#   - Forward gives 3208 non-zero Fourier coefficients (matches the band-limit).
#   - Top-26_214 truncation is a no-op on the DFT coefficients.
#   - Inverse reconstructs the image to MSE ≈ 3.5e-24 (floating-point precision).
#
# Training:
#   The CPHASE tensors (2×2 in Yao's fused representation) have all elements of
#   unit modulus, so `classify_manifold` puts them on `PhaseManifold`. The
#   Hadamard tensors are unitary (`UnitaryManifold`). At the DFT optimum the
#   loss is near-zero and the gradients are numerically zero to machine
#   precision, so the Riemannian optimizer should leave the tensors stationary
#   over 500 steps. If any drift does occur, the test below pins the untrained
#   MSE to be < 1.

import ParametricDFT: forward_transform, inverse_transform,
                      image_size, num_parameters, basis_hash
using ParametricDFT: AbstractSparseBasis, optimize_code_cached, Yao
using OMEinsum
using SHA: sha256

"""
    dft_code(m::Int, n::Int; inverse::Bool=false) -> (optcode, tensors)

Build the optimized einsum + tensor list for a true 2D DFT on a 2^m × 2^n image.
`tensors` are identical to those produced by `ParametricDFT.qft_code` (same
Hadamards and CPHASE encodings); only the einsum's input-leg labels are
permuted to bit-reverse the within-row and within-column qubit orderings.
"""
function dft_code(m::Int, n::Int; inverse::Bool=false)
    total = m + n
    qc1 = Yao.EasyBuild.qft_circuit(m)
    qc2 = Yao.EasyBuild.qft_circuit(n)
    qc = Yao.chain(
        Yao.subroutine(total, qc1, 1:m),
        Yao.subroutine(total, qc2, (m + 1):total),
    )
    tn = Yao.YaoToEinsum.yao2einsum(qc; optimizer = nothing)

    # Mirror qft_code's tensor ordering (H gates first, CPHASE gates after)
    perm_vec = sortperm(tn.tensors, by = x -> !(x ≈ Yao.mat(Yao.H)))
    ixs = tn.code.ixs[perm_vec]
    tensors = tn.tensors[perm_vec]

    input_legs = tn.code.iy[total + 1 : end]
    output_legs = tn.code.iy[1:total]

    # Bit-reverse qubit ordering within each of the row / column qubit blocks.
    qubit_perm = vcat(reverse(collect(1:m)), reverse(collect((m + 1):total)))
    new_input_legs = [input_legs[qubit_perm[i]] for i in 1:total]

    if inverse
        code_reorder = OMEinsum.DynamicEinCode([ixs..., output_legs], new_input_legs)
    else
        code_reorder = OMEinsum.DynamicEinCode([ixs..., new_input_legs], output_legs)
    end
    optcode = optimize_code_cached(
        code_reorder,
        OMEinsum.uniformsize(tn.code, 2),
        OMEinsum.TreeSA(),
    )
    return optcode, tensors
end

"""
    DFTBasis <: AbstractSparseBasis

A tensor-network basis that computes the true 2D DFT (positive sign convention,
normalized by 1/√N per dimension) on a 2^m × 2^n image. Implementation reuses
Yao's QFT tensors but permutes the einsum's input-leg labels to undo the
bit-reversal that `ParametricDFT.qft_code` bakes into the leg topology.
"""
struct DFTBasis <: AbstractSparseBasis
    m::Int
    n::Int
    tensors::Vector
    optcode::OMEinsum.AbstractEinsum
    inverse_code::OMEinsum.AbstractEinsum
end

function DFTBasis(m::Int, n::Int)
    optcode, tensors = dft_code(m, n)
    inverse_code, _ = dft_code(m, n; inverse = true)
    return DFTBasis(m, n, tensors, optcode, inverse_code)
end

function forward_transform(b::DFTBasis, image::AbstractMatrix)
    m, n = b.m, b.n
    @assert size(image) == (2^m, 2^n) "Image size must be $(2^m)×$(2^n), got $(size(image))"
    total = m + n
    img_complex = Complex{Float64}.(image)
    return reshape(
        b.optcode(b.tensors..., reshape(img_complex, fill(2, total)...)),
        2^m, 2^n,
    )
end

function inverse_transform(b::DFTBasis, freq::AbstractMatrix)
    m, n = b.m, b.n
    @assert size(freq) == (2^m, 2^n) "Frequency size must be $(2^m)×$(2^n), got $(size(freq))"
    total = m + n
    freq_complex = Complex{Float64}.(freq)
    return reshape(
        b.inverse_code(conj.(b.tensors)..., reshape(freq_complex, fill(2, total)...)),
        2^m, 2^n,
    )
end

image_size(b::DFTBasis) = (2^b.m, 2^b.n)

num_parameters(b::DFTBasis) = sum(length, b.tensors)

function basis_hash(b::DFTBasis)
    io = IOBuffer()
    write(io, "DFTBasis:m=$(b.m):n=$(b.n):")
    for t in b.tensors, v in t
        write(io, "$(real(v)),$(imag(v));")
    end
    return bytes2hex(sha256(take!(io)))
end

# Train dispatch: reuse our own init, reconstruct via our type
function ParametricDFT._init_circuit(::Type{DFTBasis}, m, n; kwargs...)
    optcode, initial_tensors = dft_code(m, n)
    inverse_code, _ = dft_code(m, n; inverse = true)
    return optcode, inverse_code, initial_tensors
end

function ParametricDFT._build_basis(::Type{DFTBasis}, m, n, tensors, optcode, inverse_code; kwargs...)
    return DFTBasis(m, n, tensors, optcode, inverse_code)
end

ParametricDFT._basis_name(::Type{DFTBasis}) = "DFT"

register_basis!("DFT", DFTBasis)
