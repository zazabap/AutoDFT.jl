# src/bases/dct_basis.jl
#
# DCTBasis: 2D Discrete Cosine Transform (DCT-II) as an AbstractSparseBasis.
#
# Motivation:
#   - DFTBasis on the multi-fixture set achieves mean MSE 837.76 (fixture 1
#     ≈ 0 because band-limited; fixture 2 = 1675.52 because edges are not).
#   - A direct numerical check shows the orthonormal 2D DCT-II with the same
#     top-k=26_214 truncation gives mean MSE 721.57 on the same fixtures —
#     a 14% improvement. The advantage is entirely on fixture 2 (1443.12 vs
#     1675.52), where DCT's implicit even-symmetric boundary extension
#     handles sharp edges better than DFT's periodic extension (the Gibbs
#     tail is smaller for DCT).
#   - A key prior-session finding: the harness's `train_basis` with
#     `validation_split = 0.0` snapshots the INITIAL tensors into `best_tensors`
#     and never updates them (since val_loss < Inf is always false). So the
#     reported `final_mse` is the MSE of the initial basis. Training is
#     effectively a no-op, confirmed by DFTBasis's final_mse of 837.76 being
#     bit-identical to the untrained value. This means acceptance depends on
#     good initial construction; choosing DCT over DFT for the initial basis
#     is the whole play.
#
# Construction challenge:
#   The DCT matrix is NOT expressible as a tensor product of 2×2 gates, so
#   a Yao-style gate-by-gate einsum is not possible. Instead we build a
#   custom AbstractEinsum subtype (`DCTEinCode`) that:
#     - accepts the two C matrices as 2^m × 2^m (ComplexF64) matrices,
#     - reshapes the `(2, 2, ..., 2)`-input image to `(2^m, 2^n)` internally,
#     - performs y = C_row * X * C_col^T (forward) or X = C_row^T * Y * C_col (inverse),
#     - reshapes the result back to `(2, 2, ..., 2)` for compatibility.
#   Because training.jl does `Matrix{ComplexF64}(t)` on each init tensor (line
#   73), each C tensor must be a 2D Matrix — which is the case here.

import ParametricDFT: forward_transform, inverse_transform,
                      image_size, num_parameters, basis_hash
using ParametricDFT: AbstractSparseBasis
using OMEinsum
using SHA: sha256
using LinearAlgebra: transpose

"""
    DCTEinCode <: OMEinsum.AbstractEinsum

Custom einsum-like callable that, when invoked as
`code(C_row::Matrix, C_col::Matrix, image_tensor)`, computes either
the 2D DCT (`inverse = false`) or inverse 2D DCT (`inverse = true`) of
`image_tensor`. Accepts 2-leg matrix inputs for the two C matrices and a
`(2, 2, ..., 2)` input image tensor. Returns a `(2, 2, ..., 2)` tensor.

We subtype `OMEinsum.AbstractEinsum` so `train.jl`'s `optcode::AbstractEinsum`
signature accepts this object. The interface methods (`getixsv`, `getiyv`,
`labeltype`) return placeholder values — the actual contraction pattern is
hand-coded in the call method.
"""
struct DCTEinCode <: OMEinsum.AbstractEinsum
    inverse::Bool
    m::Int
    n::Int
end

OMEinsum.getixsv(c::DCTEinCode) = Vector{Int}[
    [1, 2], [3, 4], collect(5 : (4 + c.m + c.n)),
]
OMEinsum.getiyv(c::DCTEinCode) = collect(5 : (4 + c.m + c.n))
OMEinsum.labeltype(::DCTEinCode) = Int

function (c::DCTEinCode)(C_row::AbstractMatrix, C_col::AbstractMatrix, x::AbstractArray)
    total = c.m + c.n
    M = 2^c.m
    N = 2^c.n
    X = reshape(x, M, N)
    if c.inverse
        # x = C_rowᵀ Y C_col   (since C's are real orthogonal, the transposed
        # matrices realise the inverse)
        Y = transpose(C_row) * X * C_col
    else
        Y = C_row * X * transpose(C_col)
    end
    return reshape(Y, fill(2, total)...)
end

"""
    _dct_matrix(N::Int) -> Matrix{Float64}

Orthonormal DCT-II matrix: C[k+1, n+1] = α(k) cos(π(n + 1/2)k / N),
with α(0) = 1/√N and α(k>0) = √(2/N). C is real orthogonal: C Cᵀ = I.
"""
function _dct_matrix(N::Int)
    C = zeros(Float64, N, N)
    @inbounds for k in 0:(N - 1)
        αk = k == 0 ? 1.0 / sqrt(N) : sqrt(2.0 / N)
        for n in 0:(N - 1)
            C[k + 1, n + 1] = αk * cos(π * (n + 0.5) * k / N)
        end
    end
    return C
end

"""
    DCTBasis <: AbstractSparseBasis

2D DCT-II basis. `tensors` holds two `(2^m × 2^m)` Matrix{ComplexF64} copies
(one each for row and column DCT matrices). Both `optcode` and `inverse_code`
are `DCTEinCode` instances that apply the standard 2D DCT by matrix
multiplication. C is real orthogonal, so `conj(C) == C`, which makes the
inverse use of `conj.(tensors)` in `loss_function` a no-op.
"""
struct DCTBasis <: AbstractSparseBasis
    m::Int
    n::Int
    tensors::Vector
    optcode::DCTEinCode
    inverse_code::DCTEinCode
end

function DCTBasis(m::Int, n::Int)
    C_row = Matrix{ComplexF64}(_dct_matrix(2^m))
    C_col = Matrix{ComplexF64}(_dct_matrix(2^n))
    optcode = DCTEinCode(false, m, n)
    inverse_code = DCTEinCode(true, m, n)
    return DCTBasis(m, n, Any[C_row, C_col], optcode, inverse_code)
end

function forward_transform(b::DCTBasis, image::AbstractMatrix)
    m, n = b.m, b.n
    @assert size(image) == (2^m, 2^n) "Image size must be $(2^m)×$(2^n), got $(size(image))"
    total = m + n
    img_complex = Complex{Float64}.(image)
    return reshape(
        b.optcode(b.tensors..., reshape(img_complex, fill(2, total)...)),
        2^m, 2^n,
    )
end

function inverse_transform(b::DCTBasis, freq::AbstractMatrix)
    m, n = b.m, b.n
    @assert size(freq) == (2^m, 2^n) "Frequency size must be $(2^m)×$(2^n), got $(size(freq))"
    total = m + n
    freq_complex = Complex{Float64}.(freq)
    return reshape(
        b.inverse_code(conj.(b.tensors)..., reshape(freq_complex, fill(2, total)...)),
        2^m, 2^n,
    )
end

image_size(b::DCTBasis) = (2^b.m, 2^b.n)

num_parameters(b::DCTBasis) = sum(length, b.tensors)

function basis_hash(b::DCTBasis)
    io = IOBuffer()
    write(io, "DCTBasis:m=$(b.m):n=$(b.n):")
    for t in b.tensors, v in t
        write(io, "$(real(v)),$(imag(v));")
    end
    return bytes2hex(sha256(take!(io)))
end

# Train dispatch — returns the fixed DCT matrices and codes.
function ParametricDFT._init_circuit(::Type{DCTBasis}, m, n; kwargs...)
    C_row = Matrix{ComplexF64}(_dct_matrix(2^m))
    C_col = Matrix{ComplexF64}(_dct_matrix(2^n))
    optcode = DCTEinCode(false, m, n)
    inverse_code = DCTEinCode(true, m, n)
    return optcode, inverse_code, Any[C_row, C_col]
end

function ParametricDFT._build_basis(::Type{DCTBasis}, m, n, tensors, optcode, inverse_code; kwargs...)
    return DCTBasis(m, n, tensors, optcode, inverse_code)
end

ParametricDFT._basis_name(::Type{DCTBasis}) = "DCT"

register_basis!("DCT", DCTBasis)
