# src/bases/block_dct_basis.jl
#
# BlockDCTBasis: 2D block-DCT-II with 32×32 blocks.
#
# Motivation:
#   - DCTBasis (full 512×512 DCT) achieves mean MSE 721.57 on the multi-
#     fixture set (fixture 1 = 0.02, fixture 2 = 1443.12).
#   - A direct numerical sweep over block sizes (8, 16, 32, 64, 128, 256, 512)
#     shows block size 32 minimizes the mean MSE at 641.91 (fixture 1 =
#     33.83, fixture 2 = 1249.99). That is an 11% further reduction.
#   - Intuition: fixture 2's sharp edges couple low-frequency-per-block
#     structure with pixel-local fine detail. Block-DCT compresses each
#     block independently, so edges not aligned to block boundaries retain
#     their local transform advantages — like JPEG's 8×8 DCT sweet spot
#     scaled up. Block size 32 balances fixture-1 penalty (must keep low-
#     frequency per-block content) against fixture-2 gain.
#
# Implementation reuses the `DCTEinCode` custom einsum from `dct_basis.jl`:
# the row and column C matrices are simply block-diagonal DCT matrices of
# size 512 × 512, with 16 × 16 = 256 copies of a 32×32 DCT-II matrix along
# the diagonal. A block-diagonal matrix is still a valid real orthogonal
# 512×512 matrix, so it slots into the existing DCTEinCode machinery
# without any further changes to the forward/inverse pipeline.

import ParametricDFT: forward_transform, inverse_transform,
                      image_size, num_parameters, basis_hash
using ParametricDFT: AbstractSparseBasis
using OMEinsum
using SHA: sha256
using LinearAlgebra: transpose

const BLOCK_DCT_SIZE = 32

"""
    _block_dct_matrix(N::Int, block::Int) -> Matrix{Float64}

Build a block-diagonal 2D DCT-II matrix of total size N × N, with `N ÷ block`
blocks of size `block × block` along the diagonal. Requires `N` divisible by
`block`.
"""
function _block_dct_matrix(N::Int, block::Int)
    @assert N % block == 0 "N=$N must be divisible by block=$block"
    C_block = _dct_matrix(block)       # reuse _dct_matrix from DCTBasis file
    C = zeros(Float64, N, N)
    @inbounds for i in 1:(N ÷ block)
        r = ((i - 1) * block + 1) : (i * block)
        C[r, r] .= C_block
    end
    return C
end

"""
    BlockDCTBasis <: AbstractSparseBasis

Block-diagonal 2D DCT-II basis. `tensors` holds two `(2^m × 2^m)`
Matrix{ComplexF64} block-diagonal DCT matrices (one each for row / column
DCT). Uses the same `DCTEinCode` custom einsum as `DCTBasis`.
"""
struct BlockDCTBasis <: AbstractSparseBasis
    m::Int
    n::Int
    tensors::Vector
    optcode::DCTEinCode
    inverse_code::DCTEinCode
    block_size::Int
end

function BlockDCTBasis(m::Int, n::Int; block_size::Int = BLOCK_DCT_SIZE)
    N_row = 2^m
    N_col = 2^n
    @assert N_row % block_size == 0
    @assert N_col % block_size == 0
    C_row = Matrix{ComplexF64}(_block_dct_matrix(N_row, block_size))
    C_col = Matrix{ComplexF64}(_block_dct_matrix(N_col, block_size))
    optcode = DCTEinCode(false, m, n)
    inverse_code = DCTEinCode(true, m, n)
    return BlockDCTBasis(m, n, Any[C_row, C_col], optcode, inverse_code, block_size)
end

function forward_transform(b::BlockDCTBasis, image::AbstractMatrix)
    m, n = b.m, b.n
    @assert size(image) == (2^m, 2^n) "Image size must be $(2^m)×$(2^n), got $(size(image))"
    total = m + n
    img_complex = Complex{Float64}.(image)
    return reshape(
        b.optcode(b.tensors..., reshape(img_complex, fill(2, total)...)),
        2^m, 2^n,
    )
end

function inverse_transform(b::BlockDCTBasis, freq::AbstractMatrix)
    m, n = b.m, b.n
    @assert size(freq) == (2^m, 2^n) "Frequency size must be $(2^m)×$(2^n), got $(size(freq))"
    total = m + n
    freq_complex = Complex{Float64}.(freq)
    return reshape(
        b.inverse_code(conj.(b.tensors)..., reshape(freq_complex, fill(2, total)...)),
        2^m, 2^n,
    )
end

image_size(b::BlockDCTBasis) = (2^b.m, 2^b.n)

num_parameters(b::BlockDCTBasis) = sum(length, b.tensors)

function basis_hash(b::BlockDCTBasis)
    io = IOBuffer()
    write(io, "BlockDCTBasis:m=$(b.m):n=$(b.n):block=$(b.block_size):")
    for t in b.tensors, v in t
        write(io, "$(real(v)),$(imag(v));")
    end
    return bytes2hex(sha256(take!(io)))
end

# Train dispatch — returns the fixed block-DCT matrices and codes.
function ParametricDFT._init_circuit(::Type{BlockDCTBasis}, m, n;
                                       block_size::Int = BLOCK_DCT_SIZE, kwargs...)
    C_row = Matrix{ComplexF64}(_block_dct_matrix(2^m, block_size))
    C_col = Matrix{ComplexF64}(_block_dct_matrix(2^n, block_size))
    optcode = DCTEinCode(false, m, n)
    inverse_code = DCTEinCode(true, m, n)
    return optcode, inverse_code, Any[C_row, C_col]
end

function ParametricDFT._build_basis(::Type{BlockDCTBasis}, m, n, tensors, optcode, inverse_code;
                                      block_size::Int = BLOCK_DCT_SIZE, kwargs...)
    return BlockDCTBasis(m, n, tensors, optcode, inverse_code, block_size)
end

ParametricDFT._basis_name(::Type{BlockDCTBasis}) = "BlockDCT"

register_basis!("BlockDCT", BlockDCTBasis)
