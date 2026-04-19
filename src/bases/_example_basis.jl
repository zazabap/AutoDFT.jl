# src/bases/_example_basis.jl — TEMPLATE (NOT auto-included)
#
# Copy to src/bases/<your_name>_basis.jl, then:
#   1. Rename IdentityBasis → <YourName>Basis throughout the file.
#   2. Replace the body of `forward_transform` / `inverse_transform` with your
#      new construction (e.g., a novel tensor network).
#   3. Optionally: override `_init_circuit` / `_build_basis` (below) so the
#      trainer can build initial tensors for your topology.
#   4. At the bottom: register_basis!("<YourName>", <YourName>Basis)
#
# Interface contract (AbstractSparseBasis):
#   forward_transform(basis, image) -> Complex matrix (size = image_size(basis))
#   inverse_transform(basis, freq)  -> Complex matrix
#   image_size(basis)               -> (height, width)
#   num_parameters(basis)           -> Int  (count of learnable scalars)
#   basis_hash(basis)               -> String (SHA-256 of params; deterministic)
#
# Additional contract for train_basis compatibility:
#   ParametricDFT._init_circuit(::Type{T}, m, n; kwargs...) -> (optcode, inverse_code, initial_tensors)
#   ParametricDFT._build_basis(::Type{T}, m, n, tensors, optcode, inverse_code; kwargs...) -> T
#   ParametricDFT._basis_name(::Type{T}) -> String (display name)
#
# Additional contract for evaluate_basis compatibility:
#   `basis` must have fields `tensors::Vector`, `optcode`, `inverse_code`
#   (or override evaluate_basis for its type).

using ParametricDFT: QFTBasis, AbstractSparseBasis,
                     forward_transform, inverse_transform,
                     image_size, num_parameters, basis_hash
using SHA: sha256

"""
    IdentityBasis <: AbstractSparseBasis

A thin wrapper around `QFTBasis`. Trains and evaluates identically to QFT — exists
only as a template demonstrating the interface contract. Copy this file and replace
`IdentityBasis` with your new type.
"""
struct IdentityBasis <: AbstractSparseBasis
    m::Int
    n::Int
    tensors::Vector
    optcode
    inverse_code
end

function IdentityBasis(m::Int, n::Int)
    inner = QFTBasis(m, n)
    return IdentityBasis(m, n, inner.tensors, inner.optcode, inner.inverse_code)
end

# AbstractSparseBasis interface
forward_transform(b::IdentityBasis, image) =
    forward_transform(QFTBasis(b.m, b.n, b.tensors, b.optcode, b.inverse_code), image)
inverse_transform(b::IdentityBasis, freq) =
    inverse_transform(QFTBasis(b.m, b.n, b.tensors, b.optcode, b.inverse_code), freq)
image_size(b::IdentityBasis) = (2^b.m, 2^b.n)
num_parameters(b::IdentityBasis) = sum(length, b.tensors)
function basis_hash(b::IdentityBasis)
    io = IOBuffer()
    write(io, "IdentityBasis:m=$(b.m):n=$(b.n):")
    for t in b.tensors, v in t
        write(io, "$(real(v)),$(imag(v));")
    end
    return bytes2hex(sha256(take!(io)))
end

# ParametricDFT train_basis dispatch — reuse QFT's init/build
ParametricDFT._init_circuit(::Type{IdentityBasis}, m, n; kwargs...) =
    ParametricDFT._init_circuit(QFTBasis, m, n; kwargs...)

ParametricDFT._build_basis(::Type{IdentityBasis}, m, n, tensors, optcode, inverse_code; kwargs...) =
    IdentityBasis(m, n, tensors, optcode, inverse_code)

ParametricDFT._basis_name(::Type{IdentityBasis}) = "Identity"

# register_basis!("Identity", IdentityBasis)  # <-- uncomment in your copy
