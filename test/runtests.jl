using Test
using AutoDFT

@testset "AutoDFT.jl" begin
    include("harness_tests.jl")
    include("bases_tests.jl")
end
