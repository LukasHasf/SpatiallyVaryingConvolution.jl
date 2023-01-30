module SpatiallyVaryingConvolution

using NDTools
using FourierTools
using FFTW
using Arpack, LinearAlgebra
using ScatteredInterpolation
using NaturalNeighborInterpolation

include("utils.jl")
include("preprocessing.jl")
include("model.jl")

export generate_model, read_psfs

end # module
