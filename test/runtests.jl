using Test, FFTW
using Arpack, LinearAlgebra
using SpatiallyVaryingConvolution
using ScatteredInterpolation
include("../src/model.jl")
include("../src/preprocessing.jl")
using MAT
using HDF5

include("utils.jl")
include("shifts.jl")
include("convolutions.jl")
