using MAT: matopen
using HDF5: h5open
export read_psfs, pad_nd, unpad
export _linshift!
"""    
    read_psfs(path::String, key::String)

Read the PSFs stored in file `path` accessible as field `key`. 
Supports MAT and HDF5 file format. 

 # Examples
 ```
 julia> readPSFs("myPSFs.mat", "PSFs")
 ```

 ```
 julia> readPSFs("myPSFs.h5", "myPSFsDataset")
 ```
"""
function read_psfs(path::String, key::String)
    if occursin(".mat", path)
        file = matopen(path)
    elseif occursin(".h5", path)
        file = h5open(path, "r")
    end
    if haskey(file, key)
        psfs = read(file, key)
        return psfs
    end
end

"""    pad_nd(x, n)

Pad `x` along the first `n` dimensions with `0` to twice its size.
"""
function pad_nd(x, n)
    return select_region(x; new_size=2 .* size(x)[1:n], pad_value=zero(eltype(x)))
end

function lower_index(N)
    return Bool(N % 2) ? (N + 3) ÷ 2 : (N + 2) ÷ 2
end

function upper_index(N)
    return Bool(N % 2) ? 3 * N ÷ 2 + 1 : 3 * N ÷ 2
end

function unpad(x, Ns...)
    low_inds = [lower_index(N) for N in Ns]
    upp_inds = [upper_index(N) for N in Ns]
    selection = [low_inds[i]:upp_inds[i] for i in eachindex(Ns)]
    x = x[selection...]
    return x
end

function _linshift!(
    dest::AbstractArray{T,N},
    src::AbstractArray{T,N},
    shifts::AbstractArray{F,1};
    filler=zero(T),
) where {T,F,N}
    myshifts = ntuple(i -> shifts[i], length(shifts))
    for ind in CartesianIndices(dest)
        shifted_ind = ind.I .- myshifts
        value = filler
        if !(
            any(shifted_ind .<= zero(eltype(shifted_ind))) || any(shifted_ind .> size(src))
        )
            value = src[shifted_ind...]
        end
        dest[ind.I...] = value
    end
end

function _prepare_buffers_forward(H::AbstractArray{T,N}, size_padded_weights) where {T,N}
    ND = ndims(H)
    # x is padded in first N-1 dimension to be as big as padded_weights
    size_x = size_padded_weights[1:(ND - 1)]
    # Y aggregates the FT of the convolution of the weighted volume and the PSF components
    Y = similar(H, size_x[1] ÷ 2 + 1, size_x[2:end]...)
    # X holds the FT of the weighted image
    X = similar(Y)
    # Buffers for the weighted image and the irfft-ed and ifftshift-ed convolution images
    buf_weighted_x = similar(H, real(T), size_x...) # Array{real(T), ND-1}(undef, size_x...)
    buf_padded_x = similar(buf_weighted_x)
    buf_irfft_Y = similar(buf_weighted_x)
    buf_ifftshift_y = similar(buf_irfft_Y)
    # RFFT and IRRFT plans
    plan = plan_rfft(buf_weighted_x; flags=FFTW.MEASURE)
    inv_plan = inv(plan)
    return Y, X, buf_weighted_x, buf_padded_x, buf_irfft_Y, buf_ifftshift_y, plan, inv_plan
end
