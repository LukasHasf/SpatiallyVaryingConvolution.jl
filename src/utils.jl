using MAT: matopen
using HDF5: h5open
export read_psfs, pad_nd, unpad
export _linshift!
export normalize_weights
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
    else
        @warn "Key $key not found in $(path)!"
    end
end

"""    pad_nd(x, n)

Pad `x` along the first `n` dimensions with `0` to twice its size.
"""
function pad_nd(x, n)
    return select_region(x; new_size=2 .* size(x)[1:n], pad_value=zero(eltype(x)))
end

"""    lower_index(N)

Give the index of where the original data starts in an array that was
padded to twice its size along a dimension which originally had length `N`.

Utility function for `unpad`.
"""
function lower_index(N)
    return Bool(N % 2) ? (N + 3) ÷ 2 : (N + 2) ÷ 2
end

"""    upper_index(N)

Give the index of where the original data ends in an array that was
padded to twice its size along a dimension which originally had length `N`.

Utility function for `unpad`.
"""
function upper_index(N)
    return Bool(N % 2) ? 3 * N ÷ 2 + 1 : 3 * N ÷ 2
end

"""    unpad(x, Ns...)

Remove the padding from an array `x` that originally had size `Ns` and was padded to twice its size
along each dimension.
"""
function unpad(x, Ns...)
    low_inds = lower_index.(Ns)
    upp_inds = upper_index.(Ns)
    selection = [low_inds[i]:upp_inds[i] for i in eachindex(Ns)]
    x = x[selection...]
    return x
end

"""    _linshift!(dest::AbstractArray{T,N}, src::AbstractArray{T,N}, shifts::AbstractArray{F,1}; filler=zero(T))

Shift source array `src` according to array of `N`-dimensional tuples `shifts` and save the result into `dest`.
The shifting is not circular, so values that get shifted out of bounds are lost. Values coming in from the other
side will be the value given for `filler`.
"""
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

"""    shift_psfs(stack::AbstractArray{T,N}, shift_indices, good_indices=1:size(stack, N)) where {T,N}

Shift each measurement image/volume in `stack` by the x-y-z-shifts given in `shift_indices` (`size(shift_indices)=(N, size(stack, N))`).

Only the measurements indexed by `good_indices` are considered.
"""
function shift_psfs(
    stack::AbstractArray{T,N}, shift_indices, good_indices=1:size(stack, N)
) where {T,N}
    # Output destination
    yi_reg = similar(stack)
    # Temporary shifting destination
    im_reg = similar(stack, size(stack)[1:(end - 1)])
    #Populate yi_reg
    # In 3D, with z-shift, do a linear shift (without wrap-around)
    if N == 4 && maximum(abs.(shift_indices[3, :])) > zero(eltype(shift_indices))
        for ind in good_indices
            selected_stack = selectdim(stack, N, ind)
            _linshift!(im_reg, selected_stack, shift_indices[:, ind])
            selectdim(yi_reg, N, ind) .= im_reg
        end
    else
        # Else, a circshift should be good enough
        for ind in good_indices
            circshift!(im_reg, selectdim(stack, N, ind), shift_indices[:, ind])
            selectdim(yi_reg, N, ind) .= im_reg
        end
    end
    return yi_reg
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

"""    normalize_weights(weights, comps)

Normalize the `weights` such that the PSF constructed  from the weighted `comps` always sum to `1`.

Size of `weights` and `comps` should be `(Ny, Nx[, Nz], nr_comps)`.
"""
function normalize_weights(weights::AbstractArray{T}, comps::AbstractArray) where {T}
    s_weightmap = size(comps)[1:(end - 1)]
    comp_sums = [sum(c) for c in eachslice(comps; dims=ndims(comps))]
    weightmap = similar(weights, s_weightmap)
    local_psf_sum = zero(T)
    local_weights = view(weights, first(CartesianIndices(s_weightmap)).I..., :)
    @inbounds @fastmath @simd for i in CartesianIndices(s_weightmap)
        local_weights = view(weights, i.I..., :)
        local_psf_sum = comp_sums' * local_weights
        weightmap[i.I...] = local_psf_sum
    end
    return weights ./ weightmap
end
