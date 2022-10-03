using Images: save
using MAT: matwrite
"""    createForwardmodel(H::AbstractArray{T, N}, padded_weights, unpadded_size) where {T, N}

Return a function that computes a spatially varying convolution defined by kernels `H` and
their padded weights `padded_weights`. The convolution accepts a three-dimensional padded
volume and returns the convolved volume.

The dimension of `H` and `padded_weights` should correspond to `(Ny, Nx[, Nz], rank)`
"""
function createForwardmodel(
    H::AbstractArray{T,N}, padded_weights, unpadded_size; reduce=false
) where {T,N}
    @assert ndims(padded_weights) == N "Weights need to be $(N)D."
    Y, X, buf_weighted_x, buf_padded_x, buf_irfft_Y, buf_ifftshift_y, plan, inv_plan = _prepare_buffers_forward(
        H, size(padded_weights)
    )

    forward =
        let Y = Y,
            X = X,
            plan = plan,
            padded_weights = padded_weights,
            buf_irfft_Y = buf_irfft_Y,
            buf_ifftshift_y = buf_ifftshift_y,
            H = H,
            inv_plan = inv_plan,
            buf_padded_x = buf_padded_x,
            buf_weighted_x = buf_weighted_x,
            unpadded_size = unpadded_size,
            N = N,
            reduce = reduce

            function forward(x)
                buf_padded_x .= padND(x, ndims(H) - 1)
                for r in 1:size(padded_weights)[end]
                    buf_weighted_x .= selectdim(padded_weights, N, r) .* buf_padded_x
                    mul!(X, plan, buf_weighted_x)
                    if r == 1
                        Y .= X .* selectdim(H, N, r)
                    else
                        Y .+= X .* selectdim(H, N, r)
                    end
                end
                mul!(buf_irfft_Y, inv_plan, Y)
                FFTW.ifftshift!(buf_ifftshift_y, buf_irfft_Y)
                if reduce
                    return dropdims(
                        sum(unpad(buf_ifftshift_y, unpadded_size...); dims=3); dims=3
                    )
                else
                    return unpad(buf_ifftshift_y, unpadded_size...)
                end
            end
        end
    return forward
end

"""    normalize_weights(weights, comps)

Normalize the `weights ` such that the PSF constructed  from the weighted `comps` always sum to `1`.
"""
function normalize_weights(weights, comps)
    @info "Normalizing weights"
    comp_sums = [sum(comps[:, :, i]) for i in 1:size(comps)[end]]
    weights = Float32.(weights)
    comps = Float32.(comps)
    weightmap = ones(eltype(weights), size(comps)[1:(end-1)])
    local_psf_sum = zero(eltype(weightmap))
    local_weights = view(weights, first(CartesianIndices(weights)).I..., :)
    @inbounds @fastmath @simd for i in CartesianIndices(size(weightmap))
        local_weights = view(weights, i.I..., :)
        local_psf_sum = comp_sums' * local_weights
        print("\r $i: $local_psf_sum \r")
        weightmap[i.I...] = local_psf_sum
    end
    return Float64.(weights ./ weightmap)
end

"""
    generate_model(psfs::AbstractArray{T,3}, rank::Int[, ref_image_index::Int]; reduce=false)

Construct the forward model using the PSFs in `psfs` employing an interpolation
 of the first `rank` components calculated from a SVD. If `reduce==true`, a 3D volume will be
 mapped to a 2D image by summation over the z-axis.

`ref_image_index` is the index of the reference PSF along dim 3 of `psfs`. 
 Default: `ref_image_index = size(psfs)[end] ÷ 2 + 1`
"""
function generateModel(
    psfs::AbstractArray{T,N}, rank::Int, ref_image_index::Int=-1; reduce=false
) where {T,N}
    if ref_image_index == -1
        # Assume reference image is in the middle
        ref_image_index = size(psfs)[end] ÷ 2 + 1
    end
    ND = ndims(psfs)
    my_reduce = reduce
    if reduce && ND == 3
        @info "reduce is true, but dimensions are 2, so reduce is ignored"
        my_reduce = false
    end
    psfs_reg, shifts = SpatiallyVaryingConvolution.registerPSFs(
        psfs, collect(selectdim(psfs, N, ref_image_index))
    )
    comps, weights = decompose(psfs_reg, rank)
    if N == 4 && any(shifts[3, :] .!= zero(Int))
        weights_interp = interpolateWeights(weights, size(comps)[1:3], shifts)
    else
        weights_interp = interpolateWeights(weights, size(comps)[1:2], shifts[1:2, :])
        if N == 4
            # Repeat x-y interpolated weights Nz times 
            weights_interp = repeat(weights_interp, 1, 1, 1, size(psfs_reg, 3))
            # Reshape to (Ny, Nx, Nz, rank)
            weights_interp = permutedims(weights_interp, [1, 2, 4, 3])
        end
    end

    Ns = size(comps)[1:(ND - 1)]
    #= Normalization of h  and weights_interp
        - PSFs at every location should have a sum of 1 -> normalize_weights
        - comps is normalized along the rank dimension according to L2 norm=#
    weights_interp_normalized = normalize_weights(weights_interp, comps)
    # Save normalized weights for later maybe
    # matwrite("normalized_weights.mat", Dict("weights"=>weights_interp))
    h = comps

    # padded values
    H = rfft(padND(h, ND - 1), 1:(ND - 1))
    padded_weights = padND(weights_interp_normalized, ND - 1)
    model = SpatiallyVaryingConvolution.createForwardmodel(
        H, padded_weights, tuple(Ns...); reduce=my_reduce
    )
    return model
end

function generateModel(
    psfs_path::String, psf_name::String, rank::Int, ref_image_index::Int=-1; reduce=false
)
    psfs = readPSFs(psfs_path, psf_name)
    return generateModel(psfs, rank, ref_image_index; reduce=reduce)
end
