using Images: save
using MAT: matwrite
"""    createForwardmodel(H::AbstractArray{T, N}, padded_weights, unpadded_size) where {T, N}

Return a function that computes a spatially varying convolution defined by kernels `H` and
their padded weights `padded_weights`. The convolution accepts a three-dimensional padded
volume and returns the convolved volume.

The dimension of `H` and `padded_weights` should correspond to `(Ny, Nx[, Nz], rank)`
"""
function createForwardmodel(
    H::AbstractArray{T,N}, padded_weights, unpadded_size; reduce=false, scaling=nothing
) where {T,N}
    @assert ndims(padded_weights) == N "Weights need to be $(N)D."
    Y, X, buf_weighted_x, buf_padded_x, buf_irfft_Y, buf_ifftshift_y, plan, inv_plan = _prepare_buffers_forward(
        H, size(padded_weights), scaling
    )
    println(scaling)
    if isnothing(scaling)
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
                buf_padded_x .= pad_nd(x, ndims(H) - 1)
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
    else
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

            function forward_scaled(x)
                buf_padded_x .= pad_nd(x, ndims(H) - 1; fac=2*scaling)
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
                    unpadded = unpad(buf_ifftshift_y, unpadded_size...)
                    return scale_fourier(unpadded, 1/scaling)
                end
            end
        end
    end
    return forward
end

"""
    generate_model(psfs::AbstractArray{T,3}, rank::Int[, ref_image_index::Int]; reduce=false)

Construct the forward model using the PSFs in `psfs` employing an interpolation
 of the first `rank` components calculated from a SVD. If `reduce==true`, a 3D volume will be
 mapped to a 2D image by summation over the z-axis.

`ref_image_index` is the index of the reference PSF along dim 3 of `psfs`. 
 Default: `ref_image_index = size(psfs)[end] ÷ 2 + 1`
"""
function generate_model(
    psfs::AbstractArray{T,N}, rank::Int; ref_image_index::Int=-1, reduce=false, shifts=nothing, scaling=nothing
) where {T,N}
    # Flag to indicate whether relative PSF shifts are given by the user or should be calculated
    given_psfs = !isnothing(shifts)
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
    if !given_psfs
        psfs_reg, shifts = SpatiallyVaryingConvolution.registerPSFs(
            psfs, collect(selectdim(psfs, N, ref_image_index))
        )
    else
        # If shifts are given, we still want to register the PSFs (move them to the same position in the FOV), 
        # so that we can decompose them. Just ignore the computed shifts.
        psfs_reg, _ = SpatiallyVaryingConvolution.registerPSFs(
            psfs, collect(selectdim(psfs, N, ref_image_index))
        )
    end
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
    padded_weights = pad_nd(weights_interp_normalized, ND - 1)
    if !isnothing(scaling)
        h = scale_fourier(h, scaling; dims=1:(ndims(h)-1))
        padded_weights = pad_nd(weights_interp_normalized, ND - 1; fac=2*scaling)
        Ns = Ns .* scaling
    end
    h_pad = pad_nd(h, ND - 1)
    H = rfft(h_pad, 1:(ND - 1))
    model = SpatiallyVaryingConvolution.createForwardmodel(
        H, padded_weights, tuple(Ns...); reduce=my_reduce, scaling=scaling
    )
    return model
end

function generate_model(
    psfs_path::String, psf_name::String, rank::Int; ref_image_index::Int=-1, reduce=false
)
    psfs = read_psfs(psfs_path, psf_name)
    return generate_model(psfs, rank; ref_image_index=ref_image_index, reduce=reduce)
end

function generate_model(
    psfs_path::String, psf_name::String, shift_name::String, rank::Int; ref_image_index::Int=-1, reduce=false, scaling=nothing
)
    psfs = read_psfs(psfs_path, psf_name)
    shifts = read_psfs(psfs_path, shift_name)
    return generate_model(psfs, rank; ref_image_index=ref_image_index, reduce=reduce, shifts=shifts, scaling=scaling)
end
