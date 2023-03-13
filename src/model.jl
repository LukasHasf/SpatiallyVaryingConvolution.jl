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
    return forward
end

"""
    generate_model(psfs::AbstractArray{T,N}, rank::Int[, ref_image_index::Int]; reduce=false, itp_method=Shepard(), positions=nothing) where {T,N}

Construct the forward model using the PSFs in `psfs` employing an interpolation
 of the first `rank` components calculated from a SVD. If `reduce==true`, a 3D volume will be
 mapped to a 2D image by summation over the z-axis.

The interpolation of the coefficients of the eigenimages is done using the `itp_method`.

If the PSF locations differ from their center of mass, `positions` can be supplied as an array of size `(N-1, size(psfs,N))`.

`ref_image_index` is the index of the reference PSF along dim 3 of `psfs`. 
 Default: `ref_image_index = size(psfs)[end] ÷ 2 + 1`
"""
function generate_model(
    psfs::AbstractArray{T,N}, rank::Int, ref_image_index::Int=-1; reduce=false, itp_method=Shepard(), positions=nothing, channels=false
) where {T,N}
    if ref_image_index == -1
        # Assume reference image is in the middle
        ref_image_index = size(psfs)[end] ÷ 2 + 1
    end
    spatial_dims = channels ? ndims(psfs) - 2 : ndims(psfs) - 1
    channel_dim = channels ? ndims(psfs) - 1 : nothing
    # If no channels, reshape  PSF to have one channel
    psfs = channels ? psfs : reshape(psfs, size(psfs)[1:spatial_dims]..., 1, size(psfs, ndims(psfs)))
    my_reduce = reduce
    if reduce && spatial_dims == 2
        @info "reduce is true, but spatial dimensions are 2, so reduce is ignored"
        my_reduce = false
    end
    models_collected = []
    psfs_registration = selectdim(sqrt.(sum(abs2, psfs; dims=channel_dim)), channel_dim, 1)
    _, shifts = SpatiallyVaryingConvolution.registerPSFs(
                    psfs_registration, collect(selectdim(psfs_registration, ndims(psfs_registration), ref_image_index))
                )
    for i in axes(psfs, channel_dim)
        channel_psfs = selectdim(psfs, channel_dim, i)
        if isnothing(positions)
                psfs_reg = shift_psfs(channel_psfs, shifts)
        else
            center_pos = size(channel_psfs)[1:(end-1)] .÷2 .+ 1
            shifts = center_pos .- positions
            psfs_reg = shift_psfs(channel_psfs, shifts)
        end
        comps, weights = decompose(psfs_reg, rank)
        if spatial_dims == 3 && any(shifts[3, :] .!= zero(Int))
            weights_interp = interpolateWeights(weights, size(comps)[1:3], shifts; itp_method=itp_method)
        else
            weights_interp = interpolateWeights(weights, size(comps)[1:2], shifts[1:2, :]; itp_method=itp_method)
            if spatial_dims == 3
                # Repeat x-y interpolated weights Nz times 
                weights_interp = repeat(weights_interp, 1, 1, 1, size(psfs_reg, 3))
                # Reshape to (Ny, Nx, Nz, rank)
                weights_interp = permutedims(weights_interp, [1, 2, 4, 3])
            end
        end

        Ns = size(comps)[1:spatial_dims]
        #= Normalization of h  and weights_interp
            - PSFs at every location should have a sum of 1 -> normalize_weights
            - comps is normalized along the rank dimension according to L2 norm=#
        weights_interp_normalized = normalize_weights(weights_interp, comps)
        # Save normalized weights for later maybe
        # matwrite("normalized_weights.mat", Dict("weights"=>weights_interp))
        h = comps
        # padded values
        H = rfft(pad_nd(h, spatial_dims), 1:spatial_dims)
        padded_weights = pad_nd(weights_interp_normalized, spatial_dims)
        model = SpatiallyVaryingConvolution.createForwardmodel(
            H, padded_weights, tuple(Ns...); reduce=my_reduce
        )
        push!(models_collected, model)
    end

    if length(models_collected)==1
        return only(models_collected)
    end

    forward = let models_collected = models_collected
        function forward(x)
            out = similar(x)
            for i in axes(x, channel_dim)
                selectdim(out, channel_dim, i) .= models_collected[i](selectdim(x, channel_dim, i))
            end
            return out
        end
    end
   
    return forward
end

"""    generate_model(psfs_path::String, psf_name::String, rank::Int, ref_image_index::Int=-1; reduce=false, itp_method=Shepard(), positions=nothing)

Alternatively, a path to a mat or hdf5 file `psfs_path` can be given, where the PSFs array can be accessed with the key `psf_name`.

`positions` can be given as an array or as a string. If `positions` is a String, it will be used as a key to load the positions array from `psfs_path`.

If `channels==true`, the PSF will be assumed to have multiple channels. The channel dimension should be `ndims(PSF)-1`.
"""
function generate_model(
    psfs_path::String, psf_name::String, rank::Int, ref_image_index::Int=-1; reduce=false, itp_method=Shepard(), positions=nothing, channels=false
)
    psfs = read_psfs(psfs_path, psf_name)
    if positions isa AbstractString
        positions = read_psfs(psfs_path, positions)
    end
    return generate_model(psfs, rank, ref_image_index; reduce=reduce, itp_method=itp_method, positions=positions, channels=channels)
end
