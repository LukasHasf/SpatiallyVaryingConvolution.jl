using Images: save
using MAT: matwrite

function createForwardmodel_FLFM(
    Y,
    X,
    padded_weights,
    buf_irfft_Y,
    buf_ifftshift_y,
    H,
    buf_padded_x,
    buf_weighted_x,
    unpadded_size,
)
    N = ndims(H)
    plan = plan_rfft(buf_weighted_x, (1, 2); flags=FFTW.MEASURE)
    inv_plan = inv(plan)
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
            N = N

            function forward(x)
                # x is padded along z as well despite it not being needed. 
                # This is not very efficient, but this way, the previous code does not need to handle the FLFM parameter.
                buf_padded_x .= pad_nd(x, N - 1)
                for r in axes(padded_weights, N)
                    buf_weighted_x .= selectdim(padded_weights, N, r) .* buf_padded_x
                    mul!(X, plan, buf_weighted_x)
                    if r == 1
                        Y .= X .* selectdim(H, N, r)
                    else
                        Y .+= X .* selectdim(H, N, r)
                    end
                end
                mul!(buf_irfft_Y, inv_plan, Y)
                FFTW.ifftshift!(buf_ifftshift_y, buf_irfft_Y, (1, 2))
                return dropdims(
                    sum(unpad(buf_ifftshift_y, unpadded_size...); dims=3); dims=3
                )
            end
        end
    return forward
end

"""    createForwardmodel(H::AbstractArray{T, N}, padded_weights, unpadded_size; flfm=false) where {T, N}

Return a function that computes a spatially varying convolution defined by kernels `H` and
their padded weights `padded_weights`. The convolution accepts a three-dimensional padded
volume and returns the convolved volume.

The dimension of `H` and `padded_weights` should correspond to `(Ny, Nx[, Nz], rank)`
"""
function createForwardmodel(
    H::AbstractArray{T,N}, padded_weights, unpadded_size; flfm=false
) where {T,N}
    @assert ndims(padded_weights) == N "Weights need to be $(N)D."
    Y, X, buf_weighted_x, buf_padded_x, buf_irfft_Y, buf_ifftshift_y, plan, inv_plan = _prepare_buffers_forward(
        H, size(padded_weights)
    )
    if flfm
        return createForwardmodel_FLFM(
            Y,
            X,
            padded_weights,
            buf_irfft_Y,
            buf_ifftshift_y,
            H,
            buf_padded_x,
            buf_weighted_x,
            unpadded_size,
        )
    end

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
            N = N

            function forward(x)
                buf_padded_x .= pad_nd(x, ndims(H) - 1)
                for r in axes(padded_weights, N)
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
                return unpad(buf_ifftshift_y, unpadded_size...)
            end
        end
    return forward
end

"""
    generate_model(psfs::AbstractArray{T,N}, rank::Int[, ref_image_index::Int]; flfm=false, itp_method=Shepard(), positions=nothing) where {T,N}

Construct the forward model using the PSFs in `psfs` employing an interpolation
 of the first `rank` components calculated from a SVD. If `flfm==true`, the convolution will happen slice-wise and the slices will be added together.

The interpolation of the coefficients of the eigenimages is done using the `itp_method`.

If the PSF locations differ from their center of mass, `positions` can be supplied as an array of size `(N-1, size(psfs,N))`.

`ref_image_index` is the index of the reference PSF along dim 3 of `psfs`. 
 Default: `ref_image_index = size(psfs)[end] ÷ 2 + 1`
"""
function generate_model(
    psfs::AbstractArray{T,N},
    rank::Int,
    ref_image_index::Int=-1;
    flfm=false,
    itp_method=Shepard(),
    positions=nothing,
) where {T,N}
    if ref_image_index == -1
        # Assume reference image is in the middle
        ref_image_index = size(psfs)[end] ÷ 2 + 1
    end
    ND = ndims(psfs)
    my_flfm = flfm
    if flfm && ND == 3
        @info "flfm is true, but dimensions are 2, so flfm is ignored"
        my_flfm = false
    end
    if isnothing(positions)
        psfs_reg, shifts = SpatiallyVaryingConvolution.registerPSFs(
            psfs, collect(selectdim(psfs, N, ref_image_index))
        )
    else
        center_pos = size(psfs)[1:(end - 1)] .÷ 2 .+ 1
        shifts = center_pos .- positions
        psfs_reg = shift_psfs(psfs, shifts)
    end
    comps, weights = decompose(psfs_reg, rank)
    if N == 4 && any(shifts[3, :] .!= zero(Int))
        weights_interp = interpolateWeights(
            weights, size(comps)[1:3], shifts; itp_method=itp_method
        )
    else
        weights_interp = interpolateWeights(
            weights, size(comps)[1:2], shifts[1:2, :]; itp_method=itp_method
        )
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
    if !flfm
        # In the general case, FT along all spatial dims
        H = rfft(pad_nd(h, ND - 1), 1:(ND - 1))
    else
        # In FLFM, FT only along (x,y)
        H = rfft(pad_nd(h, ND - 1), 1:(ND - 2))
    end
    padded_weights = pad_nd(weights_interp_normalized, ND - 1)
    model = SpatiallyVaryingConvolution.createForwardmodel(
        H, padded_weights, tuple(Ns...); flfm=my_flfm
    )
    return model
end

"""    generate_model(psfs_path::String, psf_name::String, rank::Int, ref_image_index::Int=-1; flfm=false, itp_method=Shepard(), positions=nothing)

Alternatively, a path to a mat or hdf5 file `psfs_path` can be given, where the PSFs array can be accessed with the key `psf_name`.

`positions` can be given as an array or as a string. If `positions` is a String, it will be used as a key to load the positions array from `psfs_path`.
"""
function generate_model(
    psfs_path::String,
    psf_name::String,
    rank::Int,
    ref_image_index::Int=-1;
    flfm=false,
    itp_method=Shepard(),
    positions=nothing,
)
    psfs = read_psfs(psfs_path, psf_name)
    if positions isa AbstractString
        positions = read_psfs(psfs_path, positions)
    end
    return generate_model(
        psfs, rank, ref_image_index; flfm=flfm, itp_method=itp_method, positions=positions
    )
end
