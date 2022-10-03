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

            function forward(x; pad_value=zero(eltype(x)))
                buf_padded_x .= padND(x, ndims(H) - 1; pad_value=pad_value)
                save_image("buf_padded_x.png", buf_padded_x)
                for r in 1:size(padded_weights)[end]
                    buf_weighted_x .= selectdim(padded_weights, N, r) .* buf_padded_x
                    mul!(X, plan, buf_weighted_x)
                    if r == 1
                        Y .= X .* selectdim(H, N, r)
                    else
                        Y .+= X .* selectdim(H, N, r)
                    end
                end
                save_image("Y.png", abs2.(Y))
                mul!(buf_irfft_Y, inv_plan, Y)
                FFTW.ifftshift!(buf_ifftshift_y, buf_irfft_Y)
                save_image("y.png", buf_ifftshift_y)
                if reduce
                    return dropdims(
                        sum(unpad(buf_ifftshift_y, unpadded_size...); dims=3); dims=3
                    )
                else
                    println("Unpadding in forward model")
                    unpadded = unpad(buf_ifftshift_y, unpadded_size...)
                    save_image("unpadded.png", unpadded)
                    return unpadded
                end
            end
        end
    return forward
end

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
    #= TODO: Normalization of h  and weights_interp
        - PSFs at every location should have a sum of 1 (?)
        - comps is normalized along the rank dimension according to L2 norm=#
    # Save normalized weights for later maybe
    matwrite("normalized_weights.mat", Dict("weights"=>weights_interp))
    h = comps

    # padded values

    H = rfft(padND(h, ND - 1), 1:(ND - 1))
    flatfield = similar(psfs, Ns...)
    fill!(flatfield, one(eltype(flatfield)))
    padded_weights = padND(weights_interp, ND - 1)
    model = SpatiallyVaryingConvolution.createForwardmodel(
        H, padded_weights, tuple(Ns...); reduce=my_reduce
    )
    summed_weights = sum(abs2, padded_weights; dims=3)
    mi, ma = extrema(summed_weights[:, :, 1])
    save("summed_weights_1.png", _map_to_zero_one!(summed_weights[:, :, 1], mi , ma))
    flatfield_sim = model(flatfield)
    save_image("flatfield_sim.png", flatfield_sim)
    svc_model = let flatfield_sim = flatfield_sim, model = model
        function svc_model(x)
            return model(x) #./ flatfield_sim
        end
    end
    return svc_model
end

function generateModel(
    psfs_path::String, psf_name::String, rank::Int, ref_image_index::Int=-1; reduce=false
)
    psfs = readPSFs(psfs_path, psf_name)
    return generateModel(psfs, rank, ref_image_index; reduce=reduce)
end
