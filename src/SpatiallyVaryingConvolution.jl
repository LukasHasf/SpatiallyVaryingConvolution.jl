module SpatiallyVaryingConvolution

include("utils.jl")
using NDTools
using FourierTools
using FFTW
using Arpack, LinearAlgebra
using ScatteredInterpolation

export generateModel, readPSFs
"""
    registerPSFs(stack, ref_im)

Find the shift between each PSF in `stack` and the reference PSF in `ref_im`
and return the aligned PSFs and their shifts.
 
If `ref_im` has size `(Ny, Nx)`/`(Ny, Nx, Nz)`, `stack` should have size
 `(Ny, Nx, nrPSFs)`/`(Ny, Nx, Nz, nrPSFs)`.
"""
function registerPSFs(stack::AbstractArray{T,N}, ref_im) where {T,N}
    @assert N in [3, 4] "stack needs to be a 3d/4d array but was $(N)d"
    ND = ndims(stack)
    Ns = size(stack)[1:(end - 1)]
    ps = Ns # Relative centers of all correlations
    M = size(stack)[end]
    pad_function = x -> padND(x, ND - 1)

    function crossCorr(
        x::AbstractArray{Complex{T}},
        y::AbstractArray{Complex{T}},
        iplan::AbstractFFTs.ScaledPlan,
    )
        return fftshift(iplan * (x .* y))
    end

    function norm(x)
        return sqrt(sum(abs2.(x)))
    end

    yi_reg = similar(stack, size(stack)...)
    stack_dct = copy(stack)
    ref_norm = norm(ref_im) # norm of ref_im

    # Normalize the stack
    norms = map(norm, eachslice(stack_dct; dims=ND))
    norms = reshape(norms, ones(Int, ND - 1)..., length(norms))
    stack_dct ./= norms
    ref_im ./= ref_norm

    si = similar(ref_im, Int, ND - 1, M)
    # Do FFT registration
    good_count = 1
    dummy_for_plan = similar(stack_dct, (2 .* Ns)...)
    plan = plan_rfft(dummy_for_plan; flags=FFTW.MEASURE)
    iplan = inv(plan)
    pre_comp_ref_im = conj.(plan * (pad_function(ref_im)))
    im_reg = similar(stack_dct, Ns...)
    ft_stack = similar(stack_dct, Complex{T}, (2 * Ns[1]) ÷ 2 + 1, (2 .* Ns[2:end])...)
    padded_stack_dct = pad_function(stack_dct)
    for m in 1:M
        mul!(ft_stack, plan, selectdim(padded_stack_dct, ND, m))
        corr_im = crossCorr(ft_stack, pre_comp_ref_im, iplan)
        max_value, max_location = findmax(corr_im)
        if max_value < 0.01
            println("Image $m has poor quality. Skipping")
            continue
        end

        si[:, good_count] .= 1 .+ ps .- max_location.I
        #= TODO: Circshifting is probably not the best way to move the PSF, especially considering 3D
           - Padding, shifting and cropping may be better as long as noise is low enough to not introduce serious 
           sinc artifacts
           - Circshifting in the x-y plane might be okay if autocorrelation length of the noise is small
                and noise is uncorrelated to PSF location
            - Circshifting in z would only work if there is enough padding of very low signal slices at both
                ends of the z-axis. Otherwise top parts of the PSF might become bottom parts of the PSF or vice versa=#
        circshift!(im_reg, selectdim(stack, ND, m), si[:, good_count])
        selectdim(yi_reg, ND, good_count) .= im_reg
        good_count += 1
    end
    return collect(selectdim(yi_reg, ND, 1:(good_count - 1))), si
end

"""
    decompose(yi_reg, rnk)

Calculate the SVD of a collection of PSFs `yi_reg` with reduced rank `rnk`.

`yi_reg` is expected to have shape `(Ny, Nx, nrPSFs)`/`(Ny, Nx, Nz, nrPSFs)`. Returns the `rnk`
components and the weights to reconstruct the original PSFs. `rnk` needs 
to be smaller than `nrPSFs`.
"""
function decompose(yi_reg::AbstractArray{T,N}, rnk) where {T,N}
    Ns = size(yi_reg)[1:(N - 1)]
    nrPSFs = size(yi_reg)[end]
    ymat = reshape(yi_reg, (prod(Ns), nrPSFs))

    Z = svds(ymat; nsv=rnk)[1]
    comps = reshape(Z.U, (Ns..., rnk))
    weights = similar(yi_reg, nrPSFs, rnk)
    mul!(weights, Z.V, LinearAlgebra.Diagonal(Z.S))
    return comps, weights
end

"""
    interpolate_weights(weights, shape, si)

Interpolate `weights` defined at positions `si` onto a grid of size `shape`.
"""
function interpolateWeights(weights::AbstractArray{T,N}, shape, si) where {T,N}
    Ny, Nx = shape
    rnk = size(weights)[2]

    xq = (-Nx / 2):((Nx - 1) / 2)
    yq = ((-Ny / 2):((Ny - 1) / 2))'
    X = repeat(xq, Ny)[:]
    Y = repeat(yq, Nx)[:]
    gridPoints = [X Y]'
    xi = -si[2, :]
    yi = -si[1, :]

    weights_interp = similar(weights, Ny, Nx, rnk)
    points = T.([xi yi]')
    itp_methods = [NearestNeighbor(), Multiquadratic(), Shepard()]
    for r in 1:rnk
        itp = ScatteredInterpolation.interpolate(itp_methods[1], points, weights[:, r])
        weights_interp[:, :, r] .= reshape(evaluate(itp, gridPoints), (Nx, Ny))'
    end
    return weights_interp
end

"""    createForwardmodel(H::AbstractArray{T, N}, padded_weights, unpadded_size) where {T, N}

Return a function that computes a spatially varying convolution defined by kernels `H` and
their padded weights `padded_weights`. The convolution accepts a three-dimensional padded
volume and returns the convolved volume.

The dimension of `H` and `padded_weights` should correspond to `(Ny, Nx[, Nz], rank)`
"""
function createForwardmodel(
    H::AbstractArray{T,N}, padded_weights, unpadded_size
) where {T,N}
    @assert ndims(padded_weights) == N "Weights need to be $(N)D."
    Y, X, buf_weighted_x, buf_irfft_Y, buf_ifftshift_y, plan, inv_plan = _prepare_buffers_forward(
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
            buf_weighted_x = buf_weighted_x,
            unpadded_size = unpadded_size,
            N = N

            function forward(x)
                for r in 1:size(padded_weights)[end]
                    buf_weighted_x .= selectdim(padded_weights, N, r) .* x
                    #buf_weighted_x .= view(padded_weights, :, :, :, r) .* x
                    mul!(X, plan, buf_weighted_x)
                    if r == 1
                        Y .= X .* selectdim(H, N, r)
                        #Y .= X .* view(H, :, :, :, r)
                    else
                        Y .+= X .* selectdim(H, N, r)
                        #Y .+= X .* view(H, :, :, :, r)
                    end
                end
                mul!(buf_irfft_Y, inv_plan, Y)
                ifftshift!(buf_ifftshift_y, buf_irfft_Y)
                return unpad(buf_ifftshift_y, unpadded_size...)
            end
        end
    return forward
end

"""
    generate_model(psfs::AbstractArray{T,3}, rank::Int[, ref_image_index::Int])

Construct the forward model using the PSFs in `psfs` employing an interpolation
 of the first `rank` components calculated from a SVD.

`ref_image_index` is the index of the reference PSF along dim 3 of `psfs`. 
 Default: `ref_image_index = size(psfs)[end] ÷ 2 + 1`
"""
function generateModel(
    psfs::AbstractArray{T,N}, rank::Int, ref_image_index::Int=-1
) where {T,N}
    if ref_image_index == -1
        # Assume reference image is in the middle
        ref_image_index = size(psfs)[end] ÷ 2 + 1
    end
    ND = ndims(psfs)
    psfs_reg, shifts = SpatiallyVaryingConvolution.registerPSFs(
        psfs, collect(selectdim(psfs, N, ref_image_index))
    )
    if N == 4 && any(shifts[3, :] .!= zero(Int))
        # If PSFs are shifted in z, decomposition and interpolation have to be done z-slice weights_interp
        comps = similar(psfs_reg, size(psfs_reg)[1:(end-1)]..., rank)
        weights_interp = similar(comps)
        for i in 1:size(psfs_reg, 3)
            temp_comps, weights = decompose(psfs_reg[:, :, i, :], rank)
            weights_interp[:, :, i, :] .= interpolateWeights(
                weights, size(comps)[1:2], shifts
            )
            comps[:, :, i, :] .= temp_comps
        end
    else
        comps, weights = decompose(psfs_reg, rank)
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
    h = comps

    # padded values

    H = rfft(padND(h, ND - 1), 1:(ND - 1))
    flatfield = padND(ones(Float64, Ns...), ND - 1)
    padded_weights = padND(weights_interp, ND - 1)
    model = SpatiallyVaryingConvolution.createForwardmodel(H, padded_weights, tuple(Ns...))
    flatfield_sim = model(flatfield)
    svc_model = let flatfield_sim = flatfield_sim, model = model
        function svc_model(x)
            return model(x) ./ flatfield_sim
        end
    end
    return svc_model
end

function generateModel(
    psfs_path::String, psf_name::String, rank::Int, ref_image_index::Int=-1
)
    psfs = readPSFs(psfs_path, psf_name)
    return generateModel(psfs, rank, ref_image_index)
end

end # module
