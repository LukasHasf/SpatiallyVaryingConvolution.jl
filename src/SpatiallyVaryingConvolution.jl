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
function registerPSFs(stack::Array{T,N}, ref_im) where {T,N}
    @assert N in [3, 4] "stack needs to be a 3d/4d array but was $(N)d"
    ND = ndims(stack)
    Ns = Array{Int, 1}(undef, ND-1)
    Ns .= size(stack)[1:end-1]
    ps = Ns # Relative centers of all correlations
    M = size(stack)[end]
    pad_function = N == 3 ? pad2D : pad3D

    function crossCorr(
            x::Array{ComplexF64},
            y::Array{ComplexF64},
            iplan::AbstractFFTs.ScaledPlan,
        )
            return fftshift(iplan * (x .* y))
    end
    
    function norm(x)
        return sqrt(sum(abs2.(x)))
    end

    yi_reg = Array{Float64, N}(undef, size(stack))
    stack_dct = copy(stack)
    ref_norm = norm(ref_im) # norm of ref_im

    # Normalize the stack
    norms = map(norm, eachslice(stack_dct, dims=ND))
    norms = reshape(norms, ones(Int, ND-1)...,length(norms))
    stack_dct ./= norms
    ref_im ./= ref_norm

    si = zeros(Int, (ND-1, M))
    # Do FFT registration
    good_count = 1
    dummy_for_plan = Array{eltype(stack_dct), ND-1}(undef, (2 .* Ns)...)
    plan = plan_rfft(dummy_for_plan, flags = FFTW.MEASURE)
    dummy_for_iplan = Array{ComplexF64, ND-1}(undef, (2 * Ns[1]) ÷ 2 + 1, (2 .* Ns[2:end])...)
    iplan = plan_irfft(dummy_for_iplan, size(dummy_for_plan)[1], flags = FFTW.MEASURE)
    pre_comp_ref_im = conj.(plan * (pad_function(ref_im)))
    im_reg = Array{Float64, ND-1}(undef, Ns...)
    ft_stack = Array{ComplexF64, ND-1}(undef, (2 * Ns[1]) ÷ 2 + 1, (2 .* Ns[2:end])...)
    padded_stack_dct = pad_function(stack_dct)
    for m = 1:M
        mul!(ft_stack, plan, selectdim(padded_stack_dct, ND, m))
        corr_im = crossCorr(ft_stack, pre_comp_ref_im, iplan)
        max_value, max_location = findmax(corr_im)
        if max_value < 0.01
            println("Image $m has poor quality. Skipping")
            continue
        end

        si[:, good_count] .= 1 .+ ps .- max_location.I
        circshift!(im_reg, selectdim(stack, ND, m), si[:, good_count])
        selectdim(yi_reg, ND, good_count) .= im_reg
        good_count += 1
    end
    return collect(selectdim(yi_reg, ND, 1:(good_count-1))), si
end

"""
    decompose(yi_reg, rnk)

Calculate the SVD of a collection of PSFs `yi_reg` with reduced rank `rnk`.

`yi_reg` is expected to have shape `(Ny, Nx, nrPSFs)`/`(Ny, Nx, Nz, nrPSFs)`. Returns the `rnk`
components and the weights to reconstruct the original PSFs. `rnk` needs 
to be smaller than `nrPSFs`.
"""
function decompose(yi_reg::Array{T, N}, rnk) where {T,N}
    Ns = size(yi_reg)[1:N-1]
    nrPSFs = size(yi_reg)[end]
    ymat = reshape(yi_reg, (prod(Ns), nrPSFs))

    Z = svds(ymat; nsv = rnk)[1]
    comps = reshape(Z.U, (Ns..., rnk))
    weights = Array{Float64,2}(undef, (nrPSFs, rnk))
    mul!(weights, Z.V, LinearAlgebra.Diagonal(Z.S))
    return comps, weights
end

"""
    interpolate_weights(weights, shape, si)

Interpolate `weights` defined at positions `si` onto a grid of size `shape`.
"""
function interpolateWeights(weights, shape, si)
    Ny, Nx = shape
    rnk = size(weights)[2]

    xq = -Nx/2:(Nx-1)/2
    yq = (-Ny/2:(Ny-1)/2)'
    X = repeat(xq, Ny)[:]
    Y = repeat(yq, Nx)[:]
    gridPoints = [X Y]'
    xi = -si[2, :]
    yi = -si[1, :]

    weights_interp = Array{Float64,3}(undef, (Ny, Nx, rnk))
    points = Float64.([xi yi]')
    itp_methods = [NearestNeighbor(), Multiquadratic(), Shepard()]
    for r = 1:rnk
        itp = ScatteredInterpolation.interpolate(itp_methods[1], points, weights[:,r])
        weights_interp[:, :, r] .= reshape(evaluate(itp, gridPoints), (Nx, Ny))'
    end
    return weights_interp
end

"""
    create_forwardmodel(H::Array{T, 3}, padded_weights, unpadded_size) where T

Return a function that computes a spatially varying convolution defined by kernels `H` and
    their padded weights `padded_weights`.

Expects `H` to be a stack of Fourier-transformed and padded spatially invariant kernels,
`padded_weights` to be the zero-padded weight of each kernel at each (x,y)-coordinate and
`crop_indices` the indices to use for cropping after the padded convolution.
"""
function createForwardmodel(H::Array{T, 3}, padded_weights, unpadded_size) where T
    # The size of all buffers is the size of the padded_weights
    size_x = size(padded_weights)[1:2]
    # X holds the FT of the weighted image
    # Y aggregates the FT of the convolution of the weighted image and the PSF components 
    Y = zeros(T, size_x[1]÷2 + 1, size_x[2])
    X = similar(Y)
    # RFFT and IRRFT plans
    plan = plan_rfft(Array{real(T), 2}(undef, size_x...), flags=FFTW.MEASURE)
    inv_plan = plan_irfft(Y, size_x[1])
    # Buffers for the weighted image and the irfft-ed and ifftshift-ed convolution images
    buf_weighted_x = Array{real(T), 2}(undef, size_x...)
    buf_irfft_Y = Array{real(T),2}(undef, size_x...)
    buf_ifftshift_y = Array{real(T),2}(undef, size_x...)
    forward = let Y=Y, X=X, plan=plan, padded_weights=padded_weights, buf_irfft_Y=buf_irfft_Y, buf_ifftshift_y=buf_ifftshift_y
        function forward(x)
            for r = 1:size(padded_weights)[3]
                buf_weighted_x .= view(padded_weights, :, :, r) .* x
                mul!(X, plan, buf_weighted_x)
                if r==1
                    Y .= X .* view(H, :, :, r)
                else
                    Y .+= X .* view(H, :, :, r)
                end
            end
            mul!(buf_irfft_Y, inv_plan, Y)
            ifftshift!(buf_ifftshift_y, buf_irfft_Y)
            unpad2D(buf_ifftshift_y, unpadded_size...)
        end
    end
    return forward
end

"""
    generate_model(psfs::Array{T,3}, rank::Int[, ref_image_index::Int])

Construct the forward model using the PSFs in `psfs` employing an interpolation
 of the first `rank` components calculated from a SVD.

`ref_image_index` is the index of the reference PSF along dim 3 of `psfs`. 
 Default: `ref_image_index = size(psfs)[end] ÷ 2 + 1`
"""
function generateModel(psfs::Array{T, 3},rank::Int, ref_image_index::Int=-1) where T
    if ref_image_index == -1
        # Assume reference image is in the middle
        ref_image_index = size(psfs)[end] ÷ 2 + 1
    end
    psfs_reg, shifts =
        SpatiallyVaryingConvolution.registerPSFs(psfs[:, :, :], psfs[:, :, ref_image_index])
    comps, weights = decompose(psfs_reg, rank)
    weights_interp = interpolateWeights(weights, size(comps)[1:2], shifts)
    norms = zeros(size(comps)[1:2])
    sums_of_comps = [sum(comps[:, :, i]) for i = 1:rank]
    for x = 1:size(norms)[2]
        for y = 1:size(norms)[1]
            norms[y, x] = sum(weights_interp[y, x, :] .* sums_of_comps)
        end
    end
    weights_interp ./= norms
    # Normalize components
    h = comps ./ sqrt.(sum(abs2.(comps)))
    # padded values for 2D
    Ny, Nx = size(comps)[1:2]
    H = rfft(pad2D(h), [1,2])
    flatfield = pad2D(ones(Float64, (Ny,Nx)))
    padded_weights = pad2D(weights_interp)
    model = SpatiallyVaryingConvolution.createForwardmodel(
        H,
        padded_weights,
        (Ny, Nx)
    )
    flatfield_sim = model(flatfield)
    svc_model = let flatfield_sim=flatfield_sim, model=model
        function svc_model(x)
            return model(x)./flatfield_sim
        end
    end
    return svc_model
end

function generateModel(psfs_path::String, psf_name::String,rank::Int, ref_image_index::Int=-1)
    psfs = readPSFs(psfs_path, psf_name)
    return generateModel(psfs, rank, ref_image_index)
end

end # module
