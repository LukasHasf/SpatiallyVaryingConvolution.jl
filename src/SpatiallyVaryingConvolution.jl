module SpatiallyVaryingConvolution

using MAT, HDF5
using NDTools
using FourierTools
using FFTW
using Arpack, LinearAlgebra
using ScatteredInterpolation

"""    
    readPSFs(path::String, key::String)

Read the PSFs stored in file `path` accessible as field `key`. 

Supports MAT and HDF5 file format. 
PSFs are expected to be stored as an array at `key` having shape
 (Ny, Nx, nrPSFs) in the 2D case or (Ny, Nx, nrPSFs, Nz) in the 3D case.
 # Examples
 ```
 julia> readPSFs("myPSFs.mat", "PSFs")
 ```
 ```
 julia> readPSFs("myPSFs.h5", "myPSFsDataset")
 ```
"""
function readPSFs(path::String, key::String)
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

"""
    registerPSFs(stack, ref_im)

Find the shift between each PSF in `stack` and the reference PSF in `ref_im`
and return the aligned PSFs and their shifts.
 
`stack` is expected to have shape `(Ny, Nx, nrPSfs)` and `ref_im` `(Ny, Nx)`.
"""
function registerPSFs(stack, ref_im)
    Ny, Nx, M = size(stack)

    function crossCorr(
        x::Matrix{ComplexF64},
        y::Matrix{ComplexF64},
        iplan::AbstractFFTs.ScaledPlan,
    )
        return fftshift(real.(iplan * (x .* y)))
    end

    function norm(x)
        return sqrt(sum(abs2.(x)))
    end
    pr = Ny
    pc = Nx # Relative centers of all correlations

    yi_reg = Array{Float64,3}(undef, size(stack))
    stack_dct = copy(stack)
    ref_norm = norm(ref_im) # norm of ref_im

    # Normalize the stack
    #div.(stack, ref_norm)
    for m = 1:M
        stack_dct[:, :, m] ./= norm(stack_dct[:, :, m])
    end
    ref_im ./= ref_norm

    si = zeros((2, M))
    # Do FFT registration
    println("Registering")
    good_count = 0
    dummy_for_plan = Array{eltype(stack_dct),2}(undef, (2 * Ny, 2 * Nx))
    plan = plan_rfft(dummy_for_plan)
    dummy_for_iplan = Array{ComplexF64,2}(undef, ((2 * Ny) ÷ 2 + 1, 2 * Nx))
    iplan = plan_irfft(dummy_for_iplan, size(dummy_for_plan)[1])
    pre_comp_ref_im = conj.(plan * (pad2D(ref_im)))
    for m = 1:M
        corr_im = crossCorr(plan * pad2D(stack_dct[:, :, m]), pre_comp_ref_im, iplan)
        if maximum(corr_im) < 0.01
            println("Image " * string(m) * " has poor quality. Skipping")
            continue
        end
        r, c = Tuple(findmax(corr_im)[2])

        si[:, good_count+1] .= [1 + pr - r, 1 + pc - c]

        #im_reg = ref_norm .* circshift(stack[:, :, m], si[:, good_count+1])
        im_reg = circshift(stack[:, :, m], si[:, good_count+1])
        yi_reg[:, :, good_count+1] = im_reg
        good_count += 1
    end
    yi_reg = yi_reg[:, :, 1:good_count]
    println("Done registering")
    return yi_reg, si
end

"""
    decompose(yi_reg, rnk)

Calculate the SVD of a collection of PSFs `yi_reg` with reduced rank `rnk`.

`yi_reg` is expected to have shape `(Ny, Nx, nrPSFs)`. Returns the `rnk`
components and the weights to reconstruct the original PSFs. `rnk` needs 
to be smaller than `nrPSFs`.
"""
function decompose(yi_reg, rnk)
    Ny, Nx, Mgood = size(yi_reg)
    println("Creating matrix")
    ymat = reshape(yi_reg, (Ny * Nx, Mgood))
    println("Done")

    println("Starting SVD...")
    Z = svds(ymat; nsv = rnk)[1]
    comps = reshape(Z.U, (Ny, Nx, rnk))
    weights = Array{Float64,2}(undef, (Mgood, rnk))
    mul!(weights, Z.V, LinearAlgebra.Diagonal(Z.S))
    return comps, weights
end

"""
    interpolate_weights(weights, shape, si)

Interpolate `weights` defined at positions `si` onto a grid of size `shape`.
"""
function interpolate_weights(weights, shape, si)
    Ny, Nx = shape
    rnk = size(weights)[2]

    xq = -Nx/2:(Nx-1)/2
    yq = (-Ny/2:(Ny-1)/2)'
    X = repeat(xq, Ny)[:]
    Y = repeat(yq, Nx)[:]
    gridPoints = [X Y]'
    xi = -si[2, :]
    yi = -si[1, :]

    println("Interpolating...")
    weights_interp = Array{Float64,3}(undef, (Ny, Nx, rnk))
    points = Float64.([xi yi]')
    itp_methods = [NearestNeighbor(), Multiquadratic(), Shepard()]
    for r = 1:rnk
        itp = ScatteredInterpolation.interpolate(itp_methods[1], points, weights[:,r])
        weights_interp[:, :, r] .= reshape(evaluate(itp, gridPoints), (Nx, Ny))'
    end
    return weights_interp
end

function pad2D(x)
    return select_region(x, new_size=2 .* size(x)[1:2], pad_value=zero(eltype(x)))
end

function crop2D(x, rcL, rcU, ccL, ccU)
    return x[rcL:rcU, ccL:ccU]
end

"""
    create_forwardmodel(H::Array{T, 3}, padded_weights, crop_indices) where T

Return a function that computes a spatially varying convolution defined by kernels `H` and
    their padded weights `padded_weights`.

Expects `H` to be a stack of Fourier-transformed and padded spatially invariant kernels,
`padded_weights` to be the zero-padded weight of each kernel at each (x,y)-coordinate and
`crop_indices` the indices to use for cropping after the padded convolution.
"""
function create_forwardmodel(H::Array{T, 3}, padded_weights, crop_indices) where T
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
            crop2D(buf_ifftshift_y, crop_indices...)
        end
    end
    return forward
end

"""
    generate_model(psfs::Array{T,3}, rank::Int[, ref_image_index::Int])

Construct the forward model using the PSFs in ```psfs``` employing an interpolation
 of the first ```rank``` components calculated from a SVD.

```ref_image_index``` is the index of the reference PSF along dim 3 of ```psfs```. 
 Default: ```ref_image_index = size(psfs)[end] ÷ 2 + 1```
"""
function generate_model(psfs::Array{T, 3},rank::Int, ref_image_index::Int=-1) where T
    if ref_image_index == -1
        # Assume reference image is in the middle
        ref_image_index = size(psfs)[end] ÷ 2 + 1
    end
    psfs_reg, shifts =
        SpatiallyVaryingConvolution.registerPSFs(psfs[:, :, :], psfs[:, :, ref_image_index])
    comps, weights = decompose(psfs_reg, rank)
    weights_interp = interpolate_weights(weights, size(comps)[1:2], shifts)
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
    lower_index(N) = Bool(N % 2) ? (N+3)÷2 : (N+2)÷2
    upper_index(N) = Bool(N % 2) ?  3*N÷2 +1 : 3*N÷2
    ccL = lower_index(Nx)
    ccU = upper_index(Nx)
    rcL = lower_index(Ny)
    rcU = upper_index(Ny)
    H = rfft(pad2D(h), [1,2])
    flatfield = pad2D(ones(Float64, (Ny,Nx)))
    padded_weights = pad2D(weights_interp)
    model = SpatiallyVaryingConvolution.create_forwardmodel(
        H,
        padded_weights,
        [rcL, rcU, ccL, ccU]
    )
    flatfield_sim = model(flatfield)
    svc_model = let flatfield_sim=flatfield_sim, model=model
        function svc_model(x)
            return model(x)./flatfield_sim
        end
    end
    return svc_model
end

function generate_model(psfs_path::String, psf_name::String,rank::Int, ref_image_index::Int=-1)
    psfs = readPSFs(psfs_path, psf_name)
    return generate_model(psfs, rank, ref_image_index)
end

end # module
