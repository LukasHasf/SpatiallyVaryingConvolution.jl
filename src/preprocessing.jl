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
    good_indices = []
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
        push!(good_indices, m)
        good_count += 1
    end

    if N == 4 && maximum(abs.(si[3, :])) > zero(eltype(si))
        for ind in good_indices
            selected_stack = selectdim(stack, ND, ind)
            linshift!(im_reg, selected_stack, si[:, ind])
            selectdim(yi_reg, ND, ind) .= im_reg
        end
    end

    # Populate yi_reg
    for ind in good_indices
        circshift!(im_reg, selectdim(stack, ND, ind), si[:, ind])
        selectdim(yi_reg, ND, ind) .= im_reg
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
    @assert length(shape) == size(si, 1)
    rnk = size(weights)[2]

    coo_q = [(-N / 2):((N - 1) / 2) for N in shape]
    if length(shape) == 2
        gridPoints = vcat(([x y] for y in coo_q[1] for x in coo_q[2])...)'
    elseif length(shape) == 3
        gridPoints =
            vcat(([x y z] for z in coo_q[3] for y in coo_q[1] for x in coo_q[2])...)'
    end
    xyz_indices = [2 1 3]
    coo_s = [-si[xyz_indices[i], :] for i in 1:size(si, 1)]

    new_shape = [shape[xyz_indices[i]] for i in 1:length(shape)]
    weights_interp = similar(weights, shape..., rnk)
    points = T.(hcat([coo_s[i] for i in 1:size(si, 1)]...)')
    itp_methods = [NearestNeighbor(), Multiquadratic(), Shepard()]
    for r in 1:rnk
        itp = ScatteredInterpolation.interpolate(itp_methods[1], points, weights[:, r])
        interpolated = evaluate(itp, gridPoints)
        if length(shape) == 2
            interpolated = reshape(interpolated, new_shape...)'
        elseif length(shape) == 3
            interpolated = permutedims(reshape(interpolated, new_shape...), [2 1 3])
        end
        selectdim(weights_interp, ndims(weights_interp), r) .= interpolated
    end
    return weights_interp
end
