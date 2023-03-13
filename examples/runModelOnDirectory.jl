using SpatiallyVaryingConvolution
using Images
using MAT
using ProgressMeter
using NDTools

function _map_to_zero_one!(x, min_x, max_x)
    if min_x == max_x
        return x
    end
    x .-= min_x
    return x .*= inv(max_x - min_x)
end

function img_to_rgb_array(img)
    r = red.(img)
    g = green.(img)
    b = blue.(img)
    out = Float64.(cat(r, g, b; dims=3))
    return out
end

function rgb_array_to_img(rgb)
    out_color = colorview(RGB, rgb[:, :, 1], rgb[:, :, 2], rgb[:, :, 3])
    return out_color
end

function _load(path; key="gt")
    if endswith(path, ".mat")
        return matread(path)[key]
    else
        load(path)
    end
end

function iterate_over_images(sourcedir, destinationdir, sourcefiles, model, newsize; scaling=1, channels=false)
    p = Progress(length(sourcefiles))
    for sourcefile in sourcefiles
        if isdir(joinpath(sourcedir, sourcefile))
            continue
        end
        img_path = joinpath(sourcedir, sourcefile)
        destination_path = joinpath(destinationdir, sourcefile)
        img = reverse(load(img_path); dims=1)
        img = imresize(img, newsize .÷ scaling)
        img = channels ? img_to_rgb_array(img) : Float64.(Gray.(img))
        img = select_region(img; new_size=newsize, pad_value=1e-2)
        if scaling != 1
            destination_path = joinpath(destinationdir, "forward", sourcefile)
            gt_path = joinpath(destinationdir, "groundtruth", sourcefile)
            gt = copy(img)
            _map_to_zero_one!(gt, extrema(gt)...)
            save(gt_path, colorview(Gray, reverse(gt; dims=1)))
        end
        sim = model(img)
        _map_to_zero_one!(sim, extrema(sim)...)
        sim = reverse(sim; dims=1)
        if channels
            save(destination_path, rgb_array_to_img(sim))
        else
            save(destination_path, colorview(Gray, sim))
        end
        ProgressMeter.next!(p; showvalues=[(:image, img_path)])
    end
end

function iterate_over_volumes(
    sourcedir, destinationdir, sourcefiles, model, newsize; key="gt"
)
    p = Progress(length(sourcefiles))
    for sourcefile in sourcefiles
        vol_path = joinpath(sourcedir, sourcefile)
        destination_path = joinpath(destinationdir, sourcefile)
        try
            vol = matread(vol_path)[key]
            vol = imresize(vol, newsize)
            sim = model(vol)
            mi, ma = extrema(sim)
            _map_to_zero_one!(sim, mi, ma)
            matwrite(destination_path, Dict("sim" => sim))
        catch EOFError
            #rm(vol_path)
            continue
        end
        ProgressMeter.next!(p; showvalues=[(:volume, vol_path)])
    end
end

"""    run_forwardmodel(sourcedir, destinationdir, psfpath, psfname; amount=-1, ref_image_index=-1, rank=4, positions=nothing, scaling=1)

Construct a forward model with the PSFs in `psfpath` at key `psfname`. 
The model is constructed with the options given by `rank`, `ref_image_index` and `positions`.

Images are read from `sourcedir`, convolved, and the output is saved in `destinationdir`. 

`amount` images will be processed. `-1` for all images available in `sourcedir`.

`scaling` relates to FLFM. It should be the maximum number of microlenses visible in one row or column. The image is scaled accordingly and zero-padded to original size
 before convolution.
"""
function run_forwardmodel(
    sourcedir, destinationdir, psfpath, psfname; amount=-1, ref_image_index=-1, rank=4, positions=nothing, scaling=1, channels=false
)
    psfs = matread(psfpath)[psfname]
    # For numerical stability, don't have zeros in PSF for FLFM
    if scaling != 1 && minimum(psfs) == 0
        psfs[psfs .== 0] .= 1e-7
    end
    model = generate_model(psfs, rank, ref_image_index, positions=positions, channels=channels)
    sourcefiles = amount == -1 ? readdir(sourcedir) : readdir(sourcedir)[1:amount]
    spatial_dims = channels ? ndims(psfs) - 2 : ndims(psfs) - 1
    newsize = size(psfs)[1:spatial_dims]
    isdir(destinationdir) || mkpath(destinationdir)
    if length(newsize) == 2
        iterate_over_images(sourcedir, destinationdir, sourcefiles, model, newsize, scaling=scaling, channels=channels)
    elseif length(newsize) == 3
        iterate_over_volumes(sourcedir, destinationdir, sourcefiles, model, newsize)
    end
end
