using SpatiallyVaryingConvolution
using Images
using MAT
using ProgressMeter

function _map_to_zero_one!(x, min_x, max_x)
    if min_x == max_x
        return x
    end
    x .-= min_x
    return x .*= inv(max_x - min_x)
end

function _load(path; key="gt")
    if endswith(path, ".mat")
        return matread(path)[key]
    else
        load(path)
    end
end

function iterate_over_images(sourcedir, destinationdir, sourcefiles, model, newsize)
    p = Progress(length(sourcefiles))
    for sourcefile in sourcefiles
        if isdir(joinpath(sourcedir, sourcefile))
            continue
        end
        img_path = joinpath(sourcedir, sourcefile)
        destination_path = joinpath(destinationdir, sourcefile)
        img = reverse(load(img_path); dims=1)
        img = imresize(img, newsize)
        img = Float64.(Gray.(img))
        sim = model(img)
        _map_to_zero_one!(sim, extrema(sim)...)
        save(destination_path, colorview(Gray, reverse(sim; dims=1)))
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
            rm(vol_path)
            continue
        end
        ProgressMeter.next!(p; showvalues=[(:volume, vol_path)])
    end
end

function run_forwardmodel(model, sourcedir, destinationdir; amount=-1, newsize=(64, 64))
    sourcefiles = amount == -1 ? readdir(sourcedir) : readdir(sourcedir)[1:amount]
    isdir(destinationdir) || mkpath(destinationdir)
    if length(newsize) == 2
        iterate_over_images(sourcedir, destinationdir, sourcefiles, model, newsize)
    elseif length(newsize) == 3
        iterate_over_volumes(sourcedir, destinationdir, sourcefiles, model, newsize)
    end
end

function run_forwardmodel(
    sourcedir, destinationdir, psfpath, psfname; amount=-1, ref_image_index=-1, rank=4, scaling=nothing
)
    model = generate_model(psfpath, psfname, rank; ref_image_index=ref_image_index, scaling=scaling)
    newsize = size(matread(psfpath)[psfname])[1:(end - 1)]
    return run_forwardmodel(
        model, sourcedir, destinationdir; amount=amount, newsize=newsize
    )
end

function run_forwardmodel(
    sourcedir,
    destinationdir,
    psfpath,
    psfname,
    shiftname;
    amount=-1,
    ref_image_index=-1,
    rank=4,
    scaling=nothing
)
    model = generate_model(
        psfpath, psfname, shiftname, rank; ref_image_index=ref_image_index, scaling=scaling
    )
    newsize = size(matread(psfpath)[psfname])[1:(end - 1)]
    return run_forwardmodel(
        model, sourcedir, destinationdir; amount=amount, newsize=newsize
    )
end

#= run_forwardmodel("../../../training_data/Data/Ground_truth_downsampled/", "../../../training_data/Data/JuliaForwardModel/",
 "../../../training_data/comaPSF.mat", "psfs") 
   run_forwardmodel("../../../training_data/3D/Simulated_Miniscope_3D_training_data/", "../../../training_data/3D/JuliaForwardModel/", 
   "../../../training_data/3D/comaPSF3D_square.mat", "psfs"; rank=8, amount=4000)=#
