using SpatiallyVaryingConvolution
using Images
using MAT
using ProgressMeter

function _map_to_zero_one!(x, min_x, max_x)
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
        img_path = joinpath(sourcedir, sourcefile)
        destination_path = joinpath(destinationdir, sourcefile)
        img = reverse(load(img_path); dims=1)
        img = imresize(img, newsize)
        img = Float64.(Gray.(img))
        sim = model(img)
        mi, ma = extrema(sim)
        _map_to_zero_one!(sim, mi, ma)
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

function run_forwardmodel(
    sourcedir, destinationdir, psfpath, psfname; amount=-1, ref_image_index=-1, rank=4
)
    model = generateModel(psfpath, psfname, rank, ref_image_index)
    sourcefiles = amount == -1 ? readdir(sourcedir) : readdir(sourcedir)[1:amount]
    newsize = size(matread(psfpath)[psfname])[1:(end - 1)]
    isdir(destinationdir) || mkpath(destinationdir)
    if length(newsize) == 2
        iterate_over_images(sourcedir, destinationdir, sourcefiles, model, newsize)
    elseif length(newsize) == 3
        iterate_over_volumes(sourcedir, destinationdir, sourcefiles, model, newsize)
    end
end

#= run_forwardmodel("../../../training_data/Data/Ground_truth_downsampled/", "../../../training_data/Data/JuliaForwardModel/",
 "../../../training_data/comaPSF.mat", "psfs") 
   run_forwardmodel("../../../training_data/3D/Simulated_Miniscope_3D_training_data/", "../../../training_data/3D/JuliaForwardModel/", 
   "../../../training_data/3D/comaPSF3D.mat", "psfs")=#
