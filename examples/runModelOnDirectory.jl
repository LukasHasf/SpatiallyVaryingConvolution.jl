using SpatiallyVaryingConvolution
using Images
using MAT

function _map_to_zero_one!(x, min_x, max_x)
    x .-= min_x
    x .*= inv(max_x - min_x) 
end

function iterate_over_images(sourcedir, destinationdir, sourcefiles, model, newsize)
    @simd for sourcefile in sourcefiles
        img_path = joinpath(sourcedir, sourcefile)
        destination_path = joinpath(destinationdir, sourcefile)
        img = reverse(load(img_path), dims=1)
        img = imresize(img, newsize)
        img = Float64.(Gray.(img))
        sim = model(img)
        mi, ma = extrema(sim)
        _map_to_zero_one!(sim, mi, ma)
        save(destination_path, colorview(Gray, reverse(sim, dims=1)))
        println(img_path)
    end
end

function run_forwardmodel(sourcedir, destinationdir, psfpath, psfname; amount=-1, ref_image_index=-1, rank=4)
    model = generateModel(psfpath, psfname, rank, ref_image_index)
    sourcefiles = amount==-1 ? readdir(sourcedir) : readdir(sourcedir)[1:amount]
    Ny, Nx, _ = size(matread(psfpath)[psfname])
    newsize = (Ny, Nx)
    iterate_over_images(sourcedir, destinationdir, sourcefiles, model, newsize)
end

#= run_forwardmodel("../../../training_data/Data/Ground_truth_downsampled/", "../../../training_data/Data/JuliaForwardModel/",
 "../../../training_data/comaPSF.mat", "psfs") =#