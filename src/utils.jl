using MAT, HDF5
export readPSFs, pad2D, crop2D
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

function pad2D(x)
    return select_region(x, new_size=2 .* size(x)[1:2], pad_value=zero(eltype(x)))
end

function crop2D(x, rcL, rcU, ccL, ccU)
    return x[rcL:rcU, ccL:ccU]
end