using MAT, HDF5
export readPSFs, pad2D, unpad2D, pad3D, unpad3D
"""    
    readPSFs(path::String, key::String)

Read the PSFs stored in file `path` accessible as field `key`. 
Supports MAT and HDF5 file format. 

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

"""    pad2D(x)

Pad `x` along the first two dimensions with `0` to twice its size.
"""
function pad2D(x)
    return select_region(x, new_size=2 .* size(x)[1:2], pad_value=zero(eltype(x)))
end

"""    pad3D(x)

Pad `x` along the first three dimensions with `0` to twice its size.
"""
function pad3D(x)
    return select_region(x, new_size=2 .* size(x)[1:3], pad_value=zero(eltype(x)))
end

function lowerIndex(N)
    return Bool(N % 2) ? (N+3)÷2 : (N+2)÷2
end

function upperIndex(N)
    return Bool(N % 2) ?  3*N÷2 +1 : 3*N÷2
end

function unpad2D(x, Ny, Nx)
    ccL = lowerIndex(Nx)
    ccU = upperIndex(Nx)
    rcL = lowerIndex(Ny)
    rcU = upperIndex(Ny)
    return x[rcL:rcU, ccL:ccU]
end

function unpad3D(x, Ny, Nx, Nz)
    ccL = lowerIndex(Nx)
    ccU = upperIndex(Nx)
    rcL = lowerIndex(Ny)
    rcU = upperIndex(Ny)
    dcL = lowerIndex(Nz)
    dcU = upperIndex(Nz)
    return x[rcL:rcU, ccL:ccU, dcL:dcU]
end
