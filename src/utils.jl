using MAT: matopen
using HDF5: h5open
export readPSFs, padND, unpad
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

"""    padND(x, n)

Pad `x` along the first `n` dimensions with `0` to twice its size.
"""
function padND(x, n)
    return select_region(x, new_size=2 .* size(x)[1:n], pad_value=zero(eltype(x)))
end

function lower_index(N)
    return Bool(N % 2) ? (N+3)÷2 : (N+2)÷2
end

function upper_index(N)
    return Bool(N % 2) ?  3*N÷2 +1 : 3*N÷2
end

function unpad(x, Ns...)
    low_inds = [lower_index(N) for N in Ns]
    upp_inds = [upper_index(N) for N in Ns]
    selection = [low_inds[i]:upp_inds[i] for i in eachindex(Ns)]
    return x[selection...]
end
