# SpatiallyVaryingConvolution.jl

This package contains utilities to simulate imaging a sample with an optical device with a spatially varying point spread function (PSF).

## Installation
```julia
julia> ] add https://github.com/LukasHasf/SpatiallyVaryingConvolution.jl
```

## Quickstart guide

### Loading your PSFs

This package provides an utility function `readPSFs(path, key)` that loads the PSFs from a MATLAB or HDF5 file at `path`. Both file structures support storing objects accessible by a `key`, which should be passed to the function as well. 
The following process expects the PSFs to have shape `(Ny, Nx, nrPSFs)` (2D) or `(Ny, Nx, nrPSFs, Nz)` (3D), where `Nx, Ny` is the pixel size of the microscope image, `nrPSFs` is the amount of PSFs recorded at each depth level and `Nz` is the amount of depth levels.

If you saved the PSFs already in the right shape, you can go on and delegate the loading to this package. The forward model is calculated by using `generateModel(psfs_path, key, rank, ref_image_index)`, where `rank` is the order of the truncated SVD and `ref_image_index` is the index of the PSF in the center of the FOV:

```julia
julia> forwardModel = generateModel("myPSF_file.h5", "myPSFs", 5, 4)
```

If you need to reshape your data first, you can load the data yourself or with `readPSFs`, process the data and pass the reshaped PSFs to `generateModel`:

```julia
julia> psfs = readPSFs("myPSF_file.h5", "myPSFs")

julia> psfs_reshaped = ...

julia> forwardModel = generateModel(psfs_reshaped, 5, 4)
```

For now, the forward model needs the input image (numerical array of size `(Ny, Nx)`) to be zero padded to twice its size manually. You can do that with the provided function `pad2D`. The returned image of the forward model is already cropped to the size of the microscope FOV.

``` julia
julia> padded_image = pad2D(image_to_convolve)

julia> convolved_image = forwardModel(image_to_convolve)
```
