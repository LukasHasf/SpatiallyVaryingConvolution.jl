@testset "Test behaviour of convolutions" begin
    function convolve(x, y)
        Ny, Nx = size(x)
        mask = findall(SpatiallyVaryingConvolution.padND(ones(Float64, Ny, Nx), 2) .== 1)
        x = SpatiallyVaryingConvolution.padND(x, 2)
        y = SpatiallyVaryingConvolution.padND(y, 2)
        output = ifftshift(real.(ifft(fft(x) .* fft(y))))[mask]
        output = reshape(output, Ny, Nx)
        return output ./ maximum(output)
    end
    @testset "Convolution with delta peak is identity" begin
        Ny = 501
        Nx = 500
        nrPSFs = 9
        rank = nrPSFs - 1
        psfs = zeros(Float64, Ny, Nx, nrPSFs)
        psfs[Ny ÷ 2 + 1, Nx ÷ 2 + 1, :] .= 1
        model = SpatiallyVaryingConvolution.generateModel(psfs, rank)
        input_image = rand(Float64, Ny, Nx)
        input_image ./= maximum(input_image)
        input_image_padded = SpatiallyVaryingConvolution.padND(input_image, 2)
        sim_image = model(input_image_padded)
        sim_image ./= maximum(sim_image)
        @test size(input_image) == size(sim_image)
        @test sim_image ≈ input_image
    end

    @testset "Convolution with shifted delta peaks is identity" begin
        nrPSFs = 9
        rank = nrPSFs - 1
        Ny = 501
        Nx = 500
        shifts = rand(-200:200, (2, nrPSFs))
        shifts[:, size(shifts)[2] ÷ 2 + 1] .= 0
        psfs = zeros(Float64, Ny, Nx, nrPSFs)
        center = Int.([Ny ÷ 2 + 1, Nx ÷ 2 + 1])
        for shift_index in 1:nrPSFs
            psfs[(center .+ shifts[:, shift_index])..., shift_index] = one(Float64)
        end
        model = SpatiallyVaryingConvolution.generateModel(psfs, rank)
        input_image = rand(Float64, Ny, Nx)
        input_image ./= maximum(input_image)
        input_image_padded = SpatiallyVaryingConvolution.padND(input_image, 2)
        sim_image = model(input_image_padded)
        sim_image ./= maximum(sim_image)
        @test size(input_image) == size(sim_image)
        @test sim_image ≈ input_image
    end

    @testset "Convolution with non-varying PSF is normal convolution" begin
        nrPSFs = 9
        rank = nrPSFs - 1
        Ny = 501
        Nx = 500
        shifts = rand(-200:200, (2, nrPSFs))
        shifts[:, size(shifts)[2] ÷ 2 + 1] .= 0
        psfs = zeros(Float64, Ny, Nx, nrPSFs)
        center = Int.([Ny ÷ 2 + 1, Nx ÷ 2 + 1])
        for shift_index in 1:nrPSFs
            psfs[
                center[1] + shifts[1, shift_index],
                center[2] + shifts[2, shift_index],
                shift_index,
            ] = one(Float64) / 4
            psfs[
                center[1] + shifts[1, shift_index] + 1,
                center[2] + shifts[2, shift_index],
                shift_index,
            ] = one(Float64) / 4
            psfs[
                center[1] + shifts[1, shift_index] - 1,
                center[2] + shifts[2, shift_index],
                shift_index,
            ] = one(Float64) / 4
            psfs[
                center[1] + shifts[1, shift_index],
                center[2] + shifts[2, shift_index] + 1,
                shift_index,
            ] = one(Float64) / 4
        end
        model = SpatiallyVaryingConvolution.generateModel(psfs, rank)
        input_image = zeros(Float64, Ny, Nx)
        input_image[Ny ÷ 2 + 1, Nx ÷ 2 + 1] = 1.0
        input_image_padded = SpatiallyVaryingConvolution.padND(input_image, 2)
        sim_image = model(input_image_padded)
        sim_image ./= maximum(sim_image)
        @test size(input_image) == size(sim_image)
        @test sim_image ≈ convolve(input_image, psfs[:, :, nrPSFs ÷ 2 + 1])
    end
end
