@testset "Test 2D model" begin
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
        sim_image = model(input_image)
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
        sim_image = model(input_image)
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
        sim_image = model(input_image)
        sim_image ./= maximum(sim_image)
        @test size(input_image) == size(sim_image)
        @test sim_image ≈ convolve(input_image, psfs[:, :, nrPSFs ÷ 2 + 1])
    end
end

@testset "Test 3D model" begin
    function convolve(x, y)
        Ny, Nx, Nz = size(x)
        mask = findall(
            SpatiallyVaryingConvolution.padND(ones(Float64, Ny, Nx, Nz), 3) .== 1
        )
        x = SpatiallyVaryingConvolution.padND(x, 3)
        y = SpatiallyVaryingConvolution.padND(y, 3)
        output = ifftshift(real.(ifft(fft(x) .* fft(y))))[mask]
        output = reshape(output, Ny, Nx, Nz)
        return output ./ maximum(output)
    end
    @testset "Convolution with delta peak is identity" begin
        Ny = 101
        Nx = 100
        Nz = 100
        nrPSFs = 5
        rank = nrPSFs - 1
        psfs = zeros(Float64, Ny, Nx, Nz, nrPSFs)
        psfs[Ny ÷ 2 + 1, Nx ÷ 2 + 1, Nz ÷ 2 + 1, :] .= 1
        model = SpatiallyVaryingConvolution.generateModel(psfs, rank)
        input_image = rand(Float64, Ny, Nx, Nz)
        input_image ./= maximum(input_image)
        sim_image = model(input_image)
        sim_image ./= maximum(sim_image)
        @test size(input_image) == size(sim_image)
        @test sim_image ≈ input_image
    end

    @testset "Convolution with x-y-shifted delta peaks is identity" begin
        nrPSFs = 5
        rank = nrPSFs - 1
        Ny = 101
        Nx = 100
        Nz = 100
        shifts = rand(-50:50, (3, nrPSFs))
        shifts[:, size(shifts)[2] ÷ 2 + 1] .= 0
        # Only shift in x-y-plane. z shifts are tested later
        shifts[3, :] .= 0
        psfs = zeros(Float64, Ny, Nx, Nz, nrPSFs)
        center = Int.([Ny ÷ 2 + 1, Nx ÷ 2 + 1, Nz ÷ 2 + 1])
        for shift_index in 1:nrPSFs
            psfs[(center .+ shifts[:, shift_index])..., shift_index] = one(Float64)
        end
        model = SpatiallyVaryingConvolution.generateModel(psfs, rank)
        input_image = rand(Float64, Ny, Nx, Nz)
        input_image ./= maximum(input_image)
        sim_image = model(input_image)
        sim_image ./= maximum(sim_image)
        @test size(input_image) == size(sim_image)
        @test sim_image ≈ input_image
    end

    @testset "Convolution with non-varying PSF is normal convolution" begin
        nrPSFs = 5
        rank = nrPSFs - 1
        Ny = 101
        Nx = 100
        Nz = 100
        shifts = rand(-30:30, (3, nrPSFs))
        shifts[:, size(shifts)[2] ÷ 2 + 1] .= 0
        psfs = zeros(Float64, Ny, Nx, Nz, nrPSFs)
        center = Int.([Ny ÷ 2 + 1, Nx ÷ 2 + 1, Nz ÷ 2 + 1])
        for shift_index in 1:nrPSFs
            psfs[
                center[1] + shifts[1, shift_index],
                center[2] + shifts[2, shift_index],
                center[3] + shifts[3, shift_index],
                shift_index,
            ] = one(Float64) / 4
            psfs[
                center[1] + shifts[1, shift_index] + 1,
                center[2] + shifts[2, shift_index],
                center[3] + shifts[3, shift_index],
                shift_index,
            ] = one(Float64) / 4
            psfs[
                center[1] + shifts[1, shift_index] - 1,
                center[2] + shifts[2, shift_index],
                center[3] + shifts[3, shift_index],
                shift_index,
            ] = one(Float64) / 4
            psfs[
                center[1] + shifts[1, shift_index],
                center[2] + shifts[2, shift_index] + 1,
                center[3] + shifts[3, shift_index],
                shift_index,
            ] = one(Float64) / 4
        end
        model = SpatiallyVaryingConvolution.generateModel(psfs, rank)
        input_image = zeros(Float64, Ny, Nx, Nz)
        input_image[Ny ÷ 2 + 1, Nx ÷ 2 + 1, Nz ÷ 2 + 1] = 1.0
        sim_image = model(input_image)
        sim_image ./= maximum(sim_image)
        convolved_image = convolve(input_image, psfs[:, :, :, nrPSFs ÷ 2 + 1])
        @test size(input_image) == size(sim_image)
        @test sim_image ≈ convolved_image
    end
end
