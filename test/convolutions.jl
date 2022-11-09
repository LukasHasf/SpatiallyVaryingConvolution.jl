@testset "Test 2D model" begin
    function convolve(x, y)
        Ny, Nx = size(x)
        mask = findall(SpatiallyVaryingConvolution.pad_nd(ones(Float64, Ny, Nx), 2) .== 1)
        x = SpatiallyVaryingConvolution.pad_nd(x, 2)
        y = SpatiallyVaryingConvolution.pad_nd(y, 2)
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
        model = SpatiallyVaryingConvolution.generate_model(psfs, rank)
        input_image = rand(Float64, Ny, Nx)
        input_image ./= maximum(input_image)
        sim_image = model(input_image)
        sim_image ./= maximum(sim_image)
        @test size(input_image) == size(sim_image)
        @test sim_image ≈ input_image
    end

    @testset "Convolution with custom shifted delta peak" begin
        Ny = 501
        Nx = 500
        nrPSFs = 2
        rank = nrPSFs - 1
        psfs = zeros(Float64, Ny, Nx, nrPSFs)
        shift = [10 0; 10 0]
        psfs[Ny ÷ 2 + 1 +  shift[1, 1], Nx ÷ 2 + 1 + shift[2, 1], 1] = 1
        psfs[Ny ÷ 2 + 1 +  shift[1, 2], Nx ÷ 2 + 1 + shift[2, 2], 2] = 1
        model = SpatiallyVaryingConvolution.generate_model(psfs, rank; shifts=shift)
        input_image = rand(Float64, Ny, Nx)
        input_image ./= maximum(input_image)
        sim_image = model(input_image)
        sim_image ./= maximum(input_image)
        buf = similar(sim_image)
        _linshift!(buf, input_image, shift[:, 1]; filler=zero(Float64))
        @test sim_image ≈ buf
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
        model = SpatiallyVaryingConvolution.generate_model(psfs, rank)
        input_image = rand(Float64, Ny, Nx)
        input_image ./= maximum(input_image)
        sim_image = model(input_image)
        sim_image ./= maximum(sim_image)
        @test size(input_image) == size(sim_image)
        @test sim_image ≈ input_image
    end

    @testset "Scaled convolution" begin
        nrPSFs = 9
        rank = nrPSFs - 1
        Ny = 501
        Nx = 500
        psfs = zeros(Ny, Nx, nrPSFs)
        psfs[Ny÷2+1, Nx÷2+1,:] .= 1
        scaling = 2
        model = SpatiallyVaryingConvolution.generate_model(psfs, rank; scaling=scaling)
        input_image = rand(Float64, Ny, Nx)
        input_image ./= maximum(input_image)
        sim_image = model(input_image)
        psfs_scaled = scale_fourier(psfs[:, :, 1], scaling)
        ii_pad = pad_nd(input_image, ndims(input_image); fac=scaling)
        con = fftshift(irfft(rfft(ii_pad) .* rfft(psfs_scaled), size(psfs_scaled, 1)))
        con = scale_fourier(con, 1/scaling)
        @test sum(abs, con .- sim_image) / (Ny * Nx) < 0.01
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
        model = SpatiallyVaryingConvolution.generate_model(psfs, rank)
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
            SpatiallyVaryingConvolution.pad_nd(ones(Float64, Ny, Nx, Nz), 3) .== 1
        )
        x = SpatiallyVaryingConvolution.pad_nd(x, 3)
        y = SpatiallyVaryingConvolution.pad_nd(y, 3)
        output = ifftshift(real.(ifft(fft(x) .* fft(y))))[mask]
        output = reshape(output, Ny, Nx, Nz)
        return output ./ maximum(output)
    end
    @testset "Convolution with delta peak is identity" begin
        Ny = 51
        Nx = 50
        Nz = 50
        nrPSFs = 5
        rank = nrPSFs - 1
        psfs = zeros(Float64, Ny, Nx, Nz, nrPSFs)
        psfs[Ny ÷ 2 + 1, Nx ÷ 2 + 1, Nz ÷ 2 + 1, :] .= 1
        model = SpatiallyVaryingConvolution.generate_model(psfs, rank)
        input_image = rand(Float64, Ny, Nx, Nz)
        input_image ./= maximum(input_image)
        sim_image = model(input_image)
        sim_image ./= maximum(sim_image)
        @test size(input_image) == size(sim_image)
        @test sim_image ≈ input_image
    end

    @testset "Convolution with reduction" begin
        Ny = 51
        Nx = 50
        Nz = 50
        nrPSFs = 5
        rank = nrPSFs - 1
        psfs = zeros(Float64, Ny, Nx, Nz, nrPSFs)
        psfs[Ny ÷ 2 + 1, Nx ÷ 2 + 1, Nz ÷ 2 + 1, :] .= 1
        model = SpatiallyVaryingConvolution.generate_model(psfs, rank; reduce=true)
        input_image = rand(Float64, Ny, Nx, Nz)
        input_image ./= maximum(input_image)
        sim_image = model(input_image)
        sim_image ./= maximum(sim_image)
        @test size(input_image)[1:2] == size(sim_image)
        @test sim_image ≈ sum(input_image; dims=3) ./ maximum(sum(input_image; dims=3))
    end

    @testset "Convolution with x-y-shifted delta peaks is identity" begin
        nrPSFs = 5
        rank = nrPSFs - 1
        Ny = 51
        Nx = 50
        Nz = 50
        shiftrange = max(Ny, Nx, Nz) ÷ 3
        shifts = rand((-shiftrange):shiftrange, (3, nrPSFs))
        shifts[:, size(shifts)[2] ÷ 2 + 1] .= 0
        # Only shift in x-y-plane. z shifts are tested later
        shifts[3, :] .= 0
        psfs = zeros(Float64, Ny, Nx, Nz, nrPSFs)
        center = Int.([Ny ÷ 2 + 1, Nx ÷ 2 + 1, Nz ÷ 2 + 1])
        for shift_index in 1:nrPSFs
            psfs[(center .+ shifts[:, shift_index])..., shift_index] = one(Float64)
        end
        model = SpatiallyVaryingConvolution.generate_model(psfs, rank)
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
        Ny = 51
        Nx = 50
        Nz = 50
        shiftrange = max(Ny, Nx, Nz) ÷ 4
        shifts = rand((-shiftrange):shiftrange, (3, nrPSFs))
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
        model = SpatiallyVaryingConvolution.generate_model(psfs, rank)
        input_image = zeros(Float64, Ny, Nx, Nz)
        input_image[Ny ÷ 2 + 1, Nx ÷ 2 + 1, Nz ÷ 2 + 1] = 1.0
        sim_image = model(input_image)
        sim_image ./= maximum(sim_image)
        convolved_image = convolve(input_image, psfs[:, :, :, nrPSFs ÷ 2 + 1])
        @test size(input_image) == size(sim_image)
        @test sim_image ≈ convolved_image
    end
end
