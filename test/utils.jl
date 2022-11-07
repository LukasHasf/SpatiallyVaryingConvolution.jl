function savePSFs(psfs, filename, key)
    if occursin(".mat", filename)
        file = matopen(filename, "w")
        write(file, key, psfs)
        close(file)
    elseif occursin(".h5", filename)
        h5open(filename, "w") do fid
            fid[key] = psfs
        end
    end
end

@testset "Test utils functions" begin
    @testset "Test padding/unpadding" begin
        Ny, Nx = 100, 101
        x = rand(Float64, Ny, Nx)
        @test unpad(pad_nd(x, 2), Ny, Nx) == x

        Ny, Nx = 101, 100
        x = rand(Float64, Ny, Nx)
        @test unpad(pad_nd(x, 2), Ny, Nx) == x

        @test size(pad_nd(x, 2)) == 2 .* size(x)

        Nz = 10
        x = rand(Float64, Ny, Nx, Nz)
        @test size(pad_nd(x, 2)) == ((2 .* size(x)[1:2])..., Nz)

        Ny, Nx, Nz = 100, 101, 40
        x = rand(Float64, Ny, Nx, Nz)
        @test unpad(pad_nd(x, 3), Ny, Nx, Nz) == x

        Ny, Nx, Nz = 101, 100, 39
        x = rand(Float64, Ny, Nx, Nz)
        @test unpad(pad_nd(x, 3), Ny, Nx, Nz) == x

        @test size(pad_nd(x, 3)) == 2 .* size(x)

        nrPSFs = 10
        x = rand(Float64, Ny, Nx, Nz, nrPSFs)
        @test size(pad_nd(x, 3)) == ((2 .* size(x)[1:3])..., nrPSFs)
    end

    @testset "Test loading files" begin
        random_1 = rand(Float32, 10, 10)
        random_2 = rand(Float32, 10, 10)
        mat_filename = "mat_test.mat"
        mat_key = "abc"
        h5_key = "def"
        h5_filename = "h5_test.h5"
        path = mktempdir()
        savePSFs(random_1, joinpath(path, mat_filename), mat_key)
        savePSFs(random_2, joinpath(path, h5_filename), h5_key)
        @test random_1 == read_psfs(joinpath(path, mat_filename), mat_key)
        @test random_2 == read_psfs(joinpath(path, h5_filename), h5_key)
        @test isnothing(read_psfs(joinpath(path, mat_filename), h5_key))
    end

    @testset "Test construction wrapper" begin
        psfs = rand(Float32, 200, 201, 9)
        filename = "psfs.mat"
        psfs_key = "psfs"
        rank = 8
        path = mktempdir()
        filepath = joinpath(path, filename)
        savePSFs(psfs, filepath, psfs_key)
        model1 = generate_model(psfs, rank)
        model2 = generate_model(filepath, psfs_key, rank)
        test_img = rand(Float32, 200, 201)
        img1 = model1(test_img)
        img2 = model2(test_img)
        @test img1 ≈ img2
    end

    @testset "_linshift!" begin
        A = [1, 2, 3, 4]
        B = similar(A)
        _linshift!(B, A, [0])
        @test B == A
        _linshift!(B, A, [1])
        @test B == [0, 1, 2, 3]
        _linshift!(B, A, [-1])
        @test B == [2, 3, 4, 0]

        A = [1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16]
        B = similar(A)
        _linshift!(B, A, [0, 0])
        @test B == A
        _linshift!(B, A, [1, 1])
        @test B == [0 0 0 0; 0 1 2 3; 0 5 6 7; 0 9 10 11]
        _linshift!(B, A, [-1, -1])
        @test B == [6 7 8 0; 10 11 12 0; 14 15 16 0; 0 0 0 0]
    end

    @testset "_shift_array" begin
        # Optimal 2D case
        A = rand(16, 16, 9)
        shifts = rand(0:16, 2, 9)
        good_indices = 1:9
        A_shifted = SpatiallyVaryingConvolution._shift_array(A, shifts, good_indices)
        @test size(A_shifted) == size(A)
        for (counter, ind) in enumerate(good_indices)
            @test A_shifted[:, :, counter] == circshift(A[:, :, ind], shifts[:, ind])
        end
        # 2D case with bad pictures
        good_indices = [1, 2, 3, 6, 7, 8, 9]
        A_shifted = SpatiallyVaryingConvolution._shift_array(A, shifts, good_indices)
        @test size(A_shifted) == (16, 16, 7)
        for (counter, ind) in enumerate(good_indices)
            @test A_shifted[:, :, counter] == circshift(A[:, :, ind], shifts[:, ind])
        end

        # Optimal 3D case, good measurements, only x-y shifts
        A = rand(16, 16, 16, 9)
        shifts = rand(0:16, 3, 9)
        shifts[3, :] .= zero(Int)
        good_indices = 1:9
        A_shifted = SpatiallyVaryingConvolution._shift_array(A, shifts, good_indices)
        @test size(A_shifted) == size(A)
        for (counter, ind) in enumerate(good_indices)
            @test A_shifted[:, :, :, counter] == circshift(A[:, :, :, ind], shifts[:, ind])
        end

        # 3D case with bad measurements and only x-y shifts
        good_indices = [1, 2, 3, 6, 7, 8, 9]
        A_shifted = SpatiallyVaryingConvolution._shift_array(A, shifts, good_indices)
        @test size(A_shifted) == (16, 16, 16, 7)
        for (counter, ind) in enumerate(good_indices)
            @test A_shifted[:, :, :, counter] == circshift(A[:, :, :, ind], shifts[:, ind])
        end

        # 3D case with z shifts
        shifts = rand(0:16, 3, 9)
        good_indices = 1:9
        A_shifted = SpatiallyVaryingConvolution._shift_array(A, shifts, good_indices)
        B = similar(A[:, :, :, 1])
        @test size(A_shifted) == size(A)
        for (counter, ind) in enumerate(good_indices)
            _linshift!(B, A[:, :, :, ind], shifts[:, ind])
            @test A_shifted[:, :, :, counter] == B
        end

        # 3D case with z shifts and bad measurements
        good_indices = [1, 2, 3, 6, 7, 8, 9]
        A_shifted = SpatiallyVaryingConvolution._shift_array(A, shifts, good_indices)
        B = similar(A[:, :, :, 1])
        @test size(A_shifted) == (16, 16, 16, 7)
        for (counter, ind) in enumerate(good_indices)
            _linshift!(B, A[:, :, :, ind], shifts[:, ind])
            @test A_shifted[:, :, :, counter] == B
        end
    end

    @testset "normalize_weights" begin
        Ny = 51
        Nx = 50
        nr_comps = 5
        weights = rand(Ny, Nx, nr_comps)
        comps = rand(Ny, Nx, nr_comps)
        weights_norm = normalize_weights(weights, comps)
        is_normalized = Matrix{Bool}(undef, Ny, Nx)
        for x in 1:Nx
            for y in 1:Ny
                psf = zeros(Ny, Nx)
                for c in 1:nr_comps
                    psf .+= weights_norm[y, x, c] .* comps[:, :, c]
                end
                is_normalized[y, x] = sum(psf) ≈ one(eltype(psf))
            end
        end
        @test all(is_normalized)
    end
end
