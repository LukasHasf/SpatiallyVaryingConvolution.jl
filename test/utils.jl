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
        @info "The following warnings are part of testing"
        @test isnothing(read_psfs(joinpath(path, mat_filename), h5_key))
        @test_logs (:warn, "Key $(h5_key) not found in $(joinpath(path, mat_filename))!") read_psfs(
            joinpath(path, mat_filename), h5_key
        )
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
        # Test if loading with string tries to get the positions from the PSF file. If it failf, it displays a warning
        positions_key = "positions"
        @test_logs (:warn, "Key $positions_key not found in $(filepath)!") generate_model(
            filepath, psfs_key, rank; positions=positions_key
        )
        # Now try the model with user supplied positions
        # There are three ways to construct a model with user supplied poitiosn, so check for equivalence of these ways
        positions = rand(1:200, 2, 9)
        path = mktempdir()
        filepath = joinpath(path, filename)
        matwrite(filepath, Dict(psfs_key => psfs, positions_key => positions))
        model1 = generate_model(psfs, rank; positions=positions)
        model2 = generate_model(filepath, psfs_key, rank; positions=positions_key)
        model3 = generate_model(filepath, psfs_key, rank; positions=positions)
        test_img = rand(Float32, 200, 201)
        img1 = model1(test_img)
        img2 = model2(test_img)
        img3 = model3(test_img)
        @test img1 ≈ img2
        @test img2 ≈ img3
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
