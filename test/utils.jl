@testset "Test utils functions" begin
    @testset "Test padding/unpadding" begin
        Ny, Nx = 100, 101
        x = rand(Float64, Ny, Nx)
        @test unpad(padND(x, 2), Ny, Nx) == x

        Ny, Nx = 101, 100
        x = rand(Float64, Ny, Nx)
        @test unpad(padND(x, 2), Ny, Nx) == x

        @test size(padND(x, 2)) == 2 .* size(x)

        Nz = 10
        x = rand(Float64, Ny, Nx, Nz)
        @test size(padND(x, 2)) == ((2 .* size(x)[1:2])..., Nz)

        Ny, Nx, Nz = 100, 101, 40
        x = rand(Float64, Ny, Nx, Nz)
        @test unpad(padND(x, 3), Ny, Nx, Nz) == x

        Ny, Nx, Nz = 101, 100, 39
        x = rand(Float64, Ny, Nx, Nz)
        @test unpad(padND(x, 3), Ny, Nx, Nz) == x

        @test size(padND(x, 3)) == 2 .* size(x)

        nrPSFs = 10
        x = rand(Float64, Ny, Nx, Nz, nrPSFs)
        @test size(padND(x, 3)) == ((2 .* size(x)[1:3])..., nrPSFs)
    end

    @testset "Test loading files" begin
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

        random_1 = rand(Float32, 10, 10)
        random_2 = rand(Float32, 10, 10)
        mat_filename = "mat_test.mat"
        mat_key = "abc"
        h5_key = "def"
        h5_filename = "h5_test.h5"
        savePSFs(random_1, mat_filename, mat_key)
        savePSFs(random_2, h5_filename, h5_key)
        @test random_1 == readPSFs(mat_filename, mat_key)
        @test random_2 == readPSFs(h5_filename, h5_key)
        @test isnothing(readPSFs(mat_filename, h5_key))
        rm(mat_filename)
        rm(h5_filename)
    end
end
