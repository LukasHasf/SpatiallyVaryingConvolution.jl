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
end
