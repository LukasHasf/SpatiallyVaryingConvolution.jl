@testset "Test utils functions" begin
    @testset "Test padding/unpadding" begin
        Ny, Nx = 100, 101
        x = rand(Float64, Ny, Nx)
        @test unpad2D(pad2D(x), Ny, Nx) == x

        Ny, Nx = 101, 100
        x = rand(Float64, Ny, Nx)
        @test unpad2D(pad2D(x), Ny, Nx) == x

        @test size(pad2D(x)) == 2 .* size(x)

        Nz = 10
        x = rand(Float64, Ny, Nx, Nz)
        @test size(pad2D(x)) == ((2 .* size(x)[1:2])..., Nz)
    end
end