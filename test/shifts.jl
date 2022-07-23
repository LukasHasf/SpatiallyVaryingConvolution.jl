@testset "Test shift registration" begin
    function calc_shifts(shifts)
        ND = size(shifts)[1]
        Ny = 51
        Nx = 51
        Nz = 50
        Ns = [Ny, Nx, Nz][1:ND]
        nrPSFs = size(shifts)[2]
        psfs = zeros(Float32, Ns..., nrPSFs)
        center = Int.(Ns .÷ 2 .+ 1)
        for shift_index in 1:nrPSFs
            psfs[(center .- shifts[:, shift_index])..., shift_index] = one(Float64)
        end
        _, shifts = SpatiallyVaryingConvolution.registerPSFs(
            psfs, selectdim(psfs, ndims(psfs), nrPSFs ÷ 2 + 1)
        )
        return shifts
    end

    @testset "Test shift is zero for exact same PSF" begin
        in_shifts = zeros(Int32, (2, 9))
        @test in_shifts ≈ calc_shifts(in_shifts)
        in_shifts = zeros(Int32, (3, 9))
        @test in_shifts ≈ calc_shifts(in_shifts)
    end

    @testset "Test random shifts are registered correctly" begin
        in_shifts = rand(-20:20, (2, 17))
        in_shifts[:, size(in_shifts)[2] ÷ 2 + 1] .= 0
        @test in_shifts ≈ calc_shifts(in_shifts)
        in_shifts = rand(-20:20, (3, 9))
        in_shifts[:, size(in_shifts)[2] ÷ 2 + 1] .= 0
        @test in_shifts ≈ calc_shifts(in_shifts)
    end

    @testset "Test registration works for more complex shift invariant PSFs" begin
        nrPSFs = 17
        in_shifts = rand(-20:20, (2, nrPSFs))
        in_shifts[:, size(in_shifts)[2] ÷ 2 + 1] .= 0
        Ny, Nx = 40, 40
        padded_psf = padND(rand(Float64, Ny, Nx), 2)
        psfs = Array{Float32,3}(undef, Ny, Nx, nrPSFs)
        for i in 1:nrPSFs
            psfs[:, :, i] .= unpad(circshift(padded_psf, -in_shifts[:, i]), Ny, Nx)
        end
        psfs_reg, shifts = SpatiallyVaryingConvolution.registerPSFs(
            psfs, psfs[:, :, nrPSFs ÷ 2 + 1]
        )
        @test in_shifts ≈ shifts

        nrPSFs = 9
        in_shifts = rand(-20:20, (3, nrPSFs))
        in_shifts[:, size(in_shifts)[2] ÷ 2 + 1] .= 0
        Ny, Nx, Nz = 40, 40, 40
        padded_psf = padND(rand(Float64, Ny, Nx, Nz), 3)
        psfs = Array{Float32,4}(undef, Ny, Nx, Nz, nrPSFs)
        for i in 1:nrPSFs
            psfs[:, :, :, i] .= unpad(circshift(padded_psf, -in_shifts[:, i]), Ny, Nx, Nz)
        end
        psfs_reg, shifts = SpatiallyVaryingConvolution.registerPSFs(
            psfs, psfs[:, :, :, nrPSFs ÷ 2 + 1]
        )
        @test in_shifts ≈ shifts
    end
end
