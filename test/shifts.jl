@testset "Test that shift registration works as intended" begin
    function calc_shifts(shifts)
        Ny = 501
        Nx = 501
        nrPSFs = size(shifts)[2]
        psfs = zeros(Float64, Ny, Nx, nrPSFs)
        center = Int.([Ny÷2+1, Nx÷2+1])
        for shift_index in 1:nrPSFs
            psfs[(center .+ shifts[:, shift_index])...,shift_index] = one(Float64)
        end
        psfs_reg, shifts =
        SpatiallyVaryingConvolution.registerPSFs(psfs[:, :, :], psfs[:, :, nrPSFs÷2+1])
        return shifts
    end
    @testset "Test shift is zero for exact same PSF" begin
        in_shifts = zeros(Int32, (2, 9))
        @test in_shifts ≈ -calc_shifts(in_shifts)
    end

    @testset "Test random shifts are registered correctly" begin
        in_shifts = rand(-200:200, (2, 17))
        in_shifts[:, size(in_shifts)[2]÷2+1].= 0
        @test in_shifts ≈ -calc_shifts(in_shifts)
    end
end