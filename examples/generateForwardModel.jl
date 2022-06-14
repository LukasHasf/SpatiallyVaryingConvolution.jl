### A Pluto.jl notebook ###
# v0.19.8

using Markdown
using InteractiveUtils

# ╔═╡ 95913df4-d05b-11ec-0ce6-d52c5d4770ec
begin
    using Pkg: Pkg
    Pkg.activate(".")
    using Revise
    using SpatiallyVaryingConvolution
    using DeconvOptim
    using Colors
    using FFTW
    using TestImages
    using ImageShow
end

# ╔═╡ 3a5bd644-9f29-40ae-a421-1947f1425dad
Pkg.add("ImageShow")

# ╔═╡ 37124966-7152-429c-bb80-3bf0fd49c4cc
md"""
## Generate some fake PSF data

For example, 9 PSFs, where the one in the center provides the highest resolution.
"""

# ╔═╡ 379ab12e-ae61-4b09-874b-7b973c4baf67
Ny, Nx, nrPSFs = 512, 512, 9;

# ╔═╡ 4e1b91e7-dca1-4da7-82b4-d0767c8bcfd8
psfs = zeros(Float64, Ny, Nx, nrPSFs);

# ╔═╡ 174500d1-1108-46fb-8f97-f4da26a37c1c
psf_radii = [20, 20, 20, 20, 400, 20, 20, 20, 20]

# ╔═╡ 78f15bce-d47e-4555-9879-ad6a384057e8
shifts = [
    (Ny ÷ 4, Nx ÷ 4),
    (2 * Ny ÷ 4, Nx ÷ 4),
    (3 * Ny ÷ 4, Nx ÷ 4),
    (Ny ÷ 4, 2 * Nx ÷ 4),
    (2 * Ny ÷ 4, 2 * Nx ÷ 4),
    (3 * Ny ÷ 4, 2 * Nx ÷ 4),
    (Ny ÷ 4, 3 * Nx ÷ 4),
    (2 * Ny ÷ 4, 3 * Nx ÷ 4),
    (3 * Ny ÷ 4, 3 * Nx ÷ 4),
];

# ╔═╡ fc9315e7-7887-4134-897e-f9d95479d44f
for i in 1:nrPSFs
    psfs[:, :, i] .= circshift(
        DeconvOptim.generate_psf((Ny, Nx), psf_radii[i]), -1 .* shifts[i]
    )
end

# ╔═╡ 7f9350b7-cde3-4b2b-805d-51e2bf400eb0
md"""
A maximum projection of the used PSFs
"""

# ╔═╡ ac515b97-479b-430f-99bf-d040268907df
Gray.(maximum(psfs; dims=3)[:, :, 1] .* 10)

# ╔═╡ d67df24c-4ccb-404c-84aa-109441815766
md"""
### Load an image and pad it
"""

# ╔═╡ 6b0c01fc-9970-4b55-918b-86c3486af3c2
image_to_convolve = testimage("resolution_test_512")

# ╔═╡ 5d79fa8b-72ad-4140-9aa2-3502e6a06081
padded_image = padND(Float64.(image_to_convolve), 2);

# ╔═╡ 107f10dc-7122-4043-93d8-eee2f60af1a5
Gray.(padded_image)

# ╔═╡ f8ca4fc5-3c41-40e4-b471-a763270c8024
md"""
### Generate the forward model
"""

# ╔═╡ f119d560-57a8-42d1-9f2d-9718cb2f825a
forwardModel = generateModel(psfs, 4, 5)

# ╔═╡ 6036accd-9cfa-4198-bf80-22db74d26c38
md"""
### Apply the forward model
"""

# ╔═╡ 725281de-9eb4-4ed4-a40b-85f498b0a55a
convolved_image = forwardModel(padded_image);

# ╔═╡ a9146bc4-ded0-4fa1-8a61-d9ab09aec57f
Gray.(convolved_image)

# ╔═╡ 7bff7c23-9e13-4f9f-8716-0601fcd6ba6a
ImageShow.mosaicview(image_to_convolve, convolved_image; nrow=1)

# ╔═╡ Cell order:
# ╠═3a5bd644-9f29-40ae-a421-1947f1425dad
# ╠═95913df4-d05b-11ec-0ce6-d52c5d4770ec
# ╟─37124966-7152-429c-bb80-3bf0fd49c4cc
# ╠═379ab12e-ae61-4b09-874b-7b973c4baf67
# ╠═4e1b91e7-dca1-4da7-82b4-d0767c8bcfd8
# ╠═174500d1-1108-46fb-8f97-f4da26a37c1c
# ╠═78f15bce-d47e-4555-9879-ad6a384057e8
# ╠═fc9315e7-7887-4134-897e-f9d95479d44f
# ╟─7f9350b7-cde3-4b2b-805d-51e2bf400eb0
# ╠═ac515b97-479b-430f-99bf-d040268907df
# ╟─d67df24c-4ccb-404c-84aa-109441815766
# ╠═6b0c01fc-9970-4b55-918b-86c3486af3c2
# ╠═5d79fa8b-72ad-4140-9aa2-3502e6a06081
# ╠═107f10dc-7122-4043-93d8-eee2f60af1a5
# ╟─f8ca4fc5-3c41-40e4-b471-a763270c8024
# ╠═f119d560-57a8-42d1-9f2d-9718cb2f825a
# ╟─6036accd-9cfa-4198-bf80-22db74d26c38
# ╠═725281de-9eb4-4ed4-a40b-85f498b0a55a
# ╠═a9146bc4-ded0-4fa1-8a61-d9ab09aec57f
# ╠═7bff7c23-9e13-4f9f-8716-0601fcd6ba6a
