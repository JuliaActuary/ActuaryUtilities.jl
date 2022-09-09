@testset "Derivatives" begin

    @testset "Euro Options" begin
        # tested against https://option-price.com/index.php
        params = (S = 1.0, K = 1.0, τ = 1, r = 0.05, σ = 0.25, q = 0.0)

        @test eurocall(; params...) ≈ 0.12336 atol = 1e-5
        @test europut(; params...) ≈ 0.07459 atol = 1e-5

        params = (S = 1.0, K = 1.0, τ = 1, r = 0.05, σ = 0.25, q = 0.03)
        @test eurocall(; params...) ≈ 0.105493 atol = 1e-5
        @test europut(; params...) ≈ 0.086277 atol = 1e-5

        params = (S = 1.0, K = 0.5, τ = 1, r = 0.05, σ = 0.25, q = 0.03)
        @test eurocall(; params...) ≈ 0.49494 atol = 1e-5
        @test europut(; params...) ≈ 0.00011 atol = 1e-5

        params = (S = 1.0, K = 0.5, τ = 1, r = 0.05, σ = 0.25, q = 0.03)
        @test eurocall(; params...) ≈ 0.49494 atol = 1e-5
        @test europut(; params...) ≈ 0.00011 atol = 1e-5

        params = (S = 1.0, K = 0.5, τ = 0, r = 0.05, σ = 0.25, q = 0.03)
        @test eurocall(; params...) ≈ 0.5 atol = 1e-5
        @test europut(; params...) ≈ 0.0 atol = 1e-5

        params = (S = 1.0, K = 1.5, τ = 0, r = 0.05, σ = 0.25, q = 0.03)
        @test eurocall(; params...) ≈ 0.0 atol = 1e-5
        @test europut(; params...) ≈ 0.5 atol = 1e-5
        
    end
end

