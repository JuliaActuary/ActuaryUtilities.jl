@testset "Optimal Transport" begin

    @testset "wasserstein" begin
        # equal-length samples: exact sorted matching
        @test wasserstein([1, 2, 3], [4, 5, 6]) ≈ 3.0
        @test wasserstein([3, 1, 2], [6, 4, 5]) ≈ 3.0          # order independent
        @test wasserstein([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0  # self-distance is 0
        @test wasserstein([0, 0, 0, 0], [0, 0, 2, 2]; p=1) ≈ 1.0    # ½ mass moves 2
        @test wasserstein([0, 0, 0, 0], [0, 0, 2, 2]; p=2) ≈ sqrt(2)

        # translation of a distribution ⇒ W_p equals the shift for every p
        @test wasserstein(Normal(0, 1), Normal(3, 1); p=1) ≈ 3.0 rtol = 1e-3
        @test wasserstein(Normal(0, 1), Normal(3, 1); p=2) ≈ 3.0 rtol = 1e-3
        # W_2 between Gaussians with equal mean is |σ₁ - σ₂|
        @test wasserstein(Normal(0, 1), Normal(0, 3); p=2) ≈ 2.0 rtol = 1e-3

        # sample vs. distribution, and unequal-length samples, run and are ≥ 0
        @test wasserstein(rand(Normal(0, 1), 500), Normal(0, 1)) < 0.25
        @test wasserstein([1, 2, 3, 4, 5], [10, 20]) > 0

        @test_throws ArgumentError wasserstein([1, 2], [3, 4]; p=0.5)
        # empty samples fail loudly rather than returning NaN
        @test_throws ArgumentError wasserstein(Float64[], [1.0, 2.0])
        @test_throws ArgumentError wasserstein([1.0, 2.0], Float64[])
    end

    @testset "transportmap / pushforward" begin
        T = transportmap(Normal(0, 1), Normal(3, 1))          # a rigid +3 shift
        @test T(0.0) ≈ 3.0 rtol = 1e-6
        @test T(1.0) ≈ 4.0 rtol = 1e-6

        # scale map: Normal(0,1) → Normal(0,2) doubles each quantile
        T2 = transportmap(Normal(0, 1), Normal(0, 2))
        @test T2(1.0) ≈ 2.0 rtol = 1e-6
        @test T2(-1.5) ≈ -3.0 rtol = 1e-6

        # pushing a sample through the map produces the target law
        s = rand(Normal(0, 1), 5_000)
        stressed = pushforward(s, transportmap(Normal(0, 1), Normal(3, 1)))
        @test mean(stressed) ≈ mean(s) + 3 rtol = 1e-6
        @test stressed ≈ T.(s)

        # empirical source → unbounded target must stay finite (max rank is not 1.0)
        src = rand(Normal(0, 1), 1_000)
        Te = transportmap(src, LogNormal(0, 1))
        @test all(isfinite, pushforward(src, Te))
        @test all(>(0), pushforward(src, Te))          # LogNormal support is positive
    end

    @testset "worstcase (Wasserstein DRO)" begin
        Random.seed!(1)
        s = rand(LogNormal(log(1000) - 0.5 * 0.6^2, 0.6), 100_000)

        # For CTE(α) with tail=α the tail-shove attains the sharp bound exactly.
        for (α, p, r) in ((0.95, 2, 250.0), (0.995, 2, 250.0), (0.90, 1, 100.0))
            bound = CTE(α)(s) + r * (1 - α)^(-1 / p)
            @test worstcase(CTE(α), s; radius=r, p=p) ≈ bound rtol = 1e-8
        end

        # worst case is monotone in the radius and never below the base measure
        @test worstcase(CTE(0.95), s; radius=0) ≈ CTE(0.95)(s)
        @test worstcase(CTE(0.95), s; radius=500) > worstcase(CTE(0.95), s; radius=100)

        # deeper tail is more fragile: same radius, larger loading
        loading(α) = worstcase(CTE(α), s; radius=250) - CTE(α)(s)
        @test loading(0.995) > loading(0.95)

        # Small-sample exactness: the bound must hold to machine precision, not just
        # O(1/n). Here n=10, α=0.25 is a case where a `Statistics.quantile` value +
        # `x ≥ thr` shift would exclude x_(k) and under-attain the bound by ~7%,
        # while the rank-based (order-statistic) shift matches CTE's own tail exactly.
        let sm = collect(1.0:10.0)
            for (α, p, r) in ((0.25, 1, 3.0), (0.7, 2, 2.0), (0.0, 1, 1.5))
                @test worstcase(CTE(α), sm; radius=r, p=p) ≈
                      CTE(α)(sm) + r * (1 - α)^(-1 / p) rtol = 1e-12
            end
        end

        # Ties/atoms: the tail boundary is by RANK, so exactly the worst
        # ceil((1-tail)·n) order statistics move — the CTE bound stays exact even
        # when many observations equal the threshold value.
        let st = [fill(0.0, 80); fill(50.0, 15); fill(100.0, 5)]   # heavy tie at 50
            @test worstcase(CTE(0.9), st; radius=4.0, p=2) ≈
                  CTE(0.9)(st) + 4.0 * (1 - 0.9)^(-1 / 2) rtol = 1e-12
        end

        # original sample is not mutated
        let s0 = rand(LogNormal(0, 1), 1_000), s1 = copy(s0)
            worstcase(CTE(0.95), s0; radius=10)
            @test s0 == s1
        end

        # composes with any risk measure; a measure without a natural tail needs `tail`.
        # (WangTransform has no fast array method, so use a small sample here.)
        @test worstcase(VaR(0.95), s; radius=100) ≥ VaR(0.95)(s)
        sw = rand(LogNormal(log(1000) - 0.18, 0.6), 500)
        @test worstcase(WangTransform(0.95), sw; radius=100, tail=0.95) > WangTransform(0.95)(sw)
        @test_throws ArgumentError worstcase(WangTransform(0.95), s; radius=100)
        @test_throws ArgumentError worstcase(CTE(0.95), s; radius=-1)
        @test_throws ArgumentError worstcase(CTE(0.95), Float64[]; radius=100)
    end

    @testset "driftsignificance" begin
        # a clear one-σ shift clears the permutation noise floor
        a = randn(MersenneTwister(1), 2_000)
        b = randn(MersenneTwister(2), 2_000) .+ 1
        ds = driftsignificance(a, b; nperm=400, rng=MersenneTwister(42))
        @test ds.distance ≈ wasserstein(a, b)
        @test ds.significant
        @test ds.threshold ≥ 0
        @test 0 < ds.pvalue ≤ 1
        @test ds.pvalue < 0.05

        # two same-law samples: the observed distance sits near the noise floor,
        # so it should not be flagged as material drift
        c = randn(MersenneTwister(3), 2_000)
        d = randn(MersenneTwister(4), 2_000)
        ds0 = driftsignificance(c, d; nperm=400, rng=MersenneTwister(7))
        @test !ds0.significant
    end

end
