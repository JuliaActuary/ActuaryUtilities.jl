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

        # sample vs. distribution
        @test wasserstein(rand(Normal(0, 1), 500), Normal(0, 1)) < 0.25

        # unequal-length samples are EXACT (merged-breakpoint integration of the
        # two step quantile functions), not a grid/interpolation approximation:
        # δ₀ vs ½(δ₀+δ₂) moves mass ½ a distance 2 ⇒ W₂ = √2 (interpolated
        # quantiles would give √1.25) and W₁ = 1.
        @test wasserstein([0], [0, 2]; p=2) ≈ sqrt(2)
        @test wasserstein([0], [0, 2]; p=1) ≈ 1.0
        @test wasserstein([0], [0, 2]; p=Inf) ≈ 2.0
        @test wasserstein([0.0, 10.0], [0.0, 0.0]; p=Inf) ≈ 10.0
        @test wasserstein([0.0, 0.0], [0.0, 0.0]; p=Inf) == 0.0
        # hand-integrated: segments (0,.2]:|1-10| (.2,.4]:|2-10| (.4,.5]:|3-10|
        # (.5,.6]:|3-20| (.6,.8]:|4-20| (.8,1]:|5-20| ⇒ W₁ = 12
        @test wasserstein([1, 2, 3, 4, 5], [10, 20]) ≈ 12.0
        @test wasserstein([10, 20], [1, 2, 3, 4, 5]) ≈ 12.0   # symmetric

        @test_throws ArgumentError wasserstein([1, 2], [3, 4]; p=0.5)
        # empty samples fail loudly rather than returning NaN
        @test_throws ArgumentError wasserstein(Float64[], [1.0, 2.0])
        @test_throws ArgumentError wasserstein([1.0, 2.0], Float64[])

        # Distribution forms use adaptive quantile integration and distinguish a
        # genuinely divergent moment from tail cancellation between two laws.
        @test wasserstein(Cauchy(), Dirac(0.0); p=1) == Inf
        @test wasserstein(Cauchy(), Cauchy(3, 1); p=1) ≈ 3.0
        @test wasserstein(Normal(), Normal(3, 1); p=Inf) ≈ 3.0
        @test wasserstein(Normal(), Normal(0, 2); p=Inf) == Inf
        @test wasserstein(LogNormal(0, 1), LogNormal(0, 1); p=Inf) == 0.0
        @test wasserstein(LogNormal(0, 1), LogNormal(0, 2); p=Inf) == Inf
        @test wasserstein(Cauchy(), Cauchy(3, 1); p=Inf) ≈ 3.0
        @test wasserstein(Cauchy(), Cauchy(0, 2); p=Inf) == Inf
        @test wasserstein(Uniform(0, 1), Uniform(0, 2); p=Inf) ≈ 1.0
        @test wasserstein(Uniform(-2, 1), Uniform(0, 2); p=Inf) ≈ 2.0
        @test wasserstein(randn(MersenneTwister(11), 100), Normal(); p=Inf) == Inf
        # The empirical law is bounded, so its W₁ distance from a true Cauchy
        # law is infinite even when the observations themselves came from Cauchy.
        @test wasserstein(rand(MersenneTwister(12), Cauchy(), 5_000), Cauchy()) == Inf
        # Scenario counts used in practice must not exhaust the default budget on
        # the initial quadrature pass merely because every empirical rank is a break.
        @test isfinite(wasserstein(randn(MersenneTwister(13), 10_000), Normal()))
        @test_throws ErrorException wasserstein(Uniform(0, 1), Uniform(3, 4); p=Inf, maxevals=64)
        mix = MixtureModel([Normal(0, 1), Normal(5, 1)], [0.99, 0.01])
        @test_throws ErrorException wasserstein(mix, Normal(); p=Inf)
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

        # sample → sample is a true pushforward: equally sized tie-free samples
        # map order statistic to order statistic (inverse ECDF, not interpolation)
        @test pushforward([1, 2], transportmap([1, 2], [10, 20])) == [10.0, 20.0]
        @test pushforward([3, 1, 2], transportmap([3, 1, 2], [10, 30, 20])) == [30.0, 10.0, 20.0]
        # tied source values cannot be split by a deterministic map: both zeros
        # land on the same target quantile
        @test pushforward([0, 0], transportmap([0, 0], [10, 20]))[1] ==
              pushforward([0, 0], transportmap([0, 0], [10, 20]))[2]
    end

    @testset "robustvalue (Wasserstein DRO)" begin
        Random.seed!(1)
        s = rand(LogNormal(log(1000) - 0.5 * 0.6^2, 0.6), 100_000)

        # For CTE(α) with tail=α the sharp bound is attained exactly (closed form).
        for (α, p, r) in ((0.95, 2, 250.0), (0.995, 2, 250.0), (0.90, 1, 100.0))
            bound = CTE(α)(s) + r * (1 - α)^(-1 / p)
            @test robustvalue(CTE(α), s; radius=r, p=p) ≈ bound rtol = 1e-8
        end

        # robust value is monotone in the radius and never below the base measure
        @test robustvalue(CTE(0.95), s; radius=0) ≈ CTE(0.95)(s)
        @test robustvalue(CTE(0.95), s; radius=500) > robustvalue(CTE(0.95), s; radius=100)

        # deeper tail is more fragile: same radius, larger loading
        loading(α) = robustvalue(CTE(α), s; radius=250) - CTE(α)(s)
        @test loading(0.995) > loading(0.95)

        # Small-sample exactness, including where the tail boundary cuts through an
        # atom (n=10, α=0.25 puts mass 0.8 > 1-α at ranks k:n): a whole-atom shift
        # attaining this bound would cost W₁ = r·(0.8/0.75) > r, outside the ball.
        # The closed form is both exact and feasible (its maximizer splits the atom).
        let sm = collect(1.0:10.0)
            for (α, p, r) in ((0.25, 1, 3.0), (0.7, 2, 2.0), (0.0, 1, 1.5))
                @test robustvalue(CTE(α), sm; radius=r, p=p) ≈
                      CTE(α)(sm) + r * (1 - α)^(-1 / p) rtol = 1e-12
            end
        end

        # Ties/atoms: the CTE bound stays exact even when many observations equal
        # the threshold value.
        let st = [fill(0.0, 80); fill(50.0, 15); fill(100.0, 5)]   # heavy tie at 50
            @test robustvalue(CTE(0.9), st; radius=4.0, p=2) ≈
                  CTE(0.9)(st) + 4.0 * (1 - 0.9)^(-1 / 2) rtol = 1e-12
        end

        # Generic (non-closed-form) path: the adverse scenario shifts the tail
        # ranks k:n by radius·(m/n)^(-1/p) — spending EXACTLY the budget, never
        # more. Forced here by stressing CTE at a tail level ≠ its α: n=10,
        # tail=0.25 ⇒ k=3, m=8, Δ = 3·(8/10)^(-1) = 3.75.
        let sm = collect(1.0:10.0)
            sc = [1.0, 2.0, ((3.0:10.0) .+ 3.0 * (10 / 8))...]
            @test robustvalue(CTE(0.5), sm; radius=3.0, p=1, tail=0.25) ≈ CTE(0.5)(sc)
            @test wasserstein(sc, sm; p=1) ≈ 3.0    # scenario is ON the ball boundary
        end

        # integer samples work: the scenario promotes to float before shifting
        # (previously threw InexactError on the generic path)
        @test robustvalue(VaR(0.5), [1, 2, 3]; radius=1.0) ≥ VaR(0.5)([1, 2, 3])
        @test robustvalue(CTE(0.5), [1, 2]; radius=1.0) ≈ CTE(0.5)([1, 2]) + 1.0 * 0.5^(-1 / 2)

        # original sample is not mutated (generic path materializes a copy)
        let s0 = rand(LogNormal(0, 1), 1_000), s1 = copy(s0)
            robustvalue(VaR(0.95), s0; radius=10)
            @test s0 == s1
        end

        # VaR's adverse scenario shifts the rank VaR itself selects. At an exact
        # atom boundary (n=10, tail=0.5) the strict rule would shift only ranks
        # 6:10 and leave VaR at rank 5 unmoved — zero loading.
        let sm = collect(1.0:10.0)
            @test VaR(0.5)(sm) == 5.0                 # rank 5 under the ≥ rule
            rv = robustvalue(VaR(0.5), sm; radius=1.0, p=2)
            @test rv > VaR(0.5)(sm)                   # loading is strictly positive
            # selected rank k=5, so m=6 and Δ = 1·(6/10)^(-1/2)
            @test rv ≈ 5.0 + 1.0 * (6 / 10)^(-1 / 2)
            # the constructed scenario spends exactly the W_p budget
            sc = [sm[1:4]; sm[5:10] .+ 1.0 * (6 / 10)^(-1 / 2)]
            @test wasserstein(sc, sm; p=2) ≈ 1.0 rtol = 1e-12
            # non-VaR measures keep the strict boundary: k=6, m=5
            scw = [sm[1:5]; sm[6:10] .+ 1.0 * (5 / 10)^(-1 / 2)]
            @test robustvalue(WangTransform(0.5), sm; radius=1.0, p=2, tail=0.5) ≈
                  WangTransform(0.5)(scw)
        end

        # ties at the boundary: the shifted set still contains VaR's selected atom
        let st = [1.0, 1.0, 1.0, 1.0, 2.0]            # VaR(0.8) selects rank 4 (value 1.0)
            rv = robustvalue(VaR(0.8), st; radius=0.5, p=1)
            @test rv > VaR(0.8)(st)
            @test rv ≈ 1.0 + 0.5 * (2 / 5)^(-1)       # k=4, m=2, Δ = 0.5·(5/2)
        end

        # composes with any risk measure; a measure without a natural tail needs `tail`.
        # (WangTransform has no fast array method, so use a small sample here.)
        @test robustvalue(VaR(0.95), s; radius=100) ≥ VaR(0.95)(s)
        sw = rand(LogNormal(log(1000) - 0.18, 0.6), 500)
        @test robustvalue(WangTransform(0.95), sw; radius=100, tail=0.95) > WangTransform(0.95)(sw)
        @test_throws ArgumentError robustvalue(WangTransform(0.95), s; radius=100)
        @test_throws ArgumentError robustvalue(CTE(0.95), s; radius=-1)
        @test_throws ArgumentError robustvalue(CTE(0.95), Float64[]; radius=100)
    end

end
