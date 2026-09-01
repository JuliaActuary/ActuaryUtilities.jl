# Test scaffolding for dispatch tests. Structs cannot be defined inside a
# @testset, so these live at the top level.

# A distribution whose own `mean` method throws. The error must propagate; the
# implementation must not silently reroute to quadrature.
struct BrokenMeanDist <: ContinuousUnivariateDistribution end
Distributions.mean(::BrokenMeanDist) = sum(nothing)

# A risk type whose cdf is NaN everywhere. Checked quadrature must throw, not
# return a silent number.
struct NaNCDFRisk end
ActuaryUtilities.RiskMeasures.cdf_func(::NaNCDFRisk) = x -> NaN

# An integer law with a long zero-mass prefix: all mass sits at 100 but the
# reported minimum is 0. The checked tail summation must scan past the gap
# instead of accepting an early all-zero truncation.
struct DelayedDirac <: DiscreteUnivariateDistribution end
Distributions.minimum(::DelayedDirac) = 0
Distributions.maximum(::DelayedDirac) = Inf
Distributions.cdf(::DelayedDirac, x::Real) = x < 100 ? 0.0 : 1.0
Distributions.ccdf(::DelayedDirac, x::Real) = x < 100 ? 1.0 : 0.0

@testset "Risk Measures" begin

    @test_throws AssertionError VaR(-0.5)
    @test_throws AssertionError VaR(1.0)
    @test_throws AssertionError VaR(1.5)
    @test_throws AssertionError CTE(-0.5)
    @test_throws AssertionError CTE(1.0)
    @test_throws AssertionError CTE(1.5)
    @test_throws AssertionError WangTransform(0.0)
    @test_throws AssertionError WangTransform(1.0)
    for bad in (0, -1, Inf, NaN)
        @test_throws ArgumentError DualPower(bad)
        @test_throws ArgumentError ProportionalHazard(bad)
    end
    @test DualPower(2).v == 2
    @test ProportionalHazard(2).y == 2
    @test DualPower{Float64}(2).v === 2.0
    @test ProportionalHazard{Float64}(2).y === 2.0

    # https://utstat.utoronto.ca/sam/coorses/act466/rmn.pdf pg 17
    @test RiskMeasures.g(WangTransform(cdf(Normal(), 1)), 1 - cdf(LogNormal(0, 1), 12)) ≈ 0.06879 atol = 1.0e-5
    @test RiskMeasures.Expectation()(LogNormal(0, 2 * 1)) ≈ mean(LogNormal(0, 2 * 1))


    @test CTE(0.9)(Uniform(-1, 0)) ≈ -0.05 atol = 1.0e-8
    @test RiskMeasures.Expectation()(Uniform(-1, 0)) ≈ -0.5 atol = 1.0e-8
    @test CTE(0.0)(Uniform(0, 1) - 0.5) ≈ 0.0 atol = 1.0e-8
    @test CTE(0.5)(Uniform(0, 1) - 0.5) ≈ 0.25 atol = 1.0e-8

    @test CTE(0.0)(Distributions.Normal(0, 1)) ≈ 0
    @test RiskMeasures.Expectation()(Distributions.Normal(3, 1)) ≈ 3

    # http://actuaries.org/events/congresses/cancun/afir_subject/afir_14_wang.pdf
    A = Distributions.DiscreteNonParametric([0.0, 1.0, 5.0], [0.6, 0.375, 0.025])
    B = Distributions.DiscreteNonParametric([0.0, 1.0, 11.0], [0.6, 0.39, 0.01])
    @test WangTransform(0.95)(A) ≈ 2.42 atol = 1.0e-2
    @test WangTransform(0.95)(B) ≈ 3.4 atol = 1.0e-2
    @test CTE(0.95)(A) ≈ 3
    @test CTE(0.95)(B) ≈ 3

    ## example 4.3
    @test WangTransform(0.9)(LogNormal(3, 2)) ≈ exp(3 + quantile(Normal(), 0.9) * 2 + 2^2 / 2) atol = 1.0e-3

    ## example 4.4
    C = Distributions.Exponential(1)
    α = 0.99
    @test CTE(α)(C) ≈ 5.61 atol = 1.0e-2
    @test VaR(α)(C) ≈ 4.61 atol = 1.0e-2
    @test WangTransform(α)(C) ≈ 5.02 atol = 1.0e-1

    ## example 4.5
    @test WangTransform(α)(Uniform()) ≈ 0.95 atol = 1.0e-2

    # Sepanski & Wang, "New Classes of Distortion Risk Measures and Their Estimation, Table 6
    # note the parameterization of Exp, Lomax (GP), and Weibull is different in Julia
    # than in the paper
    # TODO: add additional risk measures defined in the paper
    dists = [
        Distributions.Uniform(0, 100),
        Distributions.Exponential(1 / 0.02),
        Distributions.GeneralizedPareto(0, 580.4 / 12.61, 1 / 12.61),
        Distributions.Weibull(0.5, 5^(1 / 0.5)),
        Distributions.Weibull(1.5, 412.2^(1 / 1.5)),
    ]
    cte_targets = [
        [62.6, 75.0, 87.5, 97.5, 99.5],
        [64.38, 84.66, 119.31, 199.79, 280.26],
        [64.54, 85.61, 123.25, 219.04, 327.87],
        [66.45, 96.67, 167.36, 424.15, 810.45],
        [62.01, 76.23, 97.32, 138.63, 174.22],
    ]
    var_targets = [
        [25.0, 50.0, 75.0, 95.0, 99.0],
        [14.38, 34.66, 69.31, 149.79, 230.26],
        [13.39, 32.8, 67.45, 155.64, 255.84],
        [2.07, 12.01, 48.05, 224.36, 530.19],
        [24.14, 43.38, 68.86, 115.1, 153.31],
    ]
    alphas = [0.25, 0.5, 0.75, 0.95, 0.99]
    @testset "distribution $dist" for (i, dist) in enumerate(dists)
        @testset "alpha $α" for (j, α) in enumerate(alphas)
            @test CTE(α)(dist) ≈ cte_targets[i][j] rtol = 1.0e-2
            @test VaR(α)(dist) ≈ var_targets[i][j] rtol = 1.0e-2
        end
    end

    # Hardy, "An Introduction to Risk Measures for Actuarial Applications
    # note the difference for VaR: our VaR is L(⌈Nα⌉), the smallest k with
    # k/N ≥ α (the lower empirical quantile), as opposed to the smoothed
    # empirical estimate

    # Also, confusingly the examples for VaR don't use the same Table 1 (L) as CTE
    L = append!(
        vec(
            [
                169.1 170.4 171.3 171.9 172.3 173.3 173.8 174.3 174.9 175.9
                176.4 177.2 179.1 179.7 180.2 180.5 181.9 182.6 183.0 183.1
                183.3 184.4 186.9 187.7 188.2 188.5 191.8 191.9 193.1 193.8
                194.2 196.3 197.6 197.8 199.1 200.5 200.5 200.5 202.8 202.9
                203.0 203.7 204.4 204.8 205.1 205.8 206.7 207.5 207.9 209.2
                209.5 210.6 214.7 217.0 218.2 226.2 226.3 226.9 227.5 227.7
                229.0 231.4 231.6 233.2 237.5 237.9 238.1 240.3 241.0 241.3
                241.6 243.8 244.0 247.2 247.8 248.8 254.1 255.6 255.9 257.4
                265.0 265.0 268.9 271.2 271.6 276.5 279.2 284.1 284.3 287.8
                287.9 298.7 301.6 305.0 313.0 323.8 334.5 343.5 350.3 359.4
            ]
        ), zeros(900)
    ) |> sort

    @test VaR(0.95)(L) ≈ L[950] atol = 1.0e-2
    @test VaR(0.9505)(L) ≈ L[951] atol = 1.0e-2
    @test VaR(0.951)(L) ≈ L[951] atol = 1.0e-2
    @test VaR(0.95)(L) ≈ L[950] atol = 1.0e-2
    @test CTE(0.95)(L) ≈ 260.68 atol = 1.0e-1
    @test CTE(0.99)(L) ≈ 321.8 atol = 1.0e-1

    @testset "lower-quantile VaR" begin
        @test VaR(0.5)(Bernoulli(0.5)) == 0
        @test VaR(0.95)(collect(1.0:1000.0)) == 950
        @test VaR(0.8)([1.0, 1.0, 1.0, 1.0, 2.0]) == 1
        @test VaR(prevfloat(0.95))(collect(1.0:1000.0)) == 950
        @test VaR(nextfloat(0.95))(collect(1.0:1000.0)) == 951
        # parity: named discrete law ≡ equal-weight atomic representation ≡ quantile
        b = Binomial(10, 0.3)
        bdnp = DiscreteNonParametric(collect(support(b)), pdf.(b, support(b)))
        for α in (0.0, 0.2, 0.65, 0.99)
            @test VaR(α)(b) == VaR(α)(bdnp) == quantile(b, α)
        end
        # exact atom boundaries use F ≥ α (the lower quantile). A one-ulp step
        # above the boundary is not assertable for named laws: Distributions'
        # `quantile` is only cdf-consistent to within a ulp, so probe the
        # midpoint of the cdf gap instead.
        F3 = cdf(b, 3)
        @test VaR(F3)(b) == 3
        @test VaR(prevfloat(F3))(b) == 3
        @test VaR((F3 + cdf(b, 4)) / 2)(b) == 4
        @test VaR(0.6)(A) == 0.0            # cdf(A, 0) == 0.6 exactly
        @test VaR(nextfloat(0.6))(A) == 1.0
        @test VaR(0.95)(A) == 1.0
        # VaR(0) is the essential infimum
        @test VaR(0.0)(Normal()) == -Inf
        @test VaR(0.0)(Pareto(1.5, 1)) == 1.0
        @test VaR(0.0)([3.0, 1.0, 2.0]) == 1.0
    end

    @testset "divergent / heavy-tailed risks" begin
        @test isnan(RiskMeasures.Expectation()(Cauchy()))
        @test RiskMeasures.Expectation()(Pareto(0.5, 1.0)) == Inf
        @test isnan(CTE(0.0)(Cauchy()))     # α = 0 runs before any divergence shortcut
        @test CTE(0.95)(Cauchy()) == Inf
        @test CTE(0.5)(Pareto(0.5, 1.0)) == Inf
        @test CTE(0.95)(Pareto(1.0, 1.0)) == Inf
        # mean == -Inf is not a divergence shortcut: the upper tail is finite here
        @test CTE(0.001)(-1 * Pareto(0.5, 1.0)) ≈ -1000
        @test VaR(0.95)(Cauchy()) == quantile(Cauchy(), 0.95)
        # a distorted expectation that cannot be verified throws instead of
        # returning a plausible finite number
        @test_throws ErrorException WangTransform(0.9)(Cauchy())
        @test_throws ErrorException ProportionalHazard(2)(Pareto(0.5, 1.0))
        @test_throws ErrorException DualPower(2)(Pareto(0.5, 1.0))
    end

    @testset "stable distortions and certified quadrature" begin
        # the naive 1-(1-x)^v distortion returns ≈4.4999861 here while its
        # quadrature error certificate claims ≈7e-8 — the stable form keeps the
        # certificate honest
        @test DualPower(2)(Pareto(1.5, 1)) ≈ 4.5 rtol = 1.0e-6
        @test WangTransform(0.9)(Normal(0, 1)) ≈ quantile(Normal(), 0.9) rtol = 1.0e-6
        # two nearly equal one-sided integrals cancel to ~0; the acceptance rule
        # certifies at the components' scale instead of throwing
        @test abs(WangTransform(0.5)(Normal(0, 1))) < 1.0e-8
        # gbar is the stable complement of g
        for rm in (WangTransform(0.9), DualPower(2), ProportionalHazard(2), CTE(0.9), VaR(0.9))
            for F in (0.2, 0.5, 0.9)
                @test RiskMeasures.gbar(rm, F) ≈ 1 - RiskMeasures.g(rm, 1 - F) atol = 1.0e-15
            end
        end
    end

    @testset "atomic robustness" begin
        # Distributions accepts probability vectors off by ~1e-8; survival values
        # must be normalized and clamped or Φ⁻¹(1 + ε) throws a DomainError
        sloppy = DiscreteNonParametric([1.0, 2.0, 3.0], [0.1, 0.2, 0.7000000001])
        @test isfinite(WangTransform(0.95)(sloppy))
        # zero-probability atoms at ±Inf must not poison the sum through Inf * 0
        @test VaR(0.5)(DiscreteNonParametric([0.0, Inf], [0.9, 0.1])) == 0
        @test RiskMeasures.Expectation()(DiscreteNonParametric([1.0, Inf], [1.0, 0.0])) == 1.0
        # an equal-weight sample and its atomic representation agree for a
        # distortion with no dedicated fast path
        xs = [0.0, 1.0, 5.0]
        @test WangTransform(0.9)(xs) ≈ WangTransform(0.9)(DiscreteNonParametric(xs, fill(1 / 3, 3)))
        # finite-support named laws ≡ their atomic representations
        for d in (Binomial(10, 0.3), DiscreteUniform(1, 6), Bernoulli(0.3), Dirac(2.5))
            atoms = collect(support(d))
            dnp = DiscreteNonParametric(atoms, pdf.(d, atoms))
            @test WangTransform(0.9)(d) ≈ WangTransform(0.9)(dnp)
            @test CTE(0.9)(d) ≈ CTE(0.9)(dnp)
        end
        # infinite-support discrete laws: checked tail summation, cross-checked
        # against an explicit truncation (tail mass beyond 60 is ~1e-40)
        @test WangTransform(0.9)(Poisson(10)) ≈ 14.2936912334644 rtol = 1.0e-8
        p10 = Poisson(10)
        p10dnp = DiscreteNonParametric(collect(0:60), pdf.(p10, 0:60))
        @test CTE(cdf(p10, 12))(p10) ≈ CTE(cdf(p10, 12))(p10dnp) rtol = 1.0e-10
        @test isfinite(CTE(0.9)(Geometric(0.2)))
        # zero-mass prefixes must not read as convergence: Poisson(1000) has
        # ccdf == 1.0 for its first hundreds of atoms (all distorted weights 0),
        # and DelayedDirac concentrates all mass at 100 with minimum 0
        p1k = Poisson(1000)
        p1kdnp = DiscreteNonParametric(collect(600:1400), pdf.(p1k, 600:1400))
        @test WangTransform(0.9)(p1k) ≈ WangTransform(0.9)(p1kdnp) rtol = 1.0e-8
        @test WangTransform(0.9)(DelayedDirac()) ≈ 100.0
        @test CTE(0.5)(DelayedDirac()) ≈ 100.0
        @test ProportionalHazard(2)(DelayedDirac()) ≈ 100.0
    end

    @testset "mean fallback and dispatch" begin
        # generic truncated wrappers have no `mean` method; Expectation falls
        # back to checked quadrature
        tr = truncated(Gamma(2, 3), 1, 5)
        @test RiskMeasures.Expectation()(tr) ≈ QuadGK.quadgk(x -> x * pdf(tr, x), 1, 5)[1] rtol = 1.0e-6
        @test CTE(0.0)(tr) ≈ RiskMeasures.Expectation()(tr)
        @test VaR(0.9)(tr) <= CTE(0.9)(tr) <= 5
        # censored distributions have a real `mean` method; it is used directly
        @test RiskMeasures.Expectation()(censored(Normal(), -1, 1)) == 0.0
        # an error inside a distribution's own `mean` must propagate
        @test_throws MethodError RiskMeasures.Expectation()(BrokenMeanDist())
        # a NaN cdf must surface as an error, not a silent number
        @test_throws Exception WangTransform(0.9)(NaNCDFRisk())
        # analytic exactness on distributions
        @test RiskMeasures.Expectation()(Normal(3, 1)) == 3.0
        @test VaR(0.95)(LogNormal(0, 1)) == quantile(LogNormal(0, 1), 0.95)
    end

    @testset "numeric types and accumulation" begin
        @test RiskMeasures.Expectation()([1.0e308, 1.0e308]) == 1.0e308
        @test RiskMeasures.Expectation()(fill(typemax(Int), 4)) ≈ 9.223372036854776e18
        @test RiskMeasures.Expectation()(Float32[1, 2, 3]) ≈ 2
        @test CTE(0.5)(BigFloat[1, 2, 3, 4]) ≈ big"3.5"
        for rm in (VaR(0.5), CTE(0.5), RiskMeasures.Expectation(), WangTransform(0.5))
            @test_throws ArgumentError rm(Float64[])
        end
    end

end
