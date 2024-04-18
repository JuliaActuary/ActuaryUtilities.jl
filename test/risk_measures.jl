@testset "Risk Measures" begin

    @test_throws AssertionError VaR(-0.5)
    @test_throws AssertionError VaR(1.0)
    @test_throws AssertionError VaR(1.5)
    @test_throws AssertionError CTE(-0.5)
    @test_throws AssertionError CTE(1.0)
    @test_throws AssertionError CTE(1.5)
    @test_throws AssertionError WangTransform(0.0)
    @test_throws AssertionError WangTransform(1.0)

    # https://utstat.utoronto.ca/sam/coorses/act466/rmn.pdf pg 17
    @test RiskMeasures.g(WangTransform(cdf(Normal(), 1)), 1 - cdf(LogNormal(0, 1), 12)) ≈ 0.06879 atol = 1e-5
    @test RiskMeasures.Expectation()(LogNormal(0, 2 * 1)) ≈ mean(LogNormal(0, 2 * 1))


    @test CTE(0.9)(Uniform(-1, 0)) ≈ -0.05 atol = 1e-8
    @test RiskMeasures.Expectation()(Uniform(-1, 0)) ≈ -0.5 atol = 1e-8
    @test CTE(0.0)(Uniform(0, 1) - 0.5) ≈ 0.0 atol = 1e-8
    @test CTE(0.5)(Uniform(0, 1) - 0.5) ≈ 0.25 atol = 1e-8

    @test CTE(0.0)(Distributions.Normal(0, 1)) ≈ 0
    @test RiskMeasures.Expectation()(Distributions.Normal(3, 1)) ≈ 3

    # http://actuaries.org/events/congresses/cancun/afir_subject/afir_14_wang.pdf
    A = Distributions.DiscreteNonParametric([0.0, 1.0, 5.0], [0.6, 0.375, 0.025])
    B = Distributions.DiscreteNonParametric([0.0, 1.0, 11.0], [0.6, 0.390, 0.01])
    @test WangTransform(0.95)(A) ≈ 2.42 atol = 1e-2
    @test WangTransform(0.95)(B) ≈ 3.40 atol = 1e-2
    @test CTE(0.95)(A) ≈ 3
    @test CTE(0.95)(B) ≈ 3

    ## example 4.3
    @test WangTransform(0.9)(LogNormal(3, 2)) ≈ exp(3 + quantile(Normal(), 0.9) * 2 + 2^2 / 2) atol = 1e-3

    ## example 4.4
    C = Distributions.Exponential(1)
    α = 0.99
    @test CTE(α)(C) ≈ 5.61 atol = 1e-2
    @test VaR(α)(C) ≈ 4.61 atol = 1e-2
    @test WangTransform(α)(C) ≈ 5.02 atol = 1e-1

    ## example 4.5
    @test WangTransform(α)(Uniform()) ≈ 0.95 atol = 1e-2

    # Sepanski & Wang, "New Classes of Distortion Risk Measures and Their Estimation, Table 6
    # note the parameterization of Exp, Lomax (GP), and Weibull is different in Julia
    # than in the paper
    # TODO: add additional risk measures defined in the paper
    dists = [
        Distributions.Uniform(0, 100),
        Distributions.Exponential(1 / 0.02),
        Distributions.GeneralizedPareto(0, 580.40 / 12.61, 1 / 12.61),
        Distributions.Weibull(0.50, 5^(1 / 0.5)),
        Distributions.Weibull(1.50, 412.2^(1 / 1.5))
    ]
    cte_targets = [
        [62.6, 75.0, 87.5, 97.5, 99.5],
        [64.38, 84.66, 119.31, 199.79, 280.26],
        [64.54, 85.61, 123.25, 219.04, 327.87],
        [66.45, 96.67, 167.36, 424.15, 810.45],
        [62.01, 76.23, 97.32, 138.63, 174.22]
    ]
    var_targets = [
        [25.0, 50.0, 75.0, 95.0, 99.0],
        [14.38, 34.66, 69.31, 149.79, 230.26],
        [13.39, 32.80, 67.45, 155.64, 255.84],
        [2.07, 12.01, 48.05, 224.36, 530.19],
        [24.14, 43.38, 68.86, 115.10, 153.31]
    ]
    alphas = [0.25, 0.5, 0.75, 0.95, 0.99]
    @testset "distribution $dist" for (i, dist) in enumerate(dists)
        @testset "alpha $α" for (j, α) in enumerate(alphas)
            @test CTE(α)(dist) ≈ cte_targets[i][j] rtol = 1e-2
            @test VaR(α)(dist) ≈ var_targets[i][j] rtol = 1e-2
        end
    end

    # Hardy, "An Introduction to Risk Measures for Actuarial Applications
    # note the difference for VaR where our VaR is L(Nα+1), as opposed to L(Nα) 
    # or the smoothed empirical estimate

    # Also, confusingly the examples for VaR don't use the same Table 1 (L) as CTE
    L = append!(vec([
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
        ]), zeros(900)) |> sort

    @test VaR(0.950)(L) ≈ L[951] atol = 1e-2
    @test VaR(0.9505)(L) ≈ L[951] atol = 1e-2
    @test VaR(0.951)(L) ≈ L[952] atol = 1e-2
    @test VaR(0.95)(L) ≈ L[951] atol = 1e-2
    @test CTE(0.95)(L) ≈ 260.68 atol = 1e-1
    @test CTE(0.99)(L) ≈ 321.8 atol = 1e-1

end

