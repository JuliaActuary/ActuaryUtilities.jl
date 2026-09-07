@testset "Exact finite discrete transport" begin
    a = DiscreteNonParametric([0, 1, 2], [0.5, 0.00001, 0.49999])
    b = DiscreteNonParametric([0, 2], [0.50001, 0.49999])
    for p in (1, 2, 3, Inf)
        expected = isinf(p) ? 1.0 : 0.00001^(1 / p)
        @test wasserstein(a, b; p, maxevals = 1) ≈ expected rtol = 1.0e-10
        @test wasserstein(b, a; p) ≈ expected rtol = 1.0e-10
    end

    # Repeated empirical outcomes and finite laws represent the same measure.
    sample = [2, 0, 1, 2, 0, 2]
    law = DiscreteNonParametric([0, 1, 2], [2 // 6, 1 // 6, 3 // 6])
    target = [-1, 0, 3, 5]
    target_law = DiscreteNonParametric(target, fill(0.25, 4))
    for p in (1, 2, 3, Inf)
        reference = wasserstein(sample, target; p)
        @test wasserstein(law, target; p) ≈ reference
        @test wasserstein(sample, target_law; p) ≈ reference
        @test wasserstein(law, target_law; p) ≈ reference
        @test wasserstein(sample, law; p) ≈ 0 atol = 1.0e-14
        @test wasserstein(Binomial(3, 0.5), Dirac(0); p) ≈
            (isinf(p) ? 3.0 : sum(k^p * pdf(Binomial(3, 0.5), k) for k in 0:3)^(1 / p))
    end

    # Against Uniform(0,1), integrate |x-u| separately over each atom's rank
    # interval; the supremum occurs at an interval endpoint (one-sided limits).
    atomic = DiscreteNonParametric([0.0, 1.0], [0.8, 0.2])
    @test wasserstein(atomic, Uniform()) ≈ (0.8^2 + 0.2^2) / 2
    @test wasserstein(atomic, Uniform(); p = 2) ≈ sqrt((0.8^3 + 0.2^3) / 3)
    @test wasserstein(atomic, Uniform(); p = Inf, maxevals = 1) == 0.8
    @test wasserstein(Uniform(), atomic; p = Inf) == 0.8
    @test wasserstein(atomic, Normal(); p = Inf) == Inf

    # Zero-mass extreme atoms do not contribute a gap or an infinite cost.
    zero_atom = DiscreteNonParametric([0.0, 1.0, Inf], [0.5, 0.5, 0.0])
    @test wasserstein(zero_atom, [0.0, 1.0]; p = Inf) == 0
    @test wasserstein(Dirac(Inf), Dirac(Inf)) == 0
    @test wasserstein(Dirac(Inf), Dirac(0.0)) == Inf
    @test Expectation()(zero_atom) ≈ 0.5
    @test CTE(0.5)(zero_atom) ≈ 1.0
    for rm in (Expectation(), CTE(0.6), WangTransform(0.3), DualPower(2))
        @test rm(law) ≈ rm(sample)
    end
end
