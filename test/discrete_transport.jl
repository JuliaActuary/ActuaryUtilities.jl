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

    # A flat cdf gives a quantile jump at the atomic break. The second interval
    # starts at Q(0.5+) = 2, not Q(0.5) = 1, so its largest gap is 10 - 2.
    mix = MixtureModel([Uniform(0, 1), Uniform(2, 3)])
    gapped = DiscreteNonParametric([0.0, 10.0], [0.5, 0.5])
    @test wasserstein(gapped, mix; p = Inf, maxevals = 1) ≈ 8.0
    @test wasserstein(mix, gapped; p = Inf) ≈ 8.0
    @test wasserstein([0.0, 10.0], mix; p = Inf) ≈ 8.0
    # The upper endpoint still uses the left limit of the break.
    @test wasserstein(DiscreteNonParametric([-10.0, 3.0], [0.5, 0.5]), mix; p = Inf) ≈ 11.0
    inactive_support = MixtureModel([Uniform(-100, -99), Uniform(0, 1), Uniform(99, 100)], [0.0, 1.0, 0.0])
    @test wasserstein(Dirac(0.0), inactive_support; p = Inf) ≈ 1.0

    # Integer support must be promoted before exponentiation. Check both sample
    # paths, weighted laws, and a Binomial against its analytic fourth moment.
    for target in ([0, 0], [0, 0, 0], Dirac(0))
        @test wasserstein([0, 10^7], target; p = 3) ≈ 10^7 / cbrt(2)
    end
    @test wasserstein(DiscreteNonParametric([0, 10^7], [0.5, 0.5]), Dirac(0); p = 3) ≈ 10^7 / cbrt(2)
    n = 10^6
    fourth_moment = (Float64(n)^4 + 6 * Float64(n)^3 + 3 * Float64(n)^2 - 2 * n) / 16
    @test wasserstein(Binomial(n, 0.5), Dirac(0); p = 4) ≈ fourth_moment^(1 / 4)

    # Compare integer-rank sample sweeps with independent rank expansion onto
    # an LCM grid, including shared breaks, coprime sizes, and repeated values.
    rng = MersenneTwister(159)
    for (na, nb) in ((1, 1), (6, 6), (6, 4), (7, 11)), p in (1, 2, 3, Inf)
        sa, sb = rand(rng, -4:4, na), rand(rng, -4:4, nb)
        count = lcm(na, nb)
        gaps = abs.(repeat(sort(sa); inner = count ÷ na) .- repeat(sort(sb); inner = count ÷ nb))
        expected = isinf(p) ? maximum(gaps) : (sum(gaps .^ p) / count)^(1 / p)
        @test wasserstein(sa, sb; p, maxevals = 1) ≈ expected
        @test wasserstein(sb, sa; p) ≈ expected
    end
    @test_throws ArgumentError wasserstein(Float64[], [0.0])
    @test_throws ArgumentError wasserstein([0.0], Float64[])
    @test wasserstein([Inf], [Inf]) == 0
    @test wasserstein([Inf], [Inf, Inf]; p = Inf) == 0
    @test wasserstein(BigFloat[0, 1, 2], BigFloat[0, 0]) ≈ big"1.0" rtol = big"1e-60"

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

@testset "Integer samples do not wrap around" begin
    # The gap between two integers is taken in the accumulator's float type, not the sample's.
    @test wasserstein(UInt8[0], UInt8[1]) == 1
    @test wasserstein(Int8[-100], Int8[100]) == 200
    @test wasserstein(Int8[-100], Int8[100]; p = Inf) == 200
    @test wasserstein([typemin(Int)], [typemax(Int)]) == Float64(typemax(Int)) - Float64(typemin(Int))
    # The same samples in any representation give the same distance, in either order:
    # equal sizes, unequal sizes, and finite discrete laws.
    equal_a, equal_b = [0, 3, 7, 120], [5, 1, 127, 0]
    short = [100, 2, 9]
    law(T, xs, ps) = DiscreteNonParametric(T.(xs), ps)
    for p in (1, 2, Inf), (a, b) in ((equal_a, equal_b), (equal_a, short), (short, equal_b))
        reference = wasserstein(Float64.(a), Float64.(b); p)
        laws = ((T -> law(T, [0, 3, 120], [0.2, 0.3, 0.5])), (T -> law(T, [5, 127], [0.6, 0.4])))
        law_reference = wasserstein(laws[1](Float64), laws[2](Float64); p)
        for T in (UInt8, Int8, Int64, Float64)
            @test wasserstein(T.(a), T.(b); p) == reference
            @test wasserstein(T.(b), T.(a); p) == reference
            @test wasserstein(laws[1](T), laws[2](T); p) == law_reference
            @test wasserstein(laws[2](T), laws[1](T); p) == law_reference
            @test wasserstein(T.(a), laws[2](T); p) == wasserstein(Float64.(a), laws[2](Float64); p)
        end
    end
end
