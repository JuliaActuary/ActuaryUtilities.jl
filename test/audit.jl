# Regression and equivalence tests from the 2026-06 ecosystem audit
@testset "analytic fast paths match the generic AD path" begin
    # the generic scalar path: nested ForwardDiff through `i + yield`
    generic_duration(yield, cfs, times) = duration(yield, i -> ActuaryUtilities.FinancialMath.price(i, cfs, times))
    function generic_convexity(yield, cfs, times)
        vf = i -> ActuaryUtilities.FinancialMath.price(i, cfs, times)
        v(x) = abs(vf(yield + x))
        ForwardDiff.derivative(y -> ForwardDiff.derivative(v, y), 0.0) / v(0.0)
    end

    cases = [
        ([5.0, 5.0, 105.0], [1.0, 2.0, 3.0]),
        ([0.0, 0.0, 0.0, 0.0, 100.0], [1.0, 2.0, 3.0, 4.0, 5.0]),
        ([10.0, -2.0, 10.0, 110.0], [0.5, 1.0, 1.5, 2.0]),   # mixed signs
        ([-5.0, -5.0, -105.0], [1.0, 2.0, 3.0]),             # liability (all negative)
    ]
    yields = [
        0.03,
        0.0,
        -0.01,
        FC.Periodic(0.03, 1),
        FC.Periodic(0.04, 2),
        FC.Periodic(0.06, 12),
        FC.Continuous(0.03),
        FM.Yield.Constant(0.03),
        FM.Yield.Constant(FC.Continuous(0.03)),
        FM.Yield.Constant(FC.Periodic(0.04, 2)),
    ]
    @testset "yield=$y" for y in yields
        for (cfs, times) in cases
            @test duration(Modified(), y, cfs, times) ≈ generic_duration(y, cfs, times) rtol = 1.0e-12
            @test duration(y, cfs, times) ≈ generic_duration(y, cfs, times) rtol = 1.0e-12
            @test convexity(y, cfs, times) ≈ generic_convexity(y, cfs, times) rtol = 1.0e-12
        end
    end

    @testset "fast-path dispatch is actually selected" begin
        cfs = [5.0, 5.0, 105.0]
        times = [1.0, 2.0, 3.0]
        generic_sig = Tuple{typeof(duration), Modified, Any, Any, Any}
        for y in (0.03, FC.Periodic(0.04, 2), FC.Continuous(0.03), FM.Yield.Constant(0.03), FM.Yield.Constant(FC.Continuous(0.03)))
            m = which(duration, (Modified, typeof(y), typeof(cfs), typeof(times)))
            # an analytic method must be selected, not the generic AD fallback —
            # the prior fast paths were unreachable (`Constant{<:Continuous}` can
            # never match `Constant{<:Rate}`) and silently fell through
            @test m.sig != generic_sig
            cm = which(convexity, (typeof(y), typeof(cfs), typeof(times)))
            @test cm.sig != Tuple{typeof(convexity), Any, Any, Any}
        end
    end

    @testset "Cashflow vectors route through the fast paths with embedded times" begin
        cfs = FC.Cashflow.([5.0, 5.0, 105.0], [1.0, 2.0, 3.0])
        @test duration(0.03, cfs) ≈ duration(0.03, [5.0, 5.0, 105.0], [1.0, 2.0, 3.0])
        @test convexity(0.03, cfs) ≈ convexity(0.03, [5.0, 5.0, 105.0], [1.0, 2.0, 3.0])
    end

    @testset "AD through the fast paths (Dual yields)" begin
        cfs = [5.0, 5.0, 105.0]
        times = [1.0, 2.0, 3.0]
        # sensitivity of duration to the yield level — exercises Dual <: Real dispatch
        d_dy = ForwardDiff.derivative(y -> duration(y, cfs, times), 0.03)
        h = 1.0e-7
        fd = (duration(0.03 + h, cfs, times) - duration(0.03 - h, cfs, times)) / 2h
        @test d_dy ≈ fd rtol = 1.0e-6
    end
end

@testset "present_values" begin
    @test present_values(0.00, [1, 1, 1]) ≈ [3.0, 2.0, 1.0]
    # pvs[k] is the value at times[k-1] (time zero for k = 1) of flows k..n
    v = present_values(0.1, [10, 20], [0, 1])
    @test v ≈ [10 + 20 / 1.1, 20 / 1.1]

    # matches a direct per-timepoint computation
    cfs = [100.0, 100.0, 100.0, 100.0]
    times = [1.0, 2.0, 3.0, 4.0]
    pvs = present_values(0.05, cfs, times)
    for k in eachindex(times)
        from = k == 1 ? 0.0 : times[k - 1]
        direct = sum(cfs[j] * FC.discount(0.05, from, times[j]) for j in k:length(cfs))
        @test pvs[k] ≈ direct
    end

    # long streams no longer overflow the stack (previously recursion depth = n)
    n = 100_000
    long = present_values(0.0001, fill(1.0, n))
    @test length(long) == n
    @test long[end] ≈ 1.0 / 1.0001

    # AD propagates (previously the accumulator was hardcoded Float64)
    g = ForwardDiff.derivative(r -> sum(present_values(r, [10.0, 20.0], [1.0, 2.0])), 0.05)
    @test g < 0 # value decreases in the rate

    @test_throws DimensionMismatch present_values(0.05, [1, 2], [1.0])
end

@testset "risk measure exact empirical estimators" begin
    L = collect(1.0:1000.0)
    @test VaR(0.95)(L) == 951.0
    @test VaR(0.0)(L) == 1.0
    @test CTE(0.0)(L) ≈ sum(L) / 1000
    # CTE Choquet weights: crossing atom gets (k/n - α), the rest 1/n, all / (1-α)
    α = 0.95
    k = 951
    expected = ((k / 1000 - α) * L[k] + sum(L[(k + 1):end]) / 1000) / (1 - α)
    @test CTE(α)(L) ≈ expected

    # duplicates / plateaus are handled exactly (quadrature used to wobble here)
    dup = [1.0, 1.0, 1.0, 1.0, 2.0]
    @test VaR(0.5)(dup) == 1.0
    @test VaR(0.8)(dup) == 2.0
    @test CTE(0.8)(dup) ≈ 2.0

    # unsorted input
    shuffled = shuffle(Xoshiro(1), L)
    @test VaR(0.95)(shuffled) == 951.0
    @test CTE(0.95)(shuffled) ≈ expected

    # the exact estimator agrees with the (quadrature) Choquet definition it replaces
    sample = rand(Xoshiro(2026), 5000) .* 2 .- 0.5 # straddles zero (both integrals exercised)
    ecdf_choquet = let F = StatsBase.ecdf(sample), rm = CTE(0.9)
        H(x) = 1 - ActuaryUtilities.RiskMeasures.g(rm, 1 - x)
        i1, _ = QuadGK.quadgk(x -> 1 - H(F(x)), 0, Inf)
        i2, _ = QuadGK.quadgk(x -> H(F(x)), -Inf, 0)
        i1 - i2
    end
    @test CTE(0.9)(sample) ≈ ecdf_choquet atol = 1.0e-6

    @test Expectation()(L) ≈ 500.5

    # aliases are the same functions
    @test ValueAtRisk === VaR
    @test ConditionalTailExpectation === CTE

    # distortion measures against closed forms on Uniform(0,1):
    # ρ[DualPower(v)] = v/(v+1); ρ[ProportionalHazard(y)] = y/(y+1)
    @test DualPower(2)(Uniform(0, 1)) ≈ 2 / 3 atol = 1.0e-6
    @test DualPower(3)(Uniform(0, 1)) ≈ 3 / 4 atol = 1.0e-6
    @test ProportionalHazard(2)(Uniform(0, 1)) ≈ 2 / 3 atol = 1.0e-6
    @test ProportionalHazard(4)(Uniform(0, 1)) ≈ 4 / 5 atol = 1.0e-6
end

@testset "spread Newton solve" begin
    cfs = fill(10.0, 10)
    s = spread(0.04, 0.05, cfs)
    # repricing to near machine precision (NelderMead only achieved ~sqrt(tol))
    @test FC.pv(0.04 + s, cfs) ≈ FC.pv(0.05, cfs) rtol = 1.0e-12

    # duration-neutral mixed-sign portfolio: f′(0) ≈ 0 — an undamped Newton
    # step launched the iterate out of the valid domain (DomainError); the
    # damped step must still land on an exact root
    dn = [100.0, -52.0]
    sdn = spread(0.04, 0.05, dn)
    @test FC.pv(0.04 + sdn, dn) ≈ FC.pv(0.05, dn) atol = 1.0e-10
    rates = [0.01, 0.01, 0.03, 0.05, 0.07, 0.16, 0.35, 0.92, 1.40, 1.74, 2.31, 2.41] ./ 100
    mats = [1 / 12, 2 / 12, 3 / 12, 6 / 12, 1, 2, 3, 5, 7, 10, 20, 30]
    y = FM.fit(FM.Spline.Linear(), FM.CMTYield.(rates, mats), FM.Fit.Bootstrap())
    s2 = spread(y, y + 0.01, cfs)
    @test FC.pv(y + s2, cfs) ≈ FC.pv(y + 0.01, cfs) rtol = 1.0e-12
end

@testset "moic degenerate input errors" begin
    @test moic([-10, 20, 30]) ≈ 5.0
    @test_throws ArgumentError moic([10, 20, 30])
    @test_throws ArgumentError moic([-10, -20])
end

@testset "duration with a negative-valued valuation function" begin
    liability(i) = -100 / (1 + i)^5
    @test duration(0.03, liability) ≈ duration(0.03, i -> 100 / (1 + i)^5)
    @test convexity(0.03, liability) ≈ convexity(0.03, i -> 100 / (1 + i)^5)
end

@testset "legacy KeyRateDuration conveniences" begin
    rf_curve = FM.fit(FM.Spline.Cubic(), FM.ZCBYield.([0.04, 0.05, 0.055, 0.06, 0.062], 1:5), FM.Fit.Bootstrap())
    cfs_real = fill(10.0, 5)
    cfs_cf = FC.Cashflow.(fill(10.0, 5), [1.0, 2.0, 3.0, 4.0, 5.0])
    # a Cashflow vector uses embedded times for the krd grid, equal to the plain form
    @test duration(KeyRate(2), rf_curve, cfs_cf) ≈ duration(KeyRate(2), rf_curve, cfs_real, 1:5)
    # all-sub-1-year cashflows have an empty default grid: loud error, not a crash
    @test_throws ArgumentError duration(KeyRate(0.5), rf_curve, [10.0], [0.5])
    # a shifted timepoint outside the krd grid is a loud error, not a MethodError
    @test_throws ArgumentError duration(KeyRate(7), rf_curve, cfs_cf)
end

@testset "mismatched cfs/times lengths error loudly" begin
    # the analytic fast paths index times by eachindex(cfs) under @inbounds;
    # a silent mismatch must not read out of bounds (or zip-truncate)
    @test_throws DimensionMismatch duration(0.03, [1.0, 2.0, 3.0], [1.0, 2.0])
    @test_throws DimensionMismatch convexity(0.03, [1.0, 2.0, 3.0], 1:2)
end

@testset "locked_floater requires whole coupon periods on the forward leg" begin
    fl = FM.Bond.Floating(0.0, FC.Periodic(2), 3.0, "OIS")
    # aligned: remaining term 2.5y = 5 semiannual periods — constructs fine
    @test locked_floater(fl, 0.02, 0.5) isa FC.Composite
    # non-commensurate remaining term (2.75y = 5.5 periods) would put a stub
    # first coupon on the forward leg whose reference forward starts before
    # time zero — quietly mispriced on extrapolating curves, DomainError on
    # ZeroRateCurve — so it must refuse loudly
    @test_throws ArgumentError locked_floater(fl, 0.02, 0.25)
end

@testset "two-curve scalar convexity matches the AD path" begin
    base = FM.Yield.Constant(0.03)
    credit = FM.Yield.Constant(0.015)
    tenors = [1.0, 2.0, 5.0]
    cfs = [5.0, 5.0, 105.0]
    times = [1.0, 2.0, 3.0]
    an = convexity(base, credit, tenors, cfs, times)
    # the AD do-block form computes the same blocks via a (2n)² Hessian
    vf2 = (b, c) -> sum(cf * b(t) * c(t) for (cf, t) in zip(cfs, times))
    ad = convexity(vf2, base, credit, tenors)
    @test an.base ≈ ad.base rtol = 1.0e-10
    @test an.credit ≈ ad.credit rtol = 1.0e-10
    @test an.cross ≈ ad.cross rtol = 1.0e-10
end
