# Regression and equivalence tests from the 2026-06 ecosystem audit
@testset "analytic fast paths match the generic AD path" begin
    # The generic scalar path uses the input's own compounding convention for
    # rates and a continuous-zero shift for yield models.
    generic_duration(yield, cfs, times) = duration(i -> FC.pv(i, cfs, times), yield)
    parallel_bump(yield, x) = yield + x
    parallel_bump(yield::FM.Yield.AbstractYieldModel, x) =
        FM.Yield.TenorShift(yield, (z, t) -> FC.Continuous(x) + z)
    function generic_convexity(yield, cfs, times)
        vf = i -> FC.pv(i, cfs, times)
        v(x) = abs(vf(parallel_bump(yield, x)))
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
        PeriodicZeroSensitivityCurve(0.04),
    ]
    @testset "yield=$y" for y in yields
        for (cfs, times) in cases
            @test duration(Modified(), y, cfs, times) ≈ generic_duration(y, cfs, times) rtol = 1.0e-12
            @test duration(y, cfs, times) ≈ generic_duration(y, cfs, times) rtol = 1.0e-12
            @test convexity(y, cfs, times) ≈ generic_convexity(y, cfs, times) rtol = 1.0e-12
        end
    end

    @testset "fast-path dispatch is actually selected" begin
        # Every yield input must reach the analytic methods: the prior fast paths were
        # unreachable (`Constant{<:Continuous}` can never match `Constant{<:Rate}`) and silently
        # fell through to AD. The results are the closed forms summed in payment order, to the bit.
        analytic_duration = which(duration, (Modified, Float64, Vector{Float64}, Vector{Float64}))
        analytic_convexity = which(convexity, (Float64, Vector{Float64}, Vector{Float64}))
        @test analytic_duration.module === analytic_convexity.module === ActuaryUtilities.FinancialMath
        cfs = [5.0, 5.0, 105.0]
        times = [1.0, 2.0, 3.0]
        divisor(y::Real) = 1 + y
        divisor(y::FC.Rate{<:Real, FC.Periodic}) = 1 + FC.rate(y) / y.compounding.frequency
        divisor(y) = 1
        inv_m(y::Real) = 1
        inv_m(y::FC.Rate{<:Real, FC.Periodic}) = 1 / y.compounding.frequency
        inv_m(y) = 0
        for y in (0.03, FC.Periodic(0.04, 2), FC.Continuous(0.03), FM.Yield.Constant(0.03), FM.Yield.Constant(FC.Continuous(0.03)))
            V = Vt = Vtt = 0.0
            for (cf, t) in zip(cfs, times)
                cfd = cf * FC.discount(y, t)
                V += cfd
                Vt += t * cfd
                Vtt += t * (t + inv_m(y)) * cfd
            end
            @test which(duration, (Modified, typeof(y), typeof(cfs), typeof(times))) === analytic_duration
            @test which(convexity, (typeof(y), typeof(cfs), typeof(times))) === analytic_convexity
            @test duration(Modified(), y, cfs, times) === Vt / V / divisor(y)
            @test duration(y, cfs, times) === Vt / V / divisor(y)
            @test duration(DV01(), y, cfs, times) === Vt / (divisor(y) * 10_000)
            @test convexity(y, cfs, times) === Vtt / V / divisor(y)^2
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
    @test present_values(0.0, [1, 1, 1]) ≈ [3.0, 2.0, 1.0]
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

    # A `Cashflow` is paid at its own time, whatever its paired time, and the entries are numbers.
    curve = FM.Yield.Constant(FC.Continuous(0.03))
    v = present_values(curve, [FC.Cashflow(100.0, 2.0)], [10.0])
    @test v isa Vector{Float64} && only(v) ≈ 100 * exp(-0.06)
    wrapped = [FC.Cashflow(100.0, 2.0), FC.Cashflow(50.0, 3.0)]
    @test present_values(curve, wrapped) ≈ [100 * exp(-0.06) + 50 * exp(-0.09), 50 * exp(-0.03)]
    @test present_values(curve, wrapped, [7.0, 9.0]) == present_values(curve, wrapped)
    numeric = present_values(curve, [100.0, 50.0], [2.0, 3.0])
    @test present_values(curve, wrapped) ≈ numeric
    # mixed: the number is paid at its paired time, after the `Cashflow`'s own time
    mixed = Any[FC.Cashflow(100.0, 2.0), 50.0]
    @test present_values(curve, mixed, [10.0, 3.0]) ≈ numeric
    @test present_values(curve, (FC.Cashflow(100.0, 2.0), FC.Cashflow(50.0, 3.0))) ≈ numeric
    @test present_values(curve, (cf for cf in wrapped)) ≈ numeric
    @test present_values(curve, OffsetArray(wrapped, 0:1)) ≈ numeric
    @test present_values(curve, [100.0, 50.0], OffsetArray([2.0, 3.0], -1:0)) ≈ numeric
    # number types: Float32 stays Float32 under a Float32 rate; BigFloat and AD propagate
    @test eltype(present_values(0.03f0, [FC.Cashflow(100.0f0, 2.0f0)])) == Float32
    big = present_values(FC.Continuous(big"0.03"), [FC.Cashflow(big"100.0", big"2.0")])
    @test eltype(big) == BigFloat && only(big) ≈ 100 * exp(-big"0.06")
    g = ForwardDiff.derivative(r -> sum(present_values(FC.Continuous(r), wrapped, [7.0, 9.0])), 0.03)
    @test g ≈ ForwardDiff.derivative(r -> sum(present_values(FC.Continuous(r), [100.0, 50.0], [2.0, 3.0])), 0.03)
    da = ForwardDiff.derivative(a -> first(present_values(curve, [FC.Cashflow(a, 2.0)])), 100.0)
    @test da ≈ exp(-0.06)
end

@testset "risk measure exact empirical estimators" begin
    L = collect(1.0:1000.0)
    # VaR is the lower empirical quantile: the smallest k with k/n ≥ α
    @test VaR(0.95)(L) == 950.0
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
    @test VaR(0.8)(dup) == 1.0   # k = 4 satisfies 4/5 ≥ 0.8, so the tied atom is selected
    @test CTE(0.8)(dup) ≈ 2.0

    # unsorted input
    shuffled = shuffle(Xoshiro(1), L)
    @test VaR(0.95)(shuffled) == 950.0
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
    rates = [0.01, 0.01, 0.03, 0.05, 0.07, 0.16, 0.35, 0.92, 1.4, 1.74, 2.31, 2.41] ./ 100
    mats = [1 / 12, 2 / 12, 3 / 12, 6 / 12, 1, 2, 3, 5, 7, 10, 20, 30]
    y = FM.fit(FM.Spline.Linear(), FM.CMTYield.(rates, mats), FM.Fit.Bootstrap())
    s2 = spread(y, y + 0.01, cfs)
    @test FC.pv(y + s2, cfs) ≈ FC.pv(y + 0.01, cfs) rtol = 1.0e-12
end

@testset "spread near the edge of its domain" begin
    # Near a combined annual rate of -100% the price derivative is so large that the
    # Newton step is tiny far from the root: a small step alone must not be accepted.
    for base in (-0.99, -0.999999, -1 + 1.0e-10, -1 + 1.0e-13)
        s = spread(base, 0.05, [1.0], [1.0])
        @test FC.rate(s) ≈ 0.05 - base rtol = 1.0e-12
        @test FC.pv(base + s, [1.0], [1.0]) ≈ FC.pv(0.05, [1.0], [1.0]) rtol = 1.0e-12
    end
    # Steps stop halfway to the domain edge, so spreads close to it are reachable: the
    # combined rate must exceed -1 (base -0.5) and so must the spread itself (base 0).
    @test FC.rate(spread(-0.5, -0.99, [1.0], [1.0])) ≈ -0.49 rtol = 1.0e-12
    @test FC.rate(spread(0.0, -0.9999, [1.0], [1.0])) ≈ -0.9999 rtol = 1.0e-12
    # a semiannual base adds the spread in its own convention: 2((1 + s)^(1/2) - 1) = -0.05
    @test FC.rate(spread(FC.Periodic(-1.9, 2), FC.Periodic(-1.95, 2), [1.0], [1.0])) ≈ 0.975^2 - 1 rtol = 1.0e-12

    @test ForwardDiff.derivative(y -> FC.rate(spread(0.04, y, fill(10.0, 10))), 0.05) ≈ 1 rtol = 1.0e-10
    # where the solve is damped, the damping decision uses primal values and the root keeps
    # its partials: the spread over the base is target - base
    @test ForwardDiff.derivative(y -> FC.rate(spread(-0.5, y, [1.0], [1.0])), -0.99) ≈ 1 rtol = 1.0e-10
    @test ForwardDiff.derivative(b -> FC.rate(spread(b, -0.99, [1.0], [1.0])), -0.5) ≈ -1 rtol = 1.0e-10
    @test_throws "NaN" spread(0.03, 0.04, [NaN, 1.0])
    @test_throws ErrorException spread(0.03, 0.04, fill(10.0, 10); maxiter = 1)
    base = FM.Yield.Constant(FC.Continuous(0.03))
    @test_throws ErrorException zspread(base, FC.Cashflow(1.0, 2.0), exp(-0.1); maxiter = 1)
end

@testset "spread and zspread do not depend on notional" begin
    # Scaling every cashflow by k scales values and dollar sensitivities by k and leaves the
    # spreads unchanged. Each case is also checked against an independent closed form.
    base = FM.Yield.Constant(FC.Continuous(0.03))
    curve = FM.Yield.Constant(0.03)
    cfs, times = [5.0, 5.0, 105.0], [1.0, 2.0, 3.0]
    stream(k) = FM.Composite(FM.Composite(FC.Cashflow(k * cfs[1], times[1]), FC.Cashflow(k * cfs[2], times[2])), FC.Cashflow(k * cfs[3], times[3]))
    price = sum(c * exp(-0.05 * t) for (c, t) in zip(cfs, times))   # priced at a 5% force
    dv01 = sum(c * t * exp(-0.05 * t) for (c, t) in zip(cfs, times)) / 10_000
    unit_z = zspread(base, stream(1.0), price)
    unit_s = spread(curve, curve + 0.01, cfs, times)
    for k in (1.0e-12, 1.0, 1.0e10)
        # a two-year payment priced at a 5% force is 2% over the 3% base
        z = zspread(base, FC.Cashflow(k, 2.0), k * exp(-0.1))
        @test FC.rate(z.zspread) ≈ 0.02 atol = 1.0e-14
        @test z.zspread_dv01 ≈ k * 2 * exp(-0.1) / 10_000 rtol = 1.0e-12
        z = zspread(base, stream(k), k * price)
        @test FC.rate(z.zspread) ≈ 0.02 atol = 1.0e-14
        @test z.zspread ≈ unit_z.zspread atol = 1.0e-15
        @test z.zspread_dv01 ≈ k * dv01 rtol = 1.0e-12
        @test z.zspread_dv01 ≈ k * unit_z.zspread_dv01 rtol = 1.0e-12
        # `curve + 0.01` adds an annual 1%, so the spread over `curve` is exactly 0.01
        s = spread(curve, curve + 0.01, k .* cfs, times)
        @test FC.rate(s) ≈ 0.01 atol = 1.0e-14
        @test FC.rate(s) ≈ FC.rate(unit_s) atol = 1.0e-15
        @test FC.pv(curve + s, k .* cfs, times) ≈ k * FC.pv(curve + 0.01, cfs, times) rtol = 1.0e-12
        @test FC.rate(spread(0.04, 0.05, k .* fill(10.0, 10))) ≈ 0.01 atol = 1.0e-14
        # zero price, mixed signs: 100 at 1 and -95 at 2 have zero value at a force of log(0.95)
        mixed = FM.Composite(FC.Cashflow(100k, 1.0), FC.Cashflow(-95k, 2.0))
        @test FC.rate(zspread(base, mixed, 0.0).zspread) ≈ log(0.95) - 0.03 atol = 1.0e-14
        @test FC.rate(spread(0.03, -0.05, k .* [100.0, -95.0], [1.0, 2.0])) ≈ -0.08 atol = 1.0e-14
    end
end

@testset "Newton solves keep their own safeguards" begin
    base = FM.Yield.Constant(FC.Continuous(0.03))
    # A payment at time zero does not depend on the spread, so zspread's Newton step is infinite.
    @test_throws r"zspread did not converge \(last Newton step = -?Inf" zspread(base, FC.Cashflow(1.0, 0.0), 0.5)
    # Both solvers report the last Newton step when they run out of iterations.
    @test_throws "spread did not converge in 2 iterations" spread(0.03, 0.04, fill(10.0, 10); maxiter = 2)
    @test_throws "zspread did not converge (last Newton step = " zspread(base, FC.Cashflow(1.0, 2.0), exp(-0.1); maxiter = 1)
    # A start at the root is returned as it is.
    @test FC.rate(spread(0.04, 0.04, [1.0, 2.0])) == 0.0
    @test FC.rate(zspread(base, FC.Cashflow(1.0, 2.0), FC.pv(base, FC.Cashflow(1.0, 2.0)); s0 = 0.0).zspread) ≈ 0.0 atol = 1.0e-15
    # From a base near the edge of its domain, the damped steps reach a distant root:
    # a semiannual base adds the spread nominally, 2((1 + s)^(1/2) - 1) = 0.05 + 1.9.
    @test FC.rate(spread(FC.Periodic(-1.9, 2), FC.Periodic(0.05, 2), [1.0], [1.0])) ≈ 1.975^2 - 1 rtol = 1.0e-12
end

@testset "moic of one-sign and empty streams" begin
    @test moic([-10, 20, 30]) ≈ 5.0
    # an empty sum is zero: a total loss is 0x, no contributions is x/0, nothing at all is 0/0
    @test moic([-10, -20]) === 0.0
    @test moic([10, 20, 30]) === Inf
    @test isnan(moic(Float64[]))
    @test isnan(moic([0, 0]))
    # the amounts' type is kept
    @test moic(Float32[-10, 20]) === 2.0f0
    @test moic(Float32[-10, -20]) === 0.0f0
    @test moic(Float32[10]) === Inf32
    @test moic(FC.Cashflow.([-10.0, -5.0], [0.0, 1.0])) === 0.0
    # no contributions is +Inf, not -Inf from a negated zero, including with floating-point amounts
    @test moic([1.0]) === Inf
    @test moic([-1.0, 0.0]) === 0.0
    @test moic(FC.Cashflow.([10.0], [1.0])) === Inf
    # abstractly typed streams
    @test isnan(moic(Any[]))
    @test moic(Any[-10, 20.0]) === 2.0
    # the sums are type-stable for a concrete element type
    @test @inferred(moic([-10.0, 20.0, 30.0])) === 5.0
    @test @inferred(moic(FC.Cashflow.([-10.0, 20.0], [0.0, 1.0]))) === 2.0
    # Narrow integers: the sums widen before the contributions are negated.
    for T in (Int8, Int16, Int32)
        m = typemin(T)
        @test moic(T[m, -1, 64]) ≈ 64 / (1 - Int(m))
        @test moic(T[m, 64]) ≈ 64 / -Int(m)
        @test moic(FC.Cashflow.(T[m, -1, 64], [0, 1, 2])) ≈ 64 / (1 - Int(m))
    end
end

@testset "duration with a negative-valued valuation function" begin
    liability(i) = -100 / (1 + i)^5
    @test duration(liability, 0.03) ≈ duration(i -> 100 / (1 + i)^5, 0.03)
    @test convexity(liability, 0.03) ≈ convexity(i -> 100 / (1 + i)^5, 0.03)
end

@testset "mismatched cfs/times lengths error loudly" begin
    # the analytic fast paths index times by eachindex(cfs) under @inbounds;
    # a silent mismatch must not read out of bounds (or zip-truncate)
    @test_throws DimensionMismatch duration(0.03, [1.0, 2.0, 3.0], [1.0, 2.0])
    @test_throws DimensionMismatch convexity(0.03, [1.0, 2.0, 3.0], 1:2)
    curve = FM.Yield.Constant(0.03)
    kr = KeyRates([1.0, 2.0, 3.0])
    @test_throws DimensionMismatch sensitivities(kr, curve, [1.0, 2.0, 3.0], [1.0, 2.0])
    # Extra time positions represent zero cashflows and are intentionally harmless.
    @test sensitivities(kr, curve, [1.0, 2.0], [1.0, 2.0, 3.0]) ==
        sensitivities(kr, curve, [1.0, 2.0], [1.0, 2.0])
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
    cfs = [5.0, 5.0, 105.0]
    times = [1.0, 2.0, 3.0]
    # every block of the AD callback's 2×2 parallel Hessian is the combined curve's convexity
    an = convexity(base + credit, cfs, times)
    vf2 = (b, c) -> sum(cf * b(t) * c(t) for (cf, t) in zip(cfs, times))
    ad = convexity(vf2, base, credit)
    @test an ≈ ad.base.base rtol = 1.0e-10
    @test an ≈ ad.credit.credit rtol = 1.0e-10
    @test an ≈ ad.base.credit rtol = 1.0e-10
    @test ad.credit.base == ad.base.credit
end
