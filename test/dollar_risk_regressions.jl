@testset "Zero-value positions retain scalar dollar risk" begin
    times = [0.0, 1.0]
    kr = KeyRates([1.0, 2.0])
    yields = (
        0.0, FC.Periodic(0.0, 1), FC.Periodic(0.0, 2), FC.Continuous(0.0),
        FM.Yield.Constant(FC.Continuous(0.0)),
        FM.Yield.Constant(FC.Periodic(0.0, 1)),
        FM.ZeroRateCurve([0.0, 0.0], kr.tenors),
        PeriodicZeroSensitivityCurve(0.0),
    )
    for yield in yields, sign in (-1, 1)
        cfs = sign .* [-1.0, 1.0]
        value = CashflowValue(cfs, times)
        callback(c) = value(c)
        # At a zero rate, P(s) = -1 + exp(-s) for continuous shocks,
        # or -1 + (1+s/m)^(-m) for periodic shocks. Both give P'(0) = -1.
        expected = sign / 10_000
        @test iszero(value(yield))
        @test duration(DV01(), yield, cfs, times) ≈ expected
        @test duration(DV01(), yield, FC.Cashflow.(cfs, times)) ≈ expected
        @test duration(callback, DV01(), yield) ≈ expected
        @test duration(value, DV01(), yield) ≈ expected
        @test duration(DV01(), yield + yield, cfs, times) ≈ expected
        @test !isfinite(duration(callback, yield))
        @test !isfinite(convexity(callback, yield))
        # The dollar convexity is P''(0) = 1 + 1/m: 2 for an annual rate, 1.5 for Periodic(m = 2),
        # 1 for continuous shocks; normalized convexity is undefined here.
        inv_m = yield isa Real ? 1 : yield isa FC.Rate{<:Real, <:FC.Periodic} ? 1 / yield.compounding.frequency : 0
        expected_h = sign * (1 + inv_m)
        @test convexity(DollarConvexity(), yield, cfs, times) ≈ expected_h
        @test convexity(DollarConvexity(), yield, FC.Cashflow.(cfs, times)) ≈ expected_h
        @test convexity(callback, DollarConvexity(), yield) ≈ expected_h
        @test convexity(value, DollarConvexity(), yield) ≈ expected_h
        for s in (sensitivities(SecondOrder(), yield, cfs, times), sensitivities(callback, SecondOrder(), yield))
            @test iszero(s.value) && s.dv01 ≈ expected && s.dollar_convexity ≈ expected_h
        end
        if yield isa FM.Yield.AbstractYieldModel
            @test sum(duration(DV01(), kr, yield, cfs, times)) ≈ expected
            @test sum(duration(callback, DV01(), kr, yield)) ≈ expected
            @test sum(convexity(DollarConvexity(), kr, yield, cfs, times)) ≈ expected_h
            @test sum(convexity(callback, DollarConvexity(), kr, yield)) ≈ expected_h
        end
    end
end

@testset "Scalar dollar derivatives preserve coordinates and AD types" begin
    cfs = [5.0, 5.0, 105.0]
    times = [0.5, 1.5, 3.0]
    y = 0.04
    # Independent closed-form derivatives in each input's own coordinate.
    for (yield, discounted, divisor) in (
                (y, cfs ./ (1 + y) .^ times, 1 + y),
                (FC.Periodic(y, 2), cfs ./ (1 + y / 2) .^ (2 .* times), 1 + y / 2),
                (FC.Continuous(y), cfs .* exp.(-y .* times), 1.0),
                (FM.Yield.Constant(FC.Periodic(y, 1)), cfs ./ (1 + y) .^ times, 1.0),
            ), sign in (-1, 1)
        expected = sign * sum(times .* discounted) / divisor / 10_000
        calls = Ref(0)
        counted(c) = begin
            calls[] += 1
            FC.pv(c, sign .* cfs, times)
        end
        @test duration(counted, DV01(), yield) ≈ expected
        @test calls[] == 1
        @test duration(DV01(), yield, sign .* cfs, times) ≈ expected
        for constant_value in (0.0, 3.0, big"0.0", big"3.0")
            @test iszero(duration(_ -> constant_value, DV01(), yield))
        end
    end

    r = big"0.04"
    curve = FM.Yield.Constant(FC.Continuous(r))
    big_dv01 = duration(c -> big"100.0" * FC.discount(c, 2), DV01(), curve)
    @test big_dv01 isa BigFloat
    @test big_dv01 ≈ 200exp(-2r) / 10_000
    curve_dv01(r) = duration(c -> 100FC.discount(c, 2.0), DV01(), FM.Yield.Constant(FC.Continuous(r)))
    @test ForwardDiff.derivative(curve_dv01, 0.04) ≈ -400exp(-0.08) / 10_000
    @test ForwardDiff.derivative(r -> ForwardDiff.derivative(curve_dv01, r), 0.04) ≈ 800exp(-0.08) / 10_000

    flat = FM.Yield.Constant(FC.Continuous(0.04))
    # A zero primal cashflow with nonzero AD partials must still carry dollar risk.
    amount_dv01(x) = duration(DV01(), flat, [x], [2.0])
    @test ForwardDiff.derivative(amount_dv01, 0.0) ≈ 2exp(-0.08) / 10_000
    @test ForwardDiff.derivative(x -> ForwardDiff.derivative(y -> amount_dv01(y^2), x), 0.0) ≈ 4exp(-0.08) / 10_000
end

@testset "Zero-value contract bundles retain dollar risk" begin
    curve = FM.Yield.Constant(FC.Continuous(0.04))
    credit = FM.Yield.Constant(FC.Continuous(0.05))
    tenors = [1.0, 2.0, 5.0, 10.0]
    A = 100.0
    B = A * FC.discount(curve, 1.0) / FC.discount(curve, 5.0)
    for sign in (-1, 1)
        # A vector of contracts, so a portfolio; a vector of `Cashflow`s is fixed cashflows.
        port = FC.AbstractContract[FC.Cashflow(sign * A, 1.0), FC.Cashflow(-sign * B, 5.0)]
        fixed = FC.Cashflow.([sign * A, -sign * B], [1.0, 5.0])
        @test abs(FC.present_value(curve, port)) < 1.0e-10
        expected = duration(DV01(), curve, sign .* [A, -B], [1.0, 5.0])
        expected_h = convexity(DollarConvexity(), curve, sign .* [A, -B], [1.0, 5.0])
        @test abs(expected) > 1.0e-3 && abs(expected_h) > 1
        # A zero-value position with dollar risk: dollar measures are defined, normalized ones not.
        for order in (FirstOrder(), SecondOrder())
            r = sensitivities(order, curve, port)
            k = sensitivities(order, KeyRates(tenors), curve, port)
            @test abs(r.value) < 1.0e-10
            @test r.dv01.discount + r.dv01.index ≈ expected
            @test sum(k.dv01.discount .+ k.dv01.index) ≈ expected
            @test r.dv01.index ≈ 0 atol = 1.0e-12
            @test !isfinite(r.duration.discount) && !isfinite(r.duration.discount + r.duration.index)
            if order isa SecondOrder
                @test !isfinite(r.convexity.discount.discount)
                h = r.dollar_convexity
                @test h.discount.discount + h.discount.index + h.index.discount + h.index.index ≈ expected_h
                @test sum(k.dollar_convexity.discount.discount) ≈ expected_h
            end
        end
        @test convexity(DollarConvexity(), curve, port) ≈ expected_h
        @test convexity(DollarConvexity(), Spread(), curve, port) ≈ expected_h
        @test sum(convexity(DollarConvexity(), KeyRates(tenors), curve, port)) ≈ expected_h
        @test duration(DV01(), Effective(), curve, port) ≈ expected
        @test duration(DV01(), curve, port) ≈ expected
        @test duration(DV01(), curve, port) ≈ expected
        @test duration(DV01(), Spread(), curve, port) ≈ sensitivities(curve, port).dv01.discount
        @test sensitivities(curve, fixed).dv01 ≈ expected
        @test sum(sensitivities(KeyRates(tenors), curve, fixed).dv01) ≈ expected
        # Two-curve form: the discount role carries the whole exposure for fixed cashflows.
        r2 = sensitivities(credit, port; index = curve)
        @test r2.dv01.discount ≈ duration(DV01(), Spread(), credit, port; index = curve)
        @test r2.dv01.discount + r2.dv01.index ≈ duration(DV01(), Effective(), credit, port; index = curve)
        @test r2.dv01.index ≈ 0 atol = 1.0e-12
    end
    # Nonzero-value positions are unchanged.
    fb = FM.Bond.Fixed(0.05, FC.Periodic(1), 3.0)
    r = sensitivities(curve, fb)
    @test r.dv01.discount + r.dv01.index ≈ duration(DV01(), Effective(), curve, fb)
    @test r.dv01.discount ≈ duration(DV01(), Spread(), curve, fb)
    @test r.dv01.discount ≈ r.duration.discount * r.value / 10_000
end

@testset "Dollar convexity is the raw second derivative" begin
    tenors = [1.0, 2.0, 5.0, 10.0]
    kr = KeyRates(tenors)
    curve = FM.ZeroRateCurve([0.03, 0.035, 0.04, 0.045], tenors)
    credit = FM.Yield.Constant(FC.Continuous(0.01))
    a_cfs, a_times = [5.0, 5.0, 105.0], [1.0, 2.0, 3.0]
    b_cfs, b_times = [-3.0, -103.0], [4.0, 7.0]
    H(x...; kw...) = convexity(DollarConvexity(), x...; kw...)

    # At nonzero value it is value times convexity, in each input's own coordinate; it adds across
    # positions and reverses with them.
    for yield in (0.04, FC.Periodic(0.04, 2), FC.Continuous(0.04), curve)
        V = FC.pv(yield, a_cfs, a_times)
        @test H(yield, a_cfs, a_times) ≈ V * convexity(yield, a_cfs, a_times) rtol = 1.0e-12
        @test convexity(c -> FC.pv(c, a_cfs, a_times), DollarConvexity(), yield) ≈ H(yield, a_cfs, a_times) rtol = 1.0e-10
        s = sensitivities(SecondOrder(), yield, a_cfs, a_times)
        @test s.dollar_convexity ≈ s.value * s.convexity rtol = 1.0e-12
        @test H(yield, [a_cfs; b_cfs], [a_times; b_times]) ≈ H(yield, a_cfs, a_times) + H(yield, b_cfs, b_times) rtol = 1.0e-12
        @test H(yield, -a_cfs, a_times) == -H(yield, a_cfs, a_times)
    end

    # Key-rate matrices sum to the parallel number, and the bundle returns the same derivatives.
    M = H(kr, curve, a_cfs, a_times)
    @test M isa Matrix{Float64}
    @test sum(M) ≈ H(curve, a_cfs, a_times) rtol = 1.0e-10
    s = sensitivities(SecondOrder(), kr, curve, a_cfs, a_times)
    @test s.dollar_convexity == M
    @test s.dollar_convexity ≈ s.value .* s.convexity rtol = 1.0e-12
    @test convexity(c -> FC.pv(c, a_cfs, a_times), DollarConvexity(), kr, curve) ≈ M rtol = 1.0e-10
    # Fresh storage: the raw and normalized fields, and separate calls, share no elements.
    s.dollar_convexity[1, 1] += 1
    @test M == H(kr, curve, a_cfs, a_times)
    @test s.convexity == sensitivities(SecondOrder(), kr, curve, a_cfs, a_times).convexity

    # Two curves that play different roles: the cross blocks are each other's transposes, not
    # symmetric, and the parallel blocks are the sums of the key-rate ones.
    floater(b, c) = sum(1:5) do t
        coupon = 100 * (1 / FC.discount(b, t - 1, t) - 1 + 0.015)
        (coupon + (t == 5 ? 100 : 0)) * FC.discount(b + c, t)
    end
    blocks = convexity(floater, DollarConvexity(), kr, curve, credit)
    @test blocks.credit.base ≈ transpose(blocks.base.credit)
    @test !(blocks.base.credit ≈ transpose(blocks.base.credit))
    @test _same_sensitivity(blocks, sensitivities(floater, SecondOrder(), kr, curve, credit).dollar_convexity)
    @test blocks.base.credit ≈ floater(curve, credit) .* convexity(floater, kr, curve, credit).base.credit rtol = 1.0e-10
    parallel = convexity(floater, DollarConvexity(), curve, credit)
    @test parallel.base.credit == parallel.credit.base
    @test parallel.base.credit ≈ sum(blocks.base.credit) rtol = 1.0e-8
    @test parallel.credit.credit ≈ sum(blocks.credit.credit) rtol = 1.0e-8

    # Contracts: effective dollar convexity includes all four role blocks; spread is the discount block.
    bond = FM.Bond.Floating(0.01, FC.Periodic(1), 5.0, "OIS")
    r = sensitivities(SecondOrder(), credit, bond; index = curve)
    h = r.dollar_convexity
    four = h.discount.discount + h.discount.index + h.index.discount + h.index.index
    @test H(credit, bond; index = curve) ≈ four rtol = 1.0e-8
    @test H(Effective(), credit, bond; index = curve) == H(credit, bond; index = curve)
    @test H(Spread(), credit, bond; index = curve) ≈ h.discount.discount rtol = 1.0e-8
    @test h.discount.index ≈ r.value * r.convexity.discount.index rtol = 1.0e-12
    k = sensitivities(SecondOrder(), kr, credit, bond; index = curve).dollar_convexity
    @test H(Effective(), kr, credit, bond; index = curve) ≈ k.discount.discount + k.discount.index + k.index.discount + k.index.index rtol = 1.0e-10
    @test H(kr, credit, bond; index = curve) == H(Effective(), kr, credit, bond; index = curve)
    @test H(Spread(), kr, credit, bond; index = curve) ≈ k.discount.discount rtol = 1.0e-10
    @test sum(H(kr, credit, bond; index = curve)) ≈ four rtol = 1.0e-8

    # Named curves, market inputs and Scenarios get it through their second-order bundles.
    named = sensitivities(c -> FC.pv(c.rf + c.spread, a_cfs, a_times), SecondOrder(), (; rf = curve, spread = credit))
    @test named.dollar_convexity.rf.spread ≈ named.value * named.convexity.rf.spread rtol = 1.0e-12
    quotes = sensitivities(m -> FC.pv(FM.ZeroRateCurve(m.z, tenors, FM.Spline.Linear()), a_cfs, a_times), SecondOrder(), (; z = [0.03, 0.035, 0.04, 0.045]))
    @test quotes.dollar_convexity.z.z ≈ quotes.value .* quotes.convexity.z.z rtol = 1.0e-12
    hw = FM.ShortRate.HullWhite(0.1, 0.01, curve)
    paths = Scenarios(hw; horizon = 3.0, timestep = 0.25, n_scenarios = 16, rng = Random.Xoshiro(1))
    m = sensitivities(SecondOrder(), paths, a_cfs, a_times)
    @test m.dollar_convexity ≈ m.value * m.convexity rtol = 1.0e-12

    # Numeric types, nested AD, embedded payment times, offset inputs and zero streams.
    @test H(0.04f0, Float32.(a_cfs), Float32.(a_times)) isa Float32
    big_h = H(FC.Continuous(big"0.04"), big.(a_cfs), big.(a_times))
    @test big_h isa BigFloat
    @test big_h ≈ H(FC.Continuous(0.04), a_cfs, a_times) rtol = 1.0e-12
    @test ForwardDiff.derivative(x -> H(FC.Continuous(0.04), [x], [2.0]), 100.0) ≈ 4exp(-0.08)
    @test ForwardDiff.derivative(r -> H(FC.Continuous(r), [100.0], [2.0]), 0.04) ≈ -800exp(-0.08)
    @test H(curve, [FC.Cashflow(100.0, 2.0)], [10.0]) == H(curve, [100.0], [2.0])
    @test H(curve, OffsetArray(a_cfs, 0:2), OffsetArray(a_times, -3:-1)) == H(curve, a_cfs, a_times)
    @test H(kr, curve, OffsetArray(a_cfs, 0:2), a_times) == M
    @test H(curve, Float64[], Float64[]) === 0.0
    @test H(kr, curve, zeros(3), a_times) == zeros(4, 4)
    @test (@inferred H(curve, a_cfs, a_times)) isa Float64
    @test (@inferred H(kr, curve, a_cfs, a_times)) isa Matrix{Float64}
    # First-order results are unchanged.
    @test keys(sensitivities(curve, a_cfs, a_times)) == (:value, :duration, :dv01)
    @test keys(sensitivities(SecondOrder(), curve, a_cfs, a_times)) == (:value, :duration, :dv01, :convexity, :dollar_convexity)
end
