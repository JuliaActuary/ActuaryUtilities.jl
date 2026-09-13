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
        @test duration(DV01(), yield, callback) ≈ expected
        @test duration(DV01(), yield, value) ≈ expected
        @test duration(IR01(), yield, yield, cfs, times) ≈ expected
        @test duration(CS01(), yield, yield, cfs, times) ≈ expected
        @test !isfinite(duration(yield, callback))
        @test !isfinite(convexity(yield, callback))
        if yield isa FM.Yield.AbstractYieldModel
            @test sum(duration(DV01(), kr, yield, cfs, times)) ≈ expected
            @test sum(duration(DV01(), kr, callback, yield)) ≈ expected
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
        @test duration(DV01(), yield, counted) ≈ expected
        @test calls[] == 1
        @test duration(DV01(), yield, sign .* cfs, times) ≈ expected
        for constant_value in (0.0, 3.0, big"0.0", big"3.0")
            @test iszero(duration(DV01(), yield, _ -> constant_value))
        end
    end

    r = big"0.04"
    curve = FM.Yield.Constant(FC.Continuous(r))
    big_dv01 = duration(DV01(), curve, c -> big"100.0" * FC.discount(c, 2))
    @test big_dv01 isa BigFloat
    @test big_dv01 ≈ 200exp(-2r) / 10_000
    curve_dv01(r) = duration(DV01(), FM.Yield.Constant(FC.Continuous(r)), c -> 100FC.discount(c, 2.0))
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
        port = [FC.Cashflow(sign * A, 1.0), FC.Cashflow(-sign * B, 5.0)]
        @test abs(FC.present_value(curve, port)) < 1.0e-10
        expected = duration(DV01(), curve, tenors, sign .* [A, -B], [1.0, 5.0])
        @test abs(expected) > 1.0e-3
        r = sensitivities(port, curve, tenors)
        @test r.effective_dv01 ≈ expected
        @test r.effective_dv01 ≈ dv01(Effective(), port, curve, tenors)
        @test r.spread_dv01 ≈ dv01(Spread(), port, curve, tenors)
        @test r.forward_dv01 ≈ 0 atol = 1.0e-12
        @test !isfinite(r.effective_duration)
        @test !isfinite(r.spread_duration)
        # Two-curve form: the discount role carries the whole exposure for fixed cashflows.
        r2 = sensitivities(port, curve, credit, tenors)
        @test r2.spread_dv01 ≈ dv01(Spread(), port, curve, credit, tenors)
        @test r2.effective_dv01 ≈ dv01(Effective(), port, curve, credit, tenors)
        @test r2.forward_dv01 ≈ 0 atol = 1.0e-12
    end
    # Nonzero-value positions are unchanged.
    fb = FM.Bond.Fixed(0.05, FC.Periodic(1), 3.0)
    r = sensitivities(fb, curve, tenors)
    @test r.effective_dv01 ≈ dv01(Effective(), fb, curve, tenors)
    @test r.spread_dv01 ≈ dv01(Spread(), fb, curve, tenors)
    @test r.effective_dv01 ≈ r.effective_duration * r.value / 10_000
end
