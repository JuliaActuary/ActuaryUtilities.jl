@testset "Sensitivity cashflow amounts and embedded times" begin
    tenors = [1.0, 2.0, 5.0]
    kr = KeyRates(tenors)
    times = [0.0, 1.5, 3.5]
    fallback = [7.0, 8.0, 9.0]
    credit = FM.Yield.Constant(FC.Continuous(0.01))
    curves = (
        FM.Yield.Constant(FC.Periodic(0.04, 1)),
        FM.ZeroRateCurve([0.02, 0.03, 0.04], tenors),
        PeriodicZeroSensitivityCurve(0.04),
    )
    for curve in curves, sign in (-1, 1)
        cfs = sign .* [5.0, 5.0, 105.0]
        wrapped = FC.Cashflow.(cfs, times)
        layers = (; base = curve, credit, liquidity = credit)
        calls = (
            (cf, ts) -> duration(curve, cf, ts),
            (cf, ts) -> duration(curve, tenors, cf, ts),
            (cf, ts) -> duration(DV01(), curve, cf, ts),
            (cf, ts) -> duration(DV01(), curve, tenors, cf, ts),
            (cf, ts) -> duration(kr, curve, cf, ts),
            (cf, ts) -> duration(DV01(), kr, curve, cf, ts),
            (cf, ts) -> duration(IR01(), curve, credit, cf, ts),
            (cf, ts) -> duration(CS01(), curve, credit, cf, ts),
            (cf, ts) -> duration(IR01(), curve, credit, tenors, cf, ts),
            (cf, ts) -> duration(CS01(), curve, credit, tenors, cf, ts),
            (cf, ts) -> duration(IR01(), kr, curve, credit, cf, ts),
            (cf, ts) -> duration(CS01(), kr, curve, credit, cf, ts),
            (cf, ts) -> convexity(curve, cf, ts),
            (cf, ts) -> convexity(curve, tenors, cf, ts),
            (cf, ts) -> convexity(kr, curve, cf, ts),
            (cf, ts) -> convexity(curve, credit, tenors, cf, ts),
            (cf, ts) -> convexity(kr, curve, credit, cf, ts),
            (cf, ts) -> convexity(kr, layers, cf, ts),
            (cf, ts) -> sensitivities(kr, curve, cf, ts),
            (cf, ts) -> sensitivities(DV01(), kr, curve, cf, ts),
            (cf, ts) -> sensitivities(kr, curve, credit, cf, ts),
            (cf, ts) -> sensitivities(DV01(), kr, curve, credit, cf, ts),
            (cf, ts) -> sensitivities(kr, layers, cf, ts),
        )
        for f in calls
            expected = f(cfs, times)
            @test _same_sensitivity(f(wrapped, times), expected)
            @test _same_sensitivity(f(wrapped, fallback), expected)
            @test _same_sensitivity(f(wrapped, [fallback; 100.0]), expected)
            @test_throws DimensionMismatch f(wrapped, fallback[1:2])
        end
        # The oracle prices numeric amounts at the embedded dates, independent
        # of the analytic kernel and any cashflow-time normalization helpers.
        value(c) = sum(cfs[k] * FC.discount(c, times[k]) for k in eachindex(cfs))
        @test _same_sensitivity(sensitivities(kr, curve, wrapped, fallback), sensitivities(kr, value, curve))
        @test sum(convexity(kr, curve, wrapped, fallback)) ≈ convexity(curve, value)
        @test duration(kr, curve, wrapped) ≈ duration(kr, curve, wrapped, fallback)
        @test _same_sensitivity(sensitivities(kr, layers, wrapped), sensitivities(kr, layers, wrapped, fallback))
    end
end

@testset "Wrapped sensitivity numeric types and zero streams" begin
    kr = KeyRates([1.0, 2.0, 5.0])
    times = [0.0, 1.5, 3.5]
    fallback = [7.0, 8.0, 9.0]
    flat = FM.Yield.Constant(FC.Continuous(0.04))
    wrapped = FC.Cashflow.([5.0, 5.0, 105.0], times)
    @test (@inferred duration(kr, flat, wrapped, fallback)) ≈ duration(kr, flat, wrapped)
    @test _same_sensitivity((@inferred sensitivities(kr, flat, wrapped, fallback)), sensitivities(kr, flat, wrapped))
    @test (@inferred convexity(flat, kr.tenors, wrapped, fallback)) ≈ convexity(flat, wrapped)
    big_wrapped = FC.Cashflow.(BigFloat[5, 5, 105], big.(times))
    result = sensitivities(kr, flat, big_wrapped, fallback)
    @test result.value isa BigFloat
    @test eltype(result.durations) == eltype(result.convexities) == BigFloat
    @test _same_sensitivity(result, sensitivities(kr, flat, BigFloat[5, 5, 105], big.(times)))

    wrapped_risk(r) = sum(convexity(kr, FM.Yield.Constant(FC.Continuous(r)), wrapped, fallback))
    numeric_risk(r) = sum(convexity(kr, FM.Yield.Constant(FC.Continuous(r)), [5.0, 5.0, 105.0], times))
    @test ForwardDiff.derivative(wrapped_risk, 0.04) ≈ ForwardDiff.derivative(numeric_risk, 0.04)
    @test ForwardDiff.derivative(r -> ForwardDiff.derivative(wrapped_risk, r), 0.04) ≈
        ForwardDiff.derivative(r -> ForwardDiff.derivative(numeric_risk, r), 0.04)
    amount_risk(x) = sum(duration(DV01(), kr, flat, [FC.Cashflow(x, 2.0)], [9.0]))
    @test ForwardDiff.derivative(amount_risk, 0.0) ≈ 2exp(-0.08) / 10_000

    # Numeric elements use fallback times even alongside wrapped elements.
    mixed = Any[wrapped[1], 5.0, wrapped[3]]
    @test _same_sensitivity(sensitivities(kr, flat, mixed, [7.0, 1.5, 9.0]), sensitivities(kr, flat, wrapped))
    zero_curve = ZeroCashflowTestCurve()
    for cfs in (FC.Cashflow{Float64, Float64}[], FC.Cashflow.([0.0, -0.0, 0.0], times))
        @test iszero(convexity(zero_curve, kr.tenors, cfs, fallback))
        @test all(iszero, duration(kr, zero_curve, cfs, fallback))
        @test all(iszero, sensitivities(kr, zero_curve, cfs, fallback).convexities)
        @test_throws ArgumentError convexity(zero_curve, [2.0, 1.0], cfs, fallback)
    end
end

@testset "Derived sensitivity grids use embedded payment times" begin
    curve = FM.Yield.Constant(FC.Continuous(0.04))
    cfs = [5.0, 5.0, 105.0]
    times = [0.0, 1.5, 3.5]
    wrapped = FC.Cashflow.(cfs, times)
    fallback = [0.0, 0.25, 0.5]
    for metric in (KeyRateZero(3.0), KeyRatePar(3.0))
        @test duration(metric, curve, wrapped, fallback) ≈ duration(metric, curve, cfs, times)
        @test duration(metric, curve, wrapped, [7.0, 8.0, 9.0]) ≈ duration(metric, curve, wrapped)
    end
    for metric in (KeyRateZero(0.5), KeyRatePar(0.5))
        @test_throws ArgumentError duration(metric, curve, [FC.Cashflow(100.0, 0.5)], [5.0])
    end

    hw = FM.ShortRate.HullWhite(0.1, 0.01, curve)
    kr = KeyRates([1.0, 2.0, 5.0])
    for args in ((kr, hw), (DV01(), kr, hw)), supplied in (fallback, [7.0, 8.0, 9.0, 100.0])
        rng_numeric, rng_wrapped = MersenneTwister(42), MersenneTwister(42)
        numeric = sensitivities(args..., cfs, times; n_scenarios = 8, timestep = 0.5, rng = rng_numeric)
        actual = sensitivities(args..., wrapped, supplied; n_scenarios = 8, timestep = 0.5, rng = rng_wrapped)
        @test _same_sensitivity(actual, numeric)
        @test rand(rng_numeric) == rand(rng_wrapped)
    end
end
