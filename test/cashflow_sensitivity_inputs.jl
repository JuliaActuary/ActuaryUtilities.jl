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
        layered = curve + credit
        calls = (
            (cf, ts) -> duration(curve, cf, ts),
            (cf, ts) -> duration(DV01(), curve, cf, ts),
            (cf, ts) -> duration(kr, curve, cf, ts),
            (cf, ts) -> duration(DV01(), kr, curve, cf, ts),
            (cf, ts) -> duration(DV01(), layered, cf, ts),
            (cf, ts) -> duration(DV01(), kr, layered, cf, ts),
            (cf, ts) -> convexity(curve, cf, ts),
            (cf, ts) -> convexity(kr, curve, cf, ts),
            (cf, ts) -> convexity(layered, cf, ts),
            (cf, ts) -> convexity(kr, layered, cf, ts),
            (cf, ts) -> sensitivities(kr, curve, cf, ts),
            (cf, ts) -> sensitivities(SecondOrder(), kr, curve, cf, ts),
            (cf, ts) -> sensitivities(SecondOrder(), curve, cf, ts),
            (cf, ts) -> sensitivities(kr, layered, cf, ts),
            (cf, ts) -> sensitivities(SecondOrder(), kr, layered, cf, ts),
        )
        for f in calls
            expected = f(cfs, times)
            @test _same_sensitivity(f(wrapped, times), expected)
            @test _same_sensitivity(f(wrapped, fallback), expected)
            @test _same_sensitivity(f(wrapped, [fallback; 100.0]), expected)
            @test_throws DimensionMismatch f(wrapped, fallback[1:2])
        end
        # Price at embedded dates without using the normalization helpers.
        value(c) = sum(cfs[k] * FC.discount(c, times[k]) for k in eachindex(cfs))
        @test _same_sensitivity(sensitivities(SecondOrder(), kr, curve, wrapped, fallback), sensitivities(value, SecondOrder(), kr, curve))
        @test _same_sensitivity(sensitivities(SecondOrder(), curve, wrapped, fallback), sensitivities(value, SecondOrder(), curve))
        @test sum(convexity(kr, curve, wrapped, fallback)) ≈ convexity(value, curve)
        @test duration(kr, curve, wrapped) ≈ duration(kr, curve, wrapped, fallback)
        @test _same_sensitivity(sensitivities(kr, layered, wrapped), sensitivities(kr, layered, wrapped, fallback))
    end
end

@testset "Key-rate cashflow forms default times to periods" begin
    tenors = [1.0, 2.0, 5.0]
    kr = KeyRates(tenors)
    curve = FM.ZeroRateCurve([0.02, 0.03, 0.04], tenors)
    credit = FM.Yield.Constant(FC.Continuous(0.01))
    layered = curve + credit
    cfs = [5.0, 5.0, 105.0]
    wrapped = FC.Cashflow.(cfs, [0.5, 1.5, 3.5])
    calls = (
        (x...) -> duration(kr, curve, x...),
        (x...) -> duration(DV01(), kr, curve, x...),
        (x...) -> duration(DV01(), kr, layered, x...),
        (x...) -> convexity(kr, curve, x...),
        (x...) -> convexity(layered, x...),
        (x...) -> convexity(kr, layered, x...),
        (x...) -> sensitivities(kr, curve, x...),
        (x...) -> sensitivities(SecondOrder(), kr, curve, x...),
        (x...) -> sensitivities(SecondOrder(), curve, x...),
        (x...) -> sensitivities(kr, layered, x...),
        (x...) -> sensitivities(SecondOrder(), kr, layered, x...),
    )
    for f in calls
        # Numeric amounts are paid at periods 1:n, as in the scalar measures.
        @test isequal(f(cfs), f(cfs, 1:3))
        # Wrapped cashflows use their embedded times whatever times are supplied.
        @test isequal(f(wrapped), f(wrapped, [9.0, 9.0, 9.0]))
    end
end

@testset "Wrapped sensitivity numeric types and zero streams" begin
    kr = KeyRates([1.0, 2.0, 5.0])
    times = [0.0, 1.5, 3.5]
    fallback = [7.0, 8.0, 9.0]
    flat = FM.Yield.Constant(FC.Continuous(0.04))
    wrapped = FC.Cashflow.([5.0, 5.0, 105.0], times)
    @test (@inferred duration(kr, flat, wrapped, fallback)) ≈ duration(kr, flat, wrapped)
    @test _same_sensitivity((@inferred sensitivities(SecondOrder(), kr, flat, wrapped, fallback)), sensitivities(SecondOrder(), kr, flat, wrapped))
    @test _same_sensitivity((@inferred sensitivities(SecondOrder(), flat, wrapped, fallback)), sensitivities(SecondOrder(), flat, wrapped))
    @test (@inferred convexity(flat, wrapped, fallback)) ≈ convexity(flat, wrapped)
    big_wrapped = FC.Cashflow.(BigFloat[5, 5, 105], big.(times))
    result = sensitivities(SecondOrder(), kr, flat, big_wrapped, fallback)
    @test result.value isa BigFloat
    @test eltype(result.duration) == eltype(result.dv01) == eltype(result.convexity) == BigFloat
    @test _same_sensitivity(result, sensitivities(SecondOrder(), kr, flat, BigFloat[5, 5, 105], big.(times)))

    wrapped_risk(r) = sum(convexity(kr, FM.Yield.Constant(FC.Continuous(r)), wrapped, fallback))
    numeric_risk(r) = sum(convexity(kr, FM.Yield.Constant(FC.Continuous(r)), [5.0, 5.0, 105.0], times))
    @test ForwardDiff.derivative(wrapped_risk, 0.04) ≈ ForwardDiff.derivative(numeric_risk, 0.04)
    @test ForwardDiff.derivative(r -> ForwardDiff.derivative(wrapped_risk, r), 0.04) ≈
        ForwardDiff.derivative(r -> ForwardDiff.derivative(numeric_risk, r), 0.04)
    amount_risk(x) = sum(duration(DV01(), kr, flat, [FC.Cashflow(x, 2.0)], [9.0]))
    @test ForwardDiff.derivative(amount_risk, 0.0) ≈ 2exp(-0.08) / 10_000

    # Numeric elements use fallback times even alongside wrapped elements.
    mixed = Any[wrapped[1], 5.0, wrapped[3]]
    @test _same_sensitivity(sensitivities(SecondOrder(), kr, flat, mixed, [7.0, 1.5, 9.0]), sensitivities(SecondOrder(), kr, flat, wrapped))
    # Zero and empty wrapped streams take their time type from the embedded times, as valuation
    # does, not from the supplied times it ignores.
    curve32 = FM.Yield.Constant(FC.Continuous(0.04f0))
    kr32 = KeyRates(Float32[1, 2])
    supplied = Float64[9, 10]
    nonzero32 = FC.Cashflow.(Float32[1, 2], Float32[1, 2])
    for cfs in (FC.Cashflow.(Float32[0, 0], Float32[1, 2]), FC.Cashflow{Float32, Float32}[])
        for measure in (Macaulay(), Modified())
            @test typeof(duration(measure, curve32, cfs, supplied)) == typeof(duration(measure, curve32, nonzero32, supplied)) == Float32
        end
        @test typeof(convexity(curve32, cfs, supplied)) == typeof(convexity(curve32, nonzero32, supplied)) == Float32
        zero_bundle = sensitivities(SecondOrder(), kr32, curve32, cfs, supplied)
        bundle = sensitivities(SecondOrder(), kr32, curve32, nonzero32, supplied)
        @test typeof(zero_bundle.value) == typeof(bundle.value) == Float32
        @test eltype(zero_bundle.duration) == eltype(bundle.duration) == Float32
        @test eltype(zero_bundle.convexity) == eltype(bundle.convexity) == Float32
        @test typeof(sensitivities(SecondOrder(), curve32, cfs, supplied)) == typeof(sensitivities(SecondOrder(), curve32, nonzero32, supplied))
    end
    # Abstractly typed streams take the time type from the times their payments use: a
    # `Cashflow`'s own, a number's supplied one. Float32 amounts at BigFloat times value in BigFloat.
    nonzero_big = FC.Cashflow[FC.Cashflow(1.0f0, big(1.0)), FC.Cashflow(2.0f0, big(2.0))]
    zeros_big = FC.Cashflow[FC.Cashflow(0.0f0, big(1.0)), FC.Cashflow(0.0f0, big(2.0))]
    for supplied_times in (supplied, [9, 10])
        for measure in (Macaulay(), Modified())
            @test typeof(duration(measure, curve32, zeros_big, supplied_times)) ==
                typeof(duration(measure, curve32, nonzero_big, supplied_times)) == BigFloat
        end
        @test typeof(convexity(curve32, zeros_big, supplied_times)) ==
            typeof(convexity(curve32, nonzero_big, supplied_times)) == BigFloat
        @test typeof(duration(DV01(), curve32, Any[zeros_big...], supplied_times)) ==
            typeof(duration(DV01(), curve32, Any[nonzero_big...], supplied_times))
        # a number in the same collection is paid at its supplied time
        @test typeof(duration(Macaulay(), curve32, Any[0.0f0, zeros_big[2]], supplied_times)) ==
            typeof(duration(Macaulay(), curve32, Any[1.0f0, nonzero_big[2]], supplied_times))
    end
    zb = sensitivities(kr32, curve32, zeros_big, supplied)
    nb = sensitivities(kr32, curve32, nonzero_big, supplied)
    @test typeof(zb.value) == typeof(nb.value) == BigFloat
    @test eltype(zb.duration) == eltype(nb.duration)

    zero_curve = ZeroCashflowTestCurve()
    for cfs in (FC.Cashflow{Float64, Float64}[], FC.Cashflow.([0.0, -0.0, 0.0], times))
        @test iszero(convexity(zero_curve, cfs, fallback))
        @test all(iszero, duration(kr, zero_curve, cfs, fallback))
        @test all(iszero, sensitivities(SecondOrder(), kr, zero_curve, cfs, fallback).convexity)
    end
end

@testset "Scenarios value wrapped cashflows at their embedded times" begin
    curve = FM.Yield.Constant(FC.Continuous(0.04))
    cfs = [5.0, 5.0, 105.0]
    times = [0.0, 1.5, 3.5]
    wrapped = FC.Cashflow.(cfs, times)
    fallback = [0.0, 0.25, 0.5]
    hw = FM.ShortRate.HullWhite(0.1, 0.01, curve)
    kr = KeyRates([1.0, 2.0, 5.0])
    for m in ((), (SecondOrder(),)), supplied in (fallback, [7.0, 8.0, 9.0, 100.0])
        rng_numeric, rng_wrapped = MersenneTwister(42), MersenneTwister(42)
        numeric = sensitivities(m..., kr, Scenarios(hw; n_scenarios = 8, timestep = 0.5, horizon = 4.0, rng = rng_numeric), cfs, times)
        actual = sensitivities(m..., kr, Scenarios(hw; n_scenarios = 8, timestep = 0.5, horizon = 4.0, rng = rng_wrapped), wrapped, supplied)
        @test _same_sensitivity(actual, numeric)
        @test rand(rng_numeric) == rand(rng_wrapped)
    end
end
