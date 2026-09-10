@testset "Scalar cashflow collection boundaries" begin
    times = [1.0, 2.0, 3.0]
    curve = FM.Yield.Constant(FC.Continuous(0.04))
    collections(cfs) = (
        () -> Tuple(cfs),
        () -> (c for c in cfs),
        () -> (c for c in Iterators.Stateful(cfs)),
        () -> (c for c in cfs if true),
        () -> reshape(copy(cfs), 1, length(cfs)),
    )
    for cfs in ([5.0, 5.0, 105.0], [-5.0, -5.0, -105.0], zeros(3), Float64[])
        for make in collections(cfs), yield in (0.04, FC.Periodic(0.04, 2), curve)
            for metric in (Macaulay(), Modified(), DV01())
                @test duration(metric, yield, make(), times) ≈ duration(metric, yield, cfs, times)
                @test duration(metric, yield, make()) ≈ duration(metric, yield, cfs)
            end
            @test duration(yield, make(), times) ≈ duration(yield, cfs, times)
            @test duration(yield, make()) ≈ duration(yield, cfs)
            @test convexity(yield, make(), times) ≈ convexity(yield, cfs, times)
            @test convexity(yield, make()) ≈ convexity(yield, cfs)
            for metric in (IR01(), CS01())
                @test duration(metric, yield, yield, make(), times) ≈ duration(metric, yield, yield, cfs, times)
                @test duration(metric, yield, yield, make()) ≈ duration(metric, yield, yield, cfs)
            end
        end
        for make in collections(cfs), metric in (KeyRateZero(1), KeyRatePar(1))
            @test duration(metric, curve, make(), times) ≈ duration(metric, curve, cfs, times)
            @test duration(metric, curve, make()) ≈ duration(metric, curve, cfs)
        end
    end
    wrapped = FC.Cashflow.([5.0, 5.0, 105.0], [0.5, 1.5, 2.5])
    for make in collections(wrapped), metric in (Macaulay(), Modified(), DV01())
        @test duration(metric, curve, make()) ≈ duration(metric, curve, wrapped)
    end
    @test_throws ArgumentError duration([5.0, 5.0, 105.0], curve, times)
end

@testset "Continuous shocks on periodic-zero user curves" begin
    curve = PeriodicZeroSensitivityCurve(0.04)
    tenors = [1.0, 2.0, 3.0]
    kr = KeyRates(tenors)
    cfs = [5.0, 5.0, 105.0]
    discounted = cfs .* FC.discount.(Ref(curve), tenors)
    value = sum(discounted)
    expected_duration = sum(tenors .* discounted) / value
    expected_convexity = sum(tenors .^ 2 .* discounted) / value
    valuation(c) = FC.pv(c, cfs, tenors)
    for sign in (-1, 1)
        amounts = sign .* cfs
        vf(c) = sign * valuation(c)
        @test duration(curve, amounts, tenors) ≈ expected_duration
        @test duration(curve, vf) ≈ expected_duration
        @test duration(vf, curve, tenors) ≈ expected_duration
        @test sum(duration(kr, vf, curve)) ≈ expected_duration
        @test duration(kr, vf, curve) ≈ duration(kr, curve, amounts, tenors)
        @test convexity(curve, amounts, tenors) ≈ expected_convexity
        @test convexity(curve, tenors, amounts, tenors) ≈ expected_convexity
        @test sum(convexity(kr, vf, curve)) ≈ expected_convexity
        @test convexity(kr, vf, curve) ≈ convexity(kr, curve, amounts, tenors)
        @test duration(DV01(), curve, amounts, tenors) ≈ sign * value * expected_duration / 10_000
        @test sum(duration(DV01(), kr, vf, curve)) ≈ sign * value * expected_duration / 10_000
    end
    # Legacy zero-rate bumps share the same coordinate; central differences
    # approximate the analytic duration to second order in the bump size.
    legacy = sum(duration(KeyRateZero(t, 1.0e-5), curve, cfs, tenors, tenors) for t in tenors)
    @test legacy ≈ expected_duration rtol = 1.0e-8

    bond = FM.Bond.Fixed(0.05, FC.Periodic(1), 3.0)
    reference = FM.Yield.Constant(FC.Periodic(0.04, 1))
    @test duration(Effective(), bond, curve, tenors) ≈ duration(Effective(), bond, reference, tenors)
    @test duration(Spread(), bond, curve, tenors) ≈ duration(Spread(), bond, reference, tenors)
    spread = 0.012
    market_price = FC.pv(FM.Yield.Constant(FC.Continuous(log1p(0.04) + spread)), bond)
    result = zspread(bond, curve, market_price)
    expected = zspread(bond, reference, market_price)
    @test result.zspread ≈ spread atol = 1.0e-10
    @test result.zspread_dv01 ≈ expected.zspread_dv01
end

@testset "Named cashflow outputs own their arrays" begin
    kr = KeyRates([1.0, 2.0, 3.0])
    curve = FM.Yield.Constant(FC.Continuous(0.04))
    curves = (; base = curve, credit = curve, liquidity = curve)
    for cfs in ([5.0, 5.0, 105.0], zeros(3), Float64[])
        result = sensitivities(kr, curves, cfs, [1.0, 2.0, 3.0])
        before = copy(result.durations.credit)
        result.durations.base[1] = 123.0
        @test result.durations.credit == before
        @test result.durations.liquidity == before
        for blocks in (result.convexities, convexity(kr, curves, cfs, [1.0, 2.0, 3.0]))
            matrices = [block for row in values(blocks) for block in values(row)]
            @test all(matrices[i] !== matrices[j] for i in eachindex(matrices) for j in eachindex(matrices) if i != j)
            untouched = copy(last(matrices))
            first(matrices)[1, 1] = 456.0
            @test all(block == untouched for block in matrices[2:end])
        end
    end
end

@testset "Scalar convexity validates every supplied tenor grid" begin
    cfs = [5.0, 5.0, 105.0]
    times = [1.0, 2.0, 3.0]
    wrapped = FC.Cashflow.(cfs, times)
    bond = FM.Bond.Fixed(0.05, FC.Periodic(1), 3.0)
    # Mutating a successfully constructed KeyRates grid must also be rejected
    # when its tenors are passed to a scalar API.
    mutated = KeyRates(copy(times))
    mutated.tenors[2] = mutated.tenors[1]
    bad_grids = (Float64[], [2.0, 1.0], [1.0, 1.0], [0.0, 1.0], [-1.0, 1.0], [1.0, Inf], [1.0, NaN], mutated.tenors)
    for curve in (FM.Yield.Constant(FC.Continuous(0.04)), FM.ZeroRateCurve([0.03, 0.04, 0.05], times))
        valuation = CashflowValue(cfs, times)
        forms = (
            g -> convexity(valuation, curve, g),
            g -> convexity(curve, g, cfs, times),
            g -> convexity(curve, g, wrapped),
            g -> convexity(Effective(), bond, curve, g),
            g -> convexity(Effective(), [bond], curve, g),
            g -> convexity(curve, g, Float64[], Float64[]),
            g -> convexity(curve, g, zeros(3), times),
            g -> convexity(curve, g, empty(wrapped)),
        )
        for grid in bad_grids
            for f in forms
                @test_throws ArgumentError f(grid)
            end
            # Fail before invoking a valuation, not as a side effect of its AD.
            @test_throws ArgumentError convexity(_ -> error("valuation was called"), curve, grid)
        end
        for grid in (times, [0.5, 2.5, 5.0])
            reference = convexity(curve, valuation)
            @test convexity(valuation, curve, grid) ≈ reference
            @test convexity(curve, grid, cfs, times) ≈ reference
            @test convexity(curve, grid, wrapped) ≈ reference
            @test convexity(Effective(), bond, curve, grid) ≈ convexity(curve, c -> FC.pv(c, bond))
            @test convexity(Effective(), [bond], curve, grid) ≈ convexity(curve, c -> FC.pv(c, bond))
            @test iszero(convexity(curve, grid, Float64[], Float64[]))
            @test iszero(convexity(curve, grid, zeros(3), times))
            @test iszero(convexity(curve, grid, empty(wrapped)))
        end
    end
end
