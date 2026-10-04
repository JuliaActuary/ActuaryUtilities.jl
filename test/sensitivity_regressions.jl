@testset "Cashflow collection boundaries" begin
    times = [1.0, 2.0, 3.0]
    curve = FM.Yield.Constant(FC.Continuous(0.04))
    kr = KeyRates([1.0, 2.0, 5.0])
    hw = FM.ShortRate.HullWhite(0.1, 0.01, curve)
    scenarios() = Scenarios(hw; n_scenarios = 4, timestep = 0.5, horizon = 4.0, rng = Random.Xoshiro(7))
    # Every representation of the same amounts, paired with the same times by position.
    collections(cfs) = (
        () -> Tuple(cfs),
        () -> (c for c in cfs),
        () -> (c for c in Iterators.Stateful(cfs)),
        () -> (c for c in OnePass(cfs)),
        () -> (c for c in cfs if true),
        () -> reshape(copy(cfs), 1, length(cfs)),
        () -> OffsetArray(copy(cfs), -2),
        () -> OffsetArray(copy(cfs), 5),
    )
    time_grids = (
        Tuple(times), (t for t in times), OffsetArray(times, -1), OffsetArray([times; 9.0; NaN], 3),
        [times; NaN], view([0.0; times], 2:4),
    )
    measures(yield) = (
        (c, t...) -> duration(Macaulay(), yield, c, t...),
        (c, t...) -> duration(Modified(), yield, c, t...),
        (c, t...) -> duration(DV01(), yield, c, t...),
        (c, t...) -> duration(yield, c, t...),
        (c, t...) -> convexity(yield, c, t...),
        (c, t...) -> present_values(yield, c, t...),
    )
    curve_measures = (
        (c, t...) -> duration(kr, curve, c, t...),
        (c, t...) -> duration(DV01(), kr, curve, c, t...),
        (c, t...) -> convexity(kr, curve, c, t...),
        (c, t...) -> sensitivities(kr, curve, c, t...),
        (c, t...) -> sensitivities(DV01(), kr, curve, c, t...),
        (c, t...) -> sensitivities(kr, scenarios(), c, t...),
        (c, t...) -> sensitivities(DV01(), kr, scenarios(), c, t...),
    )
    same(a, b) = a === b || _same_sensitivity(a, b)   # breakeven can return `nothing`
    for cfs in ([5.0, 5.0, 105.0], [-5.0, -5.0, -105.0], [-100.0, 30.0, 80.0], zeros(3))
        calls = Any[measures(0.04)..., measures(FC.Periodic(0.04, 2))..., measures(curve)..., curve_measures...]
        push!(calls, (c, t...) -> breakeven(0.04, c, t...))
        cfs == [5.0, 5.0, 105.0] && push!(calls, (c, t...) -> spread(0.04, 0.05, c, t...))
        for f in calls
            expected = f(cfs, times)
            for make in collections(cfs)
                @test same(f(make(), times), expected)
            end
            for grid in time_grids
                @test same(f(cfs, grid), expected)
                @test same(f(OffsetArray(copy(cfs), 4), grid), expected)
            end
            @test_throws DimensionMismatch f(cfs, times[1:2])
            @test_throws DimensionMismatch f(OffsetArray(copy(cfs), 4), OffsetArray(times[1:2], -7))
        end
    end
    # Omitted times are the amounts' indices, an offset vector's included.
    cfs = [5.0, 5.0, 105.0]
    for yield in (0.04, curve)
        @test duration(yield, OffsetArray(cfs, 1)) ≈ duration(yield, cfs, 2:4)
        @test duration(yield, (c for c in cfs)) ≈ duration(yield, cfs, 1:3)
    end
    @test duration(kr, curve, OffsetArray(cfs, -1)) ≈ duration(kr, curve, cfs, 0:2)
    @test breakeven(0.1, OffsetArray([-10, 1, 2, 3, 4, 8], 1)) == breakeven(0.1, [-10, 1, 2, 3, 4, 8], 1:6)
    # Offset times pair by position: 100 at times 1 and 2, not at 2 and 3.
    offset = OffsetArray([1.0, 2.0, 3.0], 0:2)
    flat = FM.Yield.Constant(FC.Continuous(0.03))
    @test round(first(present_values(flat, [100.0, 100.0], offset)); digits = 2) == 191.22
    @test sensitivities(kr, flat, [100.0, 100.0], offset).value ≈ 100exp(-0.03) + 100exp(-0.06) rtol = 1.0e-15
    @test duration(Macaulay(), flat, [100.0, 100.0], offset) ≈ duration(Macaulay(), flat, [100.0, 100.0], [1.0, 2.0])
    wrapped = FC.Cashflow.([5.0, 5.0, 105.0], [0.5, 1.5, 2.5])
    for make in collections(wrapped), metric in (Macaulay(), Modified(), DV01())
        @test duration(metric, curve, make()) ≈ duration(metric, curve, wrapped)
    end
    # Cashflows before the curve are not a valuation callback.
    @test_throws MethodError duration([5.0, 5.0, 105.0], curve, times)
    @test_throws MethodError convexity([5.0, 5.0, 105.0], curve, times)
end

@testset "KeyRates owns a 1-based grid" begin
    tenors = [1.0, 2.0, 5.0]
    curve = FM.ZeroRateCurve([0.02, 0.03, 0.04], tenors)
    cfs, times = [5.0, 5.0, 105.0], [0.5, 2.5, 6.0]
    value(c) = FC.pv(c, cfs, times)
    plain = KeyRates(tenors)
    for grid in (OffsetArray(tenors, -3), OffsetArray(tenors, 10), (t for t in tenors), view([0.0; tenors], 2:4))
        kr = KeyRates(grid isa Base.Generator ? collect(grid) : grid)
        @test kr.tenors == tenors && kr.tenors isa Vector{Float64}
        @test duration(kr, curve, cfs, times) == duration(plain, curve, cfs, times)
        @test convexity(kr, curve, cfs, times) == convexity(plain, curve, cfs, times)
        @test sensitivities(kr, curve, cfs, times) == sensitivities(plain, curve, cfs, times)
        @test duration(value, kr, curve) == duration(value, plain, curve)
        @test sensitivities(value, kr, curve) == sensitivities(value, plain, curve)
    end
    # The grid is a copy: changing the caller's vector does not change it.
    owned = KeyRates(tenors)
    tenors[2] = 0.5
    @test owned.tenors == [1.0, 2.0, 5.0]
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
        @test duration(vf, curve) ≈ expected_duration
        @test sum(duration(vf, kr, curve)) ≈ expected_duration
        @test duration(vf, kr, curve) ≈ duration(kr, curve, amounts, tenors)
        @test convexity(curve, amounts, tenors) ≈ expected_convexity
        @test sum(convexity(vf, kr, curve)) ≈ expected_convexity
        @test convexity(vf, kr, curve) ≈ convexity(kr, curve, amounts, tenors)
        @test duration(DV01(), curve, amounts, tenors) ≈ sign * value * expected_duration / 10_000
        @test sum(duration(vf, DV01(), kr, curve)) ≈ sign * value * expected_duration / 10_000
    end
    bond = FM.Bond.Fixed(0.05, FC.Periodic(1), 3.0)
    reference = FM.Yield.Constant(FC.Periodic(0.04, 1))
    @test duration(Effective(), bond, curve) ≈ duration(Effective(), bond, reference)
    @test duration(Spread(), bond, curve) ≈ duration(Spread(), bond, reference)
    spread = 0.012
    market_price = FC.pv(FM.Yield.Constant(FC.Continuous(log1p(0.04) + spread)), bond)
    result = zspread(bond, curve, market_price)
    expected = zspread(bond, reference, market_price)
    @test FC.rate(result.zspread) ≈ spread atol = 1.0e-10
    @test result.zspread_dv01 ≈ expected.zspread_dv01
end

@testset "Scalar convexity forms agree without a tenor grid" begin
    cfs = [5.0, 5.0, 105.0]
    times = [1.0, 2.0, 3.0]
    wrapped = FC.Cashflow.(cfs, times)
    bond = FM.Bond.Fixed(0.05, FC.Periodic(1), 3.0)
    for curve in (FM.Yield.Constant(FC.Continuous(0.04)), FM.ZeroRateCurve([0.03, 0.04, 0.05], times))
        valuation = CashflowValue(cfs, times)
        reference = convexity(valuation, curve)
        @test convexity(curve, cfs, times) ≈ reference
        @test convexity(curve, wrapped) ≈ reference
        @test convexity(Effective(), bond, curve) ≈ convexity(c -> FC.pv(c, bond), curve)
        @test convexity(Effective(), [bond], curve) ≈ convexity(c -> FC.pv(c, bond), curve)
        @test iszero(convexity(curve, Float64[], Float64[]))
        @test iszero(convexity(curve, zeros(3), times))
        @test iszero(convexity(curve, empty(wrapped)))
    end
end
