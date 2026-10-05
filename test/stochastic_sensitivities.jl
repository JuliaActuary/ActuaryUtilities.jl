@testset "Hull-White MC: sum of KRDs = deterministic (risk-neutral guarantee)" begin
    # For fixed cashflows, E[V] = Σ cf_i × P(0,t_i) under any risk-neutral model
    # (Glasserman, 2003, Ch. 7), so the sum of key rate durations is preserved
    # between deterministic discounting and Monte Carlo under Hull-White dynamics.
    # Individual KRDs differ because HW's θ(t) calibration creates non-local
    # rate dependencies (Brigo & Mercurio, 2006, Ch. 3).
    rates = [0.03, 0.03, 0.03, 0.03, 0.03]
    tenors = [1.0, 2.0, 3.0, 4.0, 5.0]
    cfs = [5.0, 5.0, 5.0, 5.0, 105.0]

    # Deterministic KRDs
    zrc = FM.ZeroRateCurve(rates, tenors)
    det = sensitivities(KeyRates(tenors), zrc, cfs, tenors)

    # Hull-White MC KRDs (AD through Monte Carlo via pathwise differentiation)
    hw_result = sensitivities(KeyRates(tenors), zrc) do curve
        hw = FM.ShortRate.HullWhite(0.1, 0.01, curve)
        scenarios = FM.simulate(hw; n_scenarios = 500, timestep = 1 / 12, horizon = 6.0, rng = Xoshiro(42))
        sum(sum(cf * FC.discount(sc, t) for (cf, t) in zip(cfs, tenors)) for sc in scenarios) / 500
    end

    # Total duration preserved (risk-neutral pricing theorem)
    @test sum(hw_result.duration) ≈ sum(det.duration) atol = 0.05

    # Individual KRDs should differ (HW redistributes across tenors)
    @test !(hw_result.duration ≈ det.duration)

    # Present values should also agree
    @test hw_result.value ≈ det.value atol = 0.5
end

@testset "Hull-White scenarios: pathwise consistency" begin
    # `Scenarios` draws one UInt64 from the user's rng when it is constructed and rebuilds
    # Xoshiro(seed) inside each AD evaluation. Two calls seeded the same way must produce
    # bit-identical results — otherwise ForwardDiff's many evaluations of the closure each
    # draw different MC samples and KRD = -∇V/V is biased by MC noise.
    rates = [0.03, 0.03, 0.03, 0.03, 0.03]
    tenors = [1.0, 2.0, 3.0, 4.0, 5.0]
    cfs = [5.0, 5.0, 5.0, 5.0, 105.0]
    zrc = FM.ZeroRateCurve(rates, tenors)
    hw = FM.ShortRate.HullWhite(0.1, 0.01, zrc)

    for order in (FirstOrder(), SecondOrder())
        r1 = sensitivities(order, KeyRates(tenors), Scenarios(hw; horizon = 6.0, n_scenarios = 500, rng = Xoshiro(42)), cfs, tenors)
        r2 = sensitivities(order, KeyRates(tenors), Scenarios(hw; horizon = 6.0, n_scenarios = 500, rng = Xoshiro(42)), cfs, tenors)
        @test isequal(r1, r2)
        # Different seeds give different MC samples (sanity check the seed is actually used)
        r3 = sensitivities(order, KeyRates(tenors), Scenarios(hw; horizon = 6.0, n_scenarios = 500, rng = Xoshiro(43)), cfs, tenors)
        @test !(r1.value ≈ r3.value && r1.duration ≈ r3.duration)
    end

    # Omitted times default to periods 1:n, and wrapped cashflows simulate too.
    wrapped = FC.Cashflow.(cfs, tenors)
    scenarios() = Scenarios(hw; horizon = 6.0, n_scenarios = 500, rng = Xoshiro(42))
    for m in ((), (SecondOrder(),))
        seeded = sensitivities(m..., KeyRates(tenors), scenarios(), cfs, tenors)
        @test isequal(sensitivities(m..., KeyRates(tenors), scenarios(), cfs), seeded)
        @test _same_sensitivity(sensitivities(m..., KeyRates(tenors), scenarios(), wrapped), seeded)
    end
    @test !(sensitivities(KeyRates(tenors), scenarios(), wrapped).duration ≈ sensitivities(KeyRates(tenors), hw.curve, wrapped).duration)

    # A callback receives the simulated paths, from the seed fixed at construction.
    value(paths) = sum(FC.pv(p, cfs, tenors) for p in paths) / length(paths)
    for m in ((), (SecondOrder(),))
        s = Scenarios(hw; n_scenarios = 50, timestep = 0.25, horizon = 6.0, rng = Xoshiro(1))
        first_call, second_call = sensitivities(value, m..., KeyRates(tenors), s), sensitivities(value, m..., KeyRates(tenors), s)
        @test isequal(sensitivities(value, m..., KeyRates(tenors), Scenarios(hw; n_scenarios = 50, timestep = 0.25, horizon = 6.0, rng = Xoshiro(1))), first_call)
        @test isequal(first_call, second_call)
    end
end

@testset "A bare Hull-White model is a curve" begin
    tenors = [1.0, 2.0, 3.0, 4.0, 5.0]
    cfs = [5.0, 5.0, 5.0, 5.0, 105.0]
    zrc = FM.ZeroRateCurve(fill(0.03, 5), tenors)
    hw = FM.ShortRate.HullWhite(0.1, 0.01, zrc)
    kr = KeyRates(tenors)
    value(c) = FC.pv(c, cfs, tenors)
    # Without `Scenarios`, every measure values `hw` on its discount function.
    @test !(Scenarios(hw; horizon = 6.0) isa FM.Yield.AbstractYieldModel)
    for o in ((), (SecondOrder(),)), g in ((), (kr,))
        @test sensitivities(o..., g..., hw, cfs, tenors) == sensitivities(o..., g..., zrc, cfs, tenors)
        @test _same_sensitivity(sensitivities(value, o..., g..., hw), sensitivities(value, o..., g..., zrc))
    end
    @test sensitivities(kr, hw, cfs, tenors).duration == duration(kr, hw, cfs, tenors)
end

@testset "Scenarios fix the horizon, the grid and the seed" begin
    tenors = [1.0, 2.0, 5.0]
    kr = KeyRates(tenors)
    curve = FM.ZeroRateCurve([0.03, 0.032, 0.035], tenors)
    hw = FM.ShortRate.HullWhite(0.1, 0.01, curve)

    # The horizon is required and must lie on the time grid, up to rounding.
    @test_throws UndefKeywordError Scenarios(hw)
    # 0.07 / 0.01 is 7.000000000000001 and 0.9 / 0.3 is 3.0000000000000004: the paths take that many
    # steps and end at the horizon, so a payment there is valued and one a step later throws.
    for (horizon, timestep) in ((0.07, 0.01), (0.9, 0.3), (5, 1 / 12))
        s = Scenarios(hw; horizon, timestep, n_scenarios = 2)
        @test isfinite(sensitivities(s, [1.0], [horizon]).value)
        @test_throws "Cannot extrapolate" sensitivities(s, [1.0], [horizon + timestep])
    end
    # An unaligned horizon throws before the seed is drawn.
    rng = Xoshiro(3)
    untouched = copy(rng)
    @test_throws ArgumentError Scenarios(hw; horizon = 0.9, timestep = 0.5, rng)
    @test_throws ArgumentError Scenarios(hw; horizon = -1.0, rng)
    @test rand(rng) == rand(untouched)

    # Construction draws one seed from `rng`; valuation, zero streams included, never uses it.
    rng, expected = Xoshiro(4), Xoshiro(4)
    s = Scenarios(hw; horizon = 5.0, timestep = 0.25, n_scenarios = 64, rng)
    @test s.seed == rand(expected, UInt64)
    short, long = ([5.0, 105.0], [1.0, 2.0]), ([4.0, 4.0, 104.0], [1.0, 3.0, 5.0])   # 5.0 is the horizon
    value(paths) = sum(FC.pv(p, short...) for p in paths) / length(paths)
    for order in (FirstOrder(), SecondOrder()), grid in ((), (kr,))
        both = sensitivities(order, grid..., s, [short[1]; long[1]], [short[2]; long[2]])
        a, b = sensitivities(order, grid..., s, short...), sensitivities(order, grid..., s, long...)
        # The same paths every call, so values and dollar derivatives add across unequal maturities,
        # and normalized measures add when weighted by value.
        @test isequal(a, sensitivities(order, grid..., s, short...))
        @test both.value ≈ a.value + b.value rtol = 1.0e-12
        @test both.dv01 ≈ a.dv01 .+ b.dv01 rtol = 1.0e-12
        @test both.duration ≈ (a.value .* a.duration .+ b.value .* b.duration) ./ both.value rtol = 1.0e-12
        if order isa SecondOrder
            @test both.convexity ≈ (a.value .* a.convexity .+ b.value .* b.convexity) ./ both.value rtol = 1.0e-12
        end
        @test _same_sensitivity(sensitivities(value, order, grid..., s), a)
        @test isequal(sensitivities(order, grid..., s, zeros(2), [1.0, 50.0]), sensitivities(order, grid..., hw, zeros(2), [1.0, 50.0]))
    end
    @test rand(rng) == rand(expected)
    # Same seed, same results; the paths end at the horizon, so a later payment throws.
    @test isequal(sensitivities(kr, Scenarios(hw; horizon = 5.0, timestep = 0.25, n_scenarios = 64, rng = Xoshiro(4)), long...), sensitivities(kr, s, long...))
    @test_throws "Cannot extrapolate" sensitivities(s, [1.0, 1.0], [1.0, 5.5])
    @test_throws "Cannot extrapolate" sensitivities(paths -> sum(FC.pv(p, [1.0], [6.0]) for p in paths), s)
end
