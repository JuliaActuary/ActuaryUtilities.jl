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
    @test sum(hw_result.durations) ≈ sum(det.durations) atol = 0.05

    # Individual KRDs should differ (HW redistributes across tenors)
    @test !(hw_result.durations ≈ det.durations)

    # Present values should also agree
    @test hw_result.value ≈ det.value atol = 0.5
end

@testset "Hull-White scenarios: pathwise consistency" begin
    # The four `sensitivities(..., KeyRates, Scenarios(hw), ...)` methods snapshot
    # one UInt64 from the user's rng and rebuild Xoshiro(seed) inside each AD
    # evaluation. Two calls seeded the same way must produce bit-identical
    # results — otherwise ForwardDiff's many evaluations of the closure each
    # draw different MC samples and KRD = -∇V/V is biased by MC noise.
    rates = [0.03, 0.03, 0.03, 0.03, 0.03]
    tenors = [1.0, 2.0, 3.0, 4.0, 5.0]
    cfs = [5.0, 5.0, 5.0, 5.0, 105.0]
    zrc = FM.ZeroRateCurve(rates, tenors)
    hw = FM.ShortRate.HullWhite(0.1, 0.01, zrc)

    r1 = sensitivities(KeyRates(tenors), Scenarios(hw; n_scenarios = 500, rng = Xoshiro(42)), cfs, tenors)
    r2 = sensitivities(KeyRates(tenors), Scenarios(hw; n_scenarios = 500, rng = Xoshiro(42)), cfs, tenors)
    @test r1.value ≈ r2.value
    @test r1.durations ≈ r2.durations
    @test r1.convexities ≈ r2.convexities

    # DV01 form
    d1 = sensitivities(DV01(), KeyRates(tenors), Scenarios(hw; n_scenarios = 500, rng = Xoshiro(42)), cfs, tenors)
    d2 = sensitivities(DV01(), KeyRates(tenors), Scenarios(hw; n_scenarios = 500, rng = Xoshiro(42)), cfs, tenors)
    @test d1.value ≈ d2.value
    @test d1.dv01s ≈ d2.dv01s
    @test d1.convexities ≈ d2.convexities

    # Different seeds give different MC samples (sanity check the seed is actually used)
    r3 = sensitivities(KeyRates(tenors), Scenarios(hw; n_scenarios = 500, rng = Xoshiro(43)), cfs, tenors)
    @test !(r1.value ≈ r3.value && r1.durations ≈ r3.durations)

    # Omitted times default to periods 1:n, and wrapped cashflows simulate too.
    wrapped = FC.Cashflow.(cfs, tenors)
    scenarios() = Scenarios(hw; n_scenarios = 500, rng = Xoshiro(42))
    for m in ((), (DV01(),))
        seeded = sensitivities(m..., KeyRates(tenors), scenarios(), cfs, tenors)
        @test isequal(sensitivities(m..., KeyRates(tenors), scenarios(), cfs), seeded)
        @test _same_sensitivity(sensitivities(m..., KeyRates(tenors), scenarios(), wrapped), seeded)
    end
    @test !(sensitivities(KeyRates(tenors), scenarios(), wrapped).durations ≈ sensitivities(KeyRates(tenors), hw.curve, wrapped).durations)

    # A callback receives the simulated paths, from one seed per call.
    value(paths) = sum(FC.pv(p, cfs, tenors) for p in paths) / length(paths)
    for m in ((), (DV01(),))
        s = Scenarios(hw; n_scenarios = 50, timestep = 0.25, horizon = 6.0, rng = Xoshiro(1))
        first_call, second_call = sensitivities(value, m..., KeyRates(tenors), s), sensitivities(value, m..., KeyRates(tenors), s)
        @test isequal(sensitivities(value, m..., KeyRates(tenors), Scenarios(hw; n_scenarios = 50, timestep = 0.25, horizon = 6.0, rng = Xoshiro(1))), first_call)
        @test !isequal(first_call, second_call)
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
    @test !(Scenarios(hw) isa FM.Yield.AbstractYieldModel)
    @test sensitivities(kr, hw, cfs, tenors) == sensitivities(kr, zrc, cfs, tenors)
    @test sensitivities(DV01(), kr, hw, cfs, tenors) == sensitivities(DV01(), kr, zrc, cfs, tenors)
    @test sensitivities(kr, hw, cfs, tenors).durations == duration(kr, hw, cfs, tenors)
    @test _same_sensitivity(sensitivities(value, kr, hw), sensitivities(value, kr, zrc))
    @test _same_sensitivity(sensitivities(value, DV01(), kr, hw), sensitivities(value, DV01(), kr, zrc))
end
