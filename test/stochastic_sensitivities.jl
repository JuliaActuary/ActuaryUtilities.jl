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

@testset "Hull-White convenience method: pathwise consistency" begin
    # The four `sensitivities(KeyRates, hw, ...)` convenience methods snapshot
    # one UInt64 from the user's rng and rebuild Xoshiro(seed) inside each AD
    # evaluation. Two calls seeded the same way must produce bit-identical
    # results — otherwise ForwardDiff's many evaluations of the closure each
    # draw different MC samples and KRD = -∇V/V is biased by MC noise.
    rates = [0.03, 0.03, 0.03, 0.03, 0.03]
    tenors = [1.0, 2.0, 3.0, 4.0, 5.0]
    cfs = [5.0, 5.0, 5.0, 5.0, 105.0]
    zrc = FM.ZeroRateCurve(rates, tenors)
    hw = FM.ShortRate.HullWhite(0.1, 0.01, zrc)

    r1 = sensitivities(
        KeyRates(tenors), hw, cfs, tenors;
        n_scenarios = 500, rng = Xoshiro(42)
    )
    r2 = sensitivities(
        KeyRates(tenors), hw, cfs, tenors;
        n_scenarios = 500, rng = Xoshiro(42)
    )
    @test r1.value ≈ r2.value
    @test r1.durations ≈ r2.durations
    @test r1.convexities ≈ r2.convexities

    # DV01 form
    d1 = sensitivities(
        DV01(), KeyRates(tenors), hw, cfs, tenors;
        n_scenarios = 500, rng = Xoshiro(42)
    )
    d2 = sensitivities(
        DV01(), KeyRates(tenors), hw, cfs, tenors;
        n_scenarios = 500, rng = Xoshiro(42)
    )
    @test d1.value ≈ d2.value
    @test d1.dv01s ≈ d2.dv01s
    @test d1.convexities ≈ d2.convexities

    # Different seeds give different MC samples (sanity check the seed is actually used)
    r3 = sensitivities(
        KeyRates(tenors), hw, cfs, tenors;
        n_scenarios = 500, rng = Xoshiro(43)
    )
    @test !(r1.value ≈ r3.value && r1.durations ≈ r3.durations)
end
