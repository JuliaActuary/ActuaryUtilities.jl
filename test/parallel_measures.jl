@testset "Scalar IR01 and CS01 measure the combined rate" begin
    cfs = [5.0, 5.0, 105.0]
    times = [1.0, 2.0, 3.0]
    tenors = [1.0, 2.0, 3.0]
    zrc = FM.ZeroRateCurve([0.02, 0.028, 0.033], tenors)
    credit = FM.Yield.Constant(FC.Continuous(0.01))
    # Independent derivatives of value in each combined rate's own coordinate.
    annual(y) = sum(t * cf * (1 + y)^(-t - 1) for (cf, t) in zip(cfs, times)) / 10_000
    semiannual(r) = sum(t * cf * (1 + r / 2)^(-2t - 1) for (cf, t) in zip(cfs, times)) / 10_000
    continuous(c) = sum(t * cf * FC.discount(c, t) for (cf, t) in zip(cfs, times)) / 10_000
    cases = (
        (0.03, 0.01, annual(0.04)),
        (FM.Yield.Constant(FC.Periodic(0.04, 1)), 0.01, continuous(FM.Yield.Constant(FC.Periodic(0.04, 1)) + 0.01)),
        (0.04, credit, continuous(0.04 + credit)),
        (FC.Periodic(0.04, 2), FC.Continuous(0.01), semiannual(FC.rate(FC.Periodic(0.04, 2) + FC.Continuous(0.01)))),
        (zrc, credit, continuous(zrc + credit)),
    )
    for (base, spread, expected) in cases, sign in (-1, 1)
        amounts = sign .* cfs
        ir01 = duration(IR01(), base, spread, amounts, times)
        @test ir01 ≈ sign * expected rtol = 1.0e-12
        @test duration(CS01(), base, spread, amounts, times) == ir01
        @test duration(DV01(), base + spread, amounts, times) == ir01
        @test duration(IR01(), base, spread, FC.Cashflow.(amounts, times)) == ir01
    end
end

@testset "Yield-model parallel measures use continuous-zero fast paths" begin
    cfs = [5.0, 5.0, 105.0]
    times = [0.5, 2.0, 3.5]
    tenors = [1.0, 2.0, 3.0]
    zrc = FM.ZeroRateCurve([0.02, 0.028, 0.033], tenors)
    value(c) = FC.present_value(c, cfs, times)
    V = value(zrc)
    macaulay = sum(t * cf * FC.discount(zrc, t) for (cf, t) in zip(cfs, times)) / V
    @test duration(zrc, cfs, times) ≈ macaulay rtol = 1.0e-14
    @test duration(zrc, cfs, times) ≈ duration(zrc, value) rtol = 1.0e-12
    @test duration(zrc, cfs, times) ≈ sum(duration(KeyRates(tenors), zrc, cfs, times)) rtol = 1.0e-12
    @test duration(DV01(), zrc, cfs, times) ≈ macaulay * V / 10_000 rtol = 1.0e-14
    @test duration(DV01(), zrc, cfs, times) ≈ duration(DV01(), zrc, value) rtol = 1.0e-12
    @test duration(DV01(), zrc) do c
        value(c)
    end ≈ duration(DV01(), zrc, cfs, times) rtol = 1.0e-12

    # Dollar risk is defined at zero present value; normalized duration is not.
    hedge = [100.0, -100.0 * FC.discount(zrc, 1.0) / FC.discount(zrc, 3.0)]
    hedge_times = [1.0, 3.0]
    @test abs(FC.present_value(zrc, hedge, hedge_times)) < 1.0e-12
    dv01 = duration(DV01(), zrc, hedge, hedge_times)
    @test dv01 ≈ sum(t * cf * FC.discount(zrc, t) for (cf, t) in zip(hedge, hedge_times)) / 10_000
    @test dv01 < 0
    @test !isfinite(duration(zrc, hedge, hedge_times))
end

@testset "Two-curve callback measures without a tenor grid" begin
    cfs = [5.0, 5.0, 105.0]
    times = [1.0, 2.0, 3.0]
    tenors = [1.0, 2.0, 3.0]
    base = FM.ZeroRateCurve([0.02, 0.028, 0.033], tenors)
    credit = FM.ZeroRateCurve([0.01, 0.012, 0.015], tenors)
    kr = KeyRates(tenors)
    # Coupons project on the base curve and discount on base + credit, so the
    # curves play different roles and IR01 differs from CS01.
    previous_discount(b, t) = t == first(times) ? one(FC.discount(b, t)) : FC.discount(b, t - 1)
    floater(b, c) = sum(
        (k == length(times) ? 100.0 : 0.0) * FC.discount(b + c, t) +
            100.0 * (previous_discount(b, t) / FC.discount(b, t) - 1) * FC.discount(b + c, t)
            for (k, t) in enumerate(times)
    )
    ir01 = duration(IR01(), floater, base, credit)
    cs01 = duration(CS01(), floater, base, credit)
    @test ir01 ≈ sum(duration(IR01(), kr, floater, base, credit)) rtol = 1.0e-12
    @test cs01 ≈ sum(duration(CS01(), kr, floater, base, credit)) rtol = 1.0e-12
    @test abs(ir01) < abs(cs01) / 10
    @test duration(IR01(), base, credit) do b, c
        floater(b, c)
    end == ir01
    @test duration(CS01(), base, credit) do b, c
        floater(b, c)
    end == cs01

    blocks = convexity(floater, base, credit)
    matrices = convexity(kr, floater, base, credit)
    @test blocks.base ≈ sum(matrices.base) rtol = 1.0e-10
    @test blocks.credit ≈ sum(matrices.credit) rtol = 1.0e-10
    @test blocks.cross ≈ sum(matrices.cross) rtol = 1.0e-10

    fixed = convexity(base, credit, cfs, times)
    fixed_matrices = convexity(kr, base, credit, cfs, times)
    @test fixed.base ≈ sum(fixed_matrices.base) rtol = 1.0e-12
    @test fixed.cross ≈ fixed.base
    @test convexity(base, credit, FC.Cashflow.(cfs, times)) == fixed
    @test fixed.base ≈ convexity(base + credit, cfs, times) rtol = 1.0e-12
end

@testset "Removed v5 call shapes throw" begin
    cfs = [5.0, 5.0, 105.0]
    times = [1.0, 2.0, 3.0]
    tenors = [1.0, 2.0, 3.0]
    wrapped = FC.Cashflow.(cfs, times)
    curve = FM.ZeroRateCurve([0.02, 0.028, 0.033], tenors)
    credit = FM.Yield.Constant(FC.Continuous(0.01))
    value(c) = FC.present_value(c, cfs, times)
    value2(b, c) = FC.present_value(b + c, cfs, times)
    bond = FM.Bond.Fixed(0.05, FC.Periodic(1), 3.0)
    removed = (
        () -> duration(curve, tenors, cfs, times),
        () -> duration(curve, tenors, wrapped),
        () -> duration(DV01(), curve, tenors, cfs, times),
        () -> duration(DV01(), curve, tenors, wrapped),
        () -> duration(IR01(), curve, credit, tenors, cfs, times),
        () -> duration(IR01(), curve, credit, tenors, wrapped),
        () -> duration(CS01(), curve, credit, tenors, wrapped),
        () -> convexity(curve, tenors, cfs, times),
        () -> convexity(curve, tenors, wrapped),
        () -> convexity(FM.Yield.Constant(0.03), tenors, wrapped),
        () -> convexity(curve, credit, tenors, cfs, times),
        () -> convexity(curve, credit, tenors, wrapped),
        () -> duration(value, curve, tenors),
        () -> duration(DV01(), value, curve, tenors),
        () -> duration(IR01(), value2, curve, credit, tenors),
        () -> convexity(value, curve, tenors),
        () -> convexity(value2, curve, credit, tenors),
        () -> convexity(cfs, curve, tenors),
        () -> convexity(Tuple(cfs), curve, tenors),
        () -> duration(bond, curve, tenors),
        () -> duration(Effective(), bond, curve, tenors),
        () -> dv01(Effective(), bond, curve, tenors),
        () -> dv01(bond, curve, tenors),
        () -> convexity(bond, curve, tenors),
        () -> convexity(Effective(), bond, curve, tenors),
        () -> duration(DV01(), bond, curve, tenors),
    )
    for f in removed
        @test_throws MethodError f()
    end
    for name in (:KeyRate, :KeyRateZero, :KeyRatePar)
        @test !isdefined(ActuaryUtilities, name)
    end
end
