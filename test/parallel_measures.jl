@testset "Fixed cashflows on two curves have the combined rate's DV01" begin
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
        dv01 = duration(DV01(), base + spread, amounts, times)
        @test dv01 ≈ sign * expected rtol = 1.0e-12
        @test duration(DV01(), base + spread, FC.Cashflow.(amounts, times)) == dv01
        @test duration(DV01(), base + spread, amounts) ≈ dv01 rtol = 1.0e-14   # times default to 1:3
    end
end

@testset "Fixed-cashflow scalar measures keep the discounted cashflows' type" begin
    # Macaulay, modified, DV01 and convexity share one shock-coordinate kernel, so rates and
    # curves give results of the same type, empty or not, and DV01 is modified duration in
    # dollars.
    for (r, T) in ((0.03, Float64), (0.03f0, Float32), (big"0.03", BigFloat))
        cfs, times = T[5, 5, 105], T[1, 2, 3]
        for y in (r, FC.Periodic(r, 2), FC.Continuous(r), FM.Yield.Constant(FC.Continuous(r)))
            for m in (Macaulay(), Modified(), DV01())
                @test (@inferred duration(m, y, cfs, times)) isa T
                @test (@inferred duration(m, y, zero(cfs), times)) isa T
                @test (@inferred duration(m, y, cfs)) isa T
            end
            @test @inferred(convexity(y, cfs, times)) isa Real
            @test duration(DV01(), y, cfs, times) ≈ duration(Modified(), y, cfs, times) * FC.pv(y, cfs, times) / 10_000
        end
    end
    cfs, times = [5.0, 5.0, 105.0], [1.0, 2.0, 3.0]
    for rate in (r -> r, r -> FC.Periodic(r, 2), r -> FC.Continuous(r), r -> FM.Yield.Constant(FC.Continuous(r)))
        for m in (Modified(), DV01())
            @test ForwardDiff.derivative(r -> @inferred(duration(m, rate(r), cfs, times)), 0.03) isa Float64
        end
        @test ForwardDiff.derivative(r -> @inferred(convexity(rate(r), cfs, times)), 0.03) isa Float64
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
    @test duration(zrc, cfs, times) ≈ duration(value, zrc) rtol = 1.0e-12
    @test duration(zrc, cfs, times) ≈ sum(duration(KeyRates(tenors), zrc, cfs, times)) rtol = 1.0e-12
    @test duration(DV01(), zrc, cfs, times) ≈ macaulay * V / 10_000 rtol = 1.0e-14
    @test duration(DV01(), zrc, cfs, times) ≈ duration(value, DV01(), zrc) rtol = 1.0e-12
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
    ir01 = duration(floater, IR01(), base, credit)
    cs01 = duration(floater, CS01(), base, credit)
    @test ir01 ≈ sum(duration(floater, IR01(), kr, base, credit)) rtol = 1.0e-12
    @test cs01 ≈ sum(duration(floater, CS01(), kr, base, credit)) rtol = 1.0e-12
    @test abs(ir01) < abs(cs01) / 10
    @test duration(IR01(), base, credit) do b, c
        floater(b, c)
    end == ir01
    @test duration(CS01(), base, credit) do b, c
        floater(b, c)
    end == cs01

    blocks = convexity(floater, base, credit)
    matrices = convexity(floater, kr, base, credit)
    @test blocks.base.base ≈ sum(matrices.base.base) rtol = 1.0e-10
    @test blocks.credit.credit ≈ sum(matrices.credit.credit) rtol = 1.0e-10
    @test blocks.base.credit ≈ sum(matrices.base.credit) rtol = 1.0e-10
    @test blocks.credit.base == blocks.base.credit
    # The floater uses the curves differently, so the key-rate cross block is not symmetric;
    # the two cross blocks are each other's transposes.
    @test !(matrices.base.credit ≈ transpose(matrices.base.credit))
    @test matrices.credit.base ≈ transpose(matrices.base.credit)
    @test blocks == sensitivities(floater, SecondOrder(), base, credit).convexity

    fixed_value(b, c) = FC.present_value(b + c, cfs, times)
    fixed = convexity(fixed_value, base, credit)
    fixed_matrices = convexity(fixed_value, kr, base, credit)
    @test fixed.base.base ≈ sum(fixed_matrices.base.base) rtol = 1.0e-12
    @test fixed.base.credit ≈ fixed.base.base
    @test fixed.base.base ≈ convexity(base + credit, cfs, times) rtol = 1.0e-12
    # Shifting both curves together shifts the combined rate twice: all four blocks add up.
    both = fixed.base.base + fixed.base.credit + fixed.credit.base + fixed.credit.credit
    @test both ≈ 4 * convexity(base + credit, cfs, times) rtol = 1.0e-10
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
        # Fixed cashflows on several curves are valued on the combined curve.
        () -> duration(IR01(), curve, credit, cfs, times),
        () -> duration(CS01(), curve, credit, wrapped),
        () -> duration(IR01(), KeyRates(tenors), curve, credit, cfs, times),
        () -> convexity(curve, credit, cfs, times),
        () -> convexity(KeyRates(tenors), curve, credit, cfs, times),
        () -> sensitivities(KeyRates(tenors), curve, credit, cfs, times),
        () -> sensitivities(DV01(), KeyRates(tenors), curve, credit, cfs, times),
        () -> sensitivities(KeyRates(tenors), (; curve, credit), cfs, times),
        () -> convexity(KeyRates(tenors), (; curve, credit), cfs, times),
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
        () -> duration(DV01(), Effective(), bond, curve, tenors),
        () -> convexity(bond, curve, tenors),
        () -> convexity(Effective(), bond, curve, tenors),
        () -> duration(DV01(), bond, curve, tenors),
        # Key-rate grids are always a `KeyRates` marker.
        () -> sensitivities(bond, curve, tenors),
        () -> sensitivities(bond, curve, credit, tenors),
        () -> sensitivities(bond, tenors; discount = (; curve), index = curve),
        () -> sensitivities(c -> value(c.curve), (; curve); tenors),
        () -> sensitivities((; curve); tenors) do c
            value(c.curve)
        end,
        # The valuation callback comes first, before any marker or curve.
        () -> duration(curve, value),
        () -> duration(0.03, i -> FC.pv(i, cfs, times)),
        () -> duration(DV01(), curve, value),
        () -> convexity(curve, value),
        () -> duration(IR01(), value2, curve, credit),
        () -> duration(KeyRates(tenors), value, curve),
        () -> duration(DV01(), KeyRates(tenors), value, curve),
        () -> duration(CS01(), KeyRates(tenors), value2, curve, credit),
        () -> convexity(KeyRates(tenors), value, curve),
        () -> convexity(KeyRates(tenors), value2, curve, credit),
        () -> sensitivities(KeyRates(tenors), value, curve),
        () -> sensitivities(DV01(), KeyRates(tenors), value2, curve, credit),
        () -> sensitivities(KeyRates(tenors), c -> value(c.curve), (; curve)),
        # The curve comes before the contract, and the index curve is a keyword.
        () -> duration(Effective(), bond, curve),
        () -> duration(Spread(), bond, curve, credit),
        () -> duration(Effective(), KeyRates(tenors), bond, curve),
        () -> duration(DV01(), Effective(), bond, curve),
        () -> convexity(Effective(), bond, curve),
        () -> duration(KeyRates(tenors), bond, curve),
        () -> sensitivities(KeyRates(tenors), bond, curve),
        () -> sensitivities(KeyRates(tenors), bond, curve, credit),
        () -> sensitivities(KeyRates(tenors), bond; discount = (; curve), index = curve),
        () -> zspread(bond, curve, 100.0),
        () -> zspread(curve, bond, 100.0; forward = curve),
        # DV01 is a field of every `sensitivities` result, not a marker.
        () -> sensitivities(DV01(), KeyRates(tenors), curve, cfs, times),
        () -> sensitivities(value, DV01(), KeyRates(tenors), curve),
        () -> sensitivities(value2, DV01(), KeyRates(tenors), curve, credit),
    )
    for f in removed
        @test_throws MethodError f()
    end
    for name in (:KeyRate, :KeyRateZero, :KeyRatePar)
        @test !isdefined(ActuaryUtilities, name)
    end
end
