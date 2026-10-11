@testset "Callable valuations and cashflow dispatch" begin
    curve = FM.Yield.Constant(FC.Continuous(0.04))
    credit = FM.Yield.Constant(FC.Continuous(0.01))
    tenors = [1.0, 3.0, 7.0]
    kr = KeyRates(tenors)
    for sign in (-1, 1)
        value = CashflowValue([sign * 100.0], [2.0])
        closure(c) = value(c)
        pair(b, c) = value(b, c)
        @test !(value isa Function)
        @test duration(value, curve) ≈ 2.0
        @test convexity(value, curve) ≈ 4.0
        @test duration(value, DV01(), curve) ≈ 2value(curve) / 10_000
        for yield in (0.04, FC.Periodic(0.04, 2), FC.Continuous(0.04), curve)
            @test duration(value, yield) ≈ duration(closure, yield)
            @test convexity(value, yield) ≈ convexity(closure, yield)
            @test duration(value, DV01(), yield) ≈ duration(closure, DV01(), yield)
            # The callback comes first, so do-block syntax works for every input type.
            @test duration(yield) do c
                value(c)
            end == duration(closure, yield)
            @test duration(DV01(), yield) do c
                value(c)
            end == duration(closure, DV01(), yield)
            @test convexity(yield) do c
                value(c)
            end == convexity(closure, yield)
        end
        for metric in (IR01(), CS01())
            @test duration(value, metric, curve, credit) ≈ duration(pair, metric, curve, credit)
            @test duration(value, metric, kr, curve, credit) ≈ duration(pair, metric, kr, curve, credit)
        end
        @test _same_sensitivity(convexity(value, curve, credit), convexity(pair, curve, credit))
        for f in (duration, convexity)
            @test _same_sensitivity(f(value, kr, curve), f(closure, kr, curve))
        end
        @test _same_sensitivity(convexity(value, kr, curve, credit), convexity(pair, kr, curve, credit))
        @test duration(value, DV01(), kr, curve) ≈ duration(closure, DV01(), kr, curve)
        for order in ((), (FirstOrder(),), (SecondOrder(),)), grid in ((), (kr,))
            @test _same_sensitivity(sensitivities(value, order..., grid..., curve), sensitivities(closure, order..., grid..., curve))
            @test _same_sensitivity(sensitivities(value, order..., grid..., curve, credit), sensitivities(pair, order..., grid..., curve, credit))
        end
    end

    # Arrays, ranges, views, and wrapped cashflows still select collection routes.
    for cashflows in ([5.0, 5.0, 105.0], 1.0:3.0, view([5.0, 5.0, 105.0], :), FC.Cashflow.([5.0, 5.0, 105.0], [1.0, 2.0, 3.0]))
        times = 1:3
        amounts = cashflows isa AbstractVector{<:FC.Cashflow} ? FC.amount.(cashflows) : cashflows
        @test duration(curve, cashflows) ≈ duration(curve, cashflows, times)
        @test convexity(curve, cashflows) ≈ convexity(curve, cashflows, times)
        for metric in (Macaulay(), Modified(), DV01(), kr)
            @test duration(metric, curve, cashflows) ≈ duration(metric, curve, amounts, times)
        end
    end
    @test duration(curve, reshape([5.0, 5.0, 105.0], 1, 3), 1:3) ≈ duration(curve, [5.0, 5.0, 105.0], 1:3)

    for cashflows in ((5.0, 5.0, 105.0), (c for c in [5.0, 5.0, 105.0]))
        for yield in (0.04, curve)
            @test duration(yield, cashflows) ≈ duration(yield, collect(cashflows))
            @test convexity(yield, cashflows) ≈ convexity(yield, collect(cashflows))
        end
    end

    # The scenario callback is also a callable object; use identical MC draws.
    hw = FM.ShortRate.HullWhite(0.1, 0.01, curve)
    value = ScenarioValue(CashflowValue([5.0, 105.0], [1.0, 3.0]))
    scenarios() = Scenarios(hw; n_scenarios = 8, timestep = 0.5, horizon = 3.0, rng = Random.Xoshiro(1234))
    for prefix in ((), (SecondOrder(),), (kr,), (SecondOrder(), kr))
        result = sensitivities(value, prefix..., scenarios())
        reference = sensitivities(s -> value(s), prefix..., scenarios())
        @test _same_sensitivity(result, reference)
    end
end

@testset "DV01 of a valuation callback" begin
    curve = FM.Yield.Constant(FC.Continuous(0.04))
    credit = FM.Yield.Constant(FC.Continuous(0.01))
    kr = KeyRates([1.0, 3.0, 7.0])
    value = CashflowValue([5.0, 105.0], [2.0, 6.0])   # a callable struct, for one or two curves
    closure(c) = value(c)
    pair(b, c) = value(b, c)
    named(c) = value(c.rf + c.spread)
    # A callable struct, a closure and a do-block give the same DV01.
    for yield in (0.04, FC.Periodic(0.04, 2), FC.Continuous(0.04), curve)
        @test duration(value, DV01(), yield) == duration(closure, DV01(), yield)
        @test duration(DV01(), yield) do c
            value(c)
        end == duration(closure, DV01(), yield)
    end
    @test duration(value, DV01(), kr, curve) == duration(closure, DV01(), kr, curve)
    @test duration(DV01(), kr, curve) do c
        value(c)
    end == duration(closure, DV01(), kr, curve)
    @test duration(value, DV01(), curve) ≈ sensitivities(value, curve).dv01
    @test duration(value, DV01(), kr, curve) ≈ sensitivities(value, kr, curve).dv01
    # Several curves: one DV01 per role, the `dv01` field of `sensitivities`.
    for grid in ((), (kr,))
        d = duration(pair, DV01(), grid..., curve, credit)
        @test keys(d) == (:base, :credit)
        @test d.base ≈ duration(pair, IR01(), grid..., curve, credit) rtol = 1.0e-12
        @test d.credit ≈ duration(pair, CS01(), grid..., curve, credit) rtol = 1.0e-12
        @test _same_sensitivity(d, sensitivities(pair, grid..., curve, credit).dv01)
        @test _same_sensitivity(duration(value, DV01(), grid..., curve, credit), d)
        @test _same_sensitivity(
            duration(DV01(), grid..., curve, credit) do b, c
                pair(b, c)
            end, d
        )
        n = duration(named, DV01(), grid..., (; rf = curve, spread = credit))
        @test keys(n) == (:rf, :spread)
        @test _same_sensitivity(n, sensitivities(named, grid..., (; rf = curve, spread = credit)).dv01)
        @test n.rf ≈ d.base rtol = 1.0e-12
    end
    # `dv01` is removed: `duration(DV01(), ...)` is the one standalone DV01.
    @test !isdefined(ActuaryUtilities, :dv01)
end
