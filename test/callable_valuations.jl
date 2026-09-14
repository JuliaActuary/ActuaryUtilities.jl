struct CashflowValue{C, T}
    cashflows::C
    times::T
end
(v::CashflowValue)(curve) = FC.pv(curve, v.cashflows, v.times)
(v::CashflowValue)(base, credit) = v(base + credit)

struct ScenarioValue{V}
    value::V
end
(v::ScenarioValue)(scenarios) = sum(v.value, scenarios) / length(scenarios)

_same_sensitivity(a, b) = isapprox(a, b; rtol = 1.0e-12, atol = 1.0e-12)
_same_sensitivity(a::NamedTuple, b::NamedTuple) =
    keys(a) == keys(b) && all(map(_same_sensitivity, values(a), values(b)))

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
        @test duration(curve, value) ≈ 2.0
        @test convexity(curve, value) ≈ 4.0
        @test duration(DV01(), curve, value) ≈ 2value(curve) / 10_000
        for yield in (0.04, FC.Periodic(0.04, 2), FC.Continuous(0.04), curve)
            @test duration(yield, value) ≈ duration(yield, closure)
            @test convexity(yield, value) ≈ convexity(yield, closure)
            @test duration(DV01(), yield, value) ≈ duration(DV01(), yield, closure)
        end
        @test duration(value, curve, tenors) ≈ duration(closure, curve, tenors)
        @test convexity(value, curve, tenors) ≈ convexity(closure, curve, tenors)
        @test duration(DV01(), value, curve, tenors) ≈ duration(DV01(), closure, curve, tenors)
        for metric in (IR01(), CS01())
            @test duration(metric, value, curve, credit, tenors) ≈ duration(metric, pair, curve, credit, tenors)
            @test duration(metric, kr, value, curve, credit) ≈ duration(metric, kr, pair, curve, credit)
        end
        @test _same_sensitivity(convexity(value, curve, credit, tenors), convexity(pair, curve, credit, tenors))
        for f in (duration, convexity, sensitivities)
            @test _same_sensitivity(f(kr, value, curve), f(kr, closure, curve))
        end
        for f in (convexity, sensitivities)
            @test _same_sensitivity(f(kr, value, curve, credit), f(kr, pair, curve, credit))
        end
        @test duration(DV01(), kr, value, curve) ≈ duration(DV01(), kr, closure, curve)
        @test _same_sensitivity(sensitivities(DV01(), kr, value, curve), sensitivities(DV01(), kr, closure, curve))
        @test _same_sensitivity(sensitivities(DV01(), kr, value, curve, credit), sensitivities(DV01(), kr, pair, curve, credit))
    end

    # Arrays, ranges, views, and wrapped cashflows still select collection routes.
    for cashflows in ([5.0, 5.0, 105.0], 1.0:3.0, view([5.0, 5.0, 105.0], :), FC.Cashflow.([5.0, 5.0, 105.0], [1.0, 2.0, 3.0]))
        times = 1:3
        amounts = cashflows isa AbstractVector{<:FC.Cashflow} ? FC.amount.(cashflows) : cashflows
        @test duration(curve, cashflows) ≈ duration(curve, cashflows, times)
        @test convexity(curve, cashflows) ≈ convexity(curve, cashflows, times)
        for metric in (Macaulay(), Modified(), DV01(), KeyRate(1), kr)
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
    for prefix in ((), (DV01(),))
        result = sensitivities(prefix..., kr, value, hw; n_scenarios = 8, timestep = 0.5, horizon = 3.0, rng = Random.Xoshiro(1234))
        reference = sensitivities(prefix..., kr, s -> value(s), hw; n_scenarios = 8, timestep = 0.5, horizon = 3.0, rng = Random.Xoshiro(1234))
        @test _same_sensitivity(result, reference)
    end
    @test isempty(Test.detect_ambiguities(ActuaryUtilities; recursive = true))
end
