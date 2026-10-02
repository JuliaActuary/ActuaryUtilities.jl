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
        for f in (duration, convexity, sensitivities)
            @test _same_sensitivity(f(value, kr, curve), f(closure, kr, curve))
        end
        for f in (convexity, sensitivities)
            @test _same_sensitivity(f(value, kr, curve, credit), f(pair, kr, curve, credit))
        end
        @test duration(value, DV01(), kr, curve) ≈ duration(closure, DV01(), kr, curve)
        @test _same_sensitivity(sensitivities(value, DV01(), kr, curve), sensitivities(closure, DV01(), kr, curve))
        @test _same_sensitivity(sensitivities(value, DV01(), kr, curve, credit), sensitivities(pair, DV01(), kr, curve, credit))
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
    for prefix in ((), (DV01(),))
        result = sensitivities(value, prefix..., kr, scenarios())
        reference = sensitivities(s -> value(s), prefix..., kr, scenarios())
        @test _same_sensitivity(result, reference)
    end
    @test isempty(Test.detect_ambiguities(ActuaryUtilities; recursive = true))
end
