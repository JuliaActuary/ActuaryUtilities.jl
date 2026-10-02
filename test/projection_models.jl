@testset "contract measures value in FinanceModels valuation contexts" begin
    index = FM.Yield.Spline(FM.Spline.Linear(), [1.0, 2.0, 5.0, 10.0], [0.02, 0.025, 0.04, 0.045])
    credit = FM.Yield.Constant(FC.Continuous(0.06))
    tenors = [1.0, 2.0, 5.0, 10.0]
    fixed = FM.Bond.Fixed(0.05, FC.Periodic(2), 5.0)
    floater = FM.Bond.Floating(0.005, FC.Periodic(2), 5.0, :index)
    # A swap's floating leg is a transformed contract.
    swap = FM.InterestRateSwap(index, 5.0; frequency = 4, model_key = :index)
    forward = FM.Forward(1.0, floater)
    portfolio = [fixed, swap, forward]
    for c in (floater, swap, forward, portfolio)
        explicit(cs) = FC.pv(FM.Models(cs.credit, Dict(:index => cs.index)), c)
        reference = sensitivities(KeyRates(tenors), explicit, (; index, credit))
        actual = sensitivities(KeyRates(tenors), c, index, credit)
        @test actual.value ≈ reference.value
        @test actual.forward_key_rate ≈ reference.key_rate.index
        @test actual.spread_key_rate ≈ reference.key_rate.credit
        @test actual.effective_dv01 ≈ reference.dv01.index + reference.dv01.credit atol = 1.0e-12
    end
    for c in (floater, swap, forward)
        spread = 0.007
        target = FC.pv(FM.Models(credit + FC.Continuous(spread); index), c)
        solved = zspread(c, credit, target; forward = index)
        @test solved.zspread ≈ spread atol = 1.0e-10
        @test solved.zspread_dv01 ≈
            -ForwardDiff.derivative(s -> FC.pv(FM.Models(credit + FC.Continuous(s); index), c), spread) / 10_000
    end
    # One curve cannot supply an FX model.
    converted = FM.FX.Converted(floater, FM.FX.Pair(:EUR, :USD), :fx)
    @test_throws (VERSION >= v"1.12" ? FieldError : ErrorException) duration(Effective(), converted, index)
end

# Pays 1 at t = 1 and t = 4, valued in closed form without cashflows.
struct ClosedFormPair <: FC.AbstractContract end
FC.present_value(ctx, ::ClosedFormPair) = FC.discount(ctx, 1.0) + FC.discount(ctx, 4.0)
struct UnvaluedContract <: FC.AbstractContract end

@testset "closed-form contracts keep their own value" begin
    curve = FM.Yield.Constant(FC.Continuous(0.03))
    d1, d4 = exp(-0.03), exp(-0.12)
    c = ClosedFormPair()
    @test duration(Effective(), c, curve) ≈ (d1 + 4d4) / (d1 + d4)
    @test duration(Spread(), c, curve) ≈ (d1 + 4d4) / (d1 + d4)
    @test convexity(Effective(), c, curve) ≈ (d1 + 16d4) / (d1 + d4)
    @test sensitivities(KeyRates([1.0, 4.0]), c, curve).value ≈ d1 + d4
    # A contract with neither a cashflow projection nor a closed form has no value.
    @test_throws MethodError duration(Effective(), UnvaluedContract(), curve)
end

@testset "composites of closed forms have risk" begin
    curve = FM.Yield.Constant(FC.Continuous(0.03))
    kr = KeyRates([1.0, 4.0])
    pair = ClosedFormPair()
    bond = FM.Bond.Fixed(0.05, FC.Periodic(1), 3.0)
    for (composite, parts) in ((FC.Composite(pair, pair), (pair, pair)), (FC.Composite(pair, bond), (pair, bond)))
        # A composite is worth the sum of its parts, under every bumped curve.
        value(c) = sum(FC.pv(c, p) for p in parts)
        @test duration(Effective(), composite, curve) ≈ duration(curve, value)
        @test duration(Spread(), composite, curve) ≈ duration(curve, value)
        @test convexity(Effective(), composite, curve) ≈ convexity(curve, value)
        @test dv01(composite, curve) ≈ sum(dv01(p, curve) for p in parts)
        s = sensitivities(kr, composite, curve)
        @test s.value ≈ value(curve)
        @test s.effective_key_rate ≈ duration(kr, value, curve)
        @test duration(Effective(), [composite, bond], curve) ≈ duration(curve, c -> value(c) + FC.pv(c, bond))
    end
end

# At t = 2, pays principal plus the index forward rate from t = 1 to 2, valued in closed form
# against the valuation context: it discounts on `ctx` and reads its index curve by key.
struct ClosedFormFloater <: FC.AbstractContract
    key::Symbol
end
FC.present_value(ctx, c::ClosedFormFloater) = FC.discount(ctx, 2.0) / FC.discount(ctx[c.key], 1.0, 2.0)

@testset "a custom floater resets on the index curve" begin
    curve = FM.Yield.Constant(FC.Continuous(0.03))
    floater = ClosedFormFloater(:index)
    reset(c) = FC.pv(FM.Models(c, Dict(:index => c)), floater)
    held(c) = FC.pv(FM.Models(c, Dict(:index => curve)), floater)
    @test duration(Effective(), floater, curve) ≈ duration(curve, reset)
    @test duration(Spread(), floater, curve) ≈ duration(curve, held)
    # Rate risk ends at the reset; spread risk runs to payment.
    @test duration(Effective(), floater, curve) ≈ 1.0
    @test duration(Spread(), floater, curve) ≈ 2.0
    @test sensitivities(KeyRates([1.0, 2.0]), floater, curve).effective_duration ≈ 1.0
end
