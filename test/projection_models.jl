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
        explicit(cs) = FC.pv(FM.Models(cs.discount, Dict(:index => cs.index)), c)
        for order in (FirstOrder(), SecondOrder()), grid in ((), (KeyRates(tenors),))
            reference = sensitivities(explicit, order, grid..., (; discount = credit, index))
            @test _same_sensitivity(sensitivities(order, grid..., credit, c; index), reference)
        end
        @test duration(DV01(), Effective(), credit, c; index) ≈ sum(values(sensitivities(credit, c; index).dv01)) atol = 1.0e-12
    end
    for c in (floater, swap, forward)
        spread = 0.007
        target = FC.pv(FM.Models(credit + FC.Continuous(spread); index), c)
        solved = zspread(credit, c, target; index)
        @test FC.rate(solved.zspread) ≈ spread atol = 1.0e-10
        @test solved.zspread_dv01 ≈
            -ForwardDiff.derivative(s -> FC.pv(FM.Models(credit + FC.Continuous(s); index), c), spread) / 10_000
    end
    # One curve cannot supply an FX model.
    converted = FM.FX.Converted(floater, FM.FX.Pair(:EUR, :USD), :fx)
    @test_throws (VERSION >= v"1.12" ? FieldError : ErrorException) duration(Effective(), index, converted)
end

# Pays 1 at t = 1 and t = 4, valued in closed form without cashflows.
struct ClosedFormPair <: FC.AbstractContract end
FC.present_value(ctx, ::ClosedFormPair) = FC.discount(ctx, 1.0) + FC.discount(ctx, 4.0)
struct UnvaluedContract <: FC.AbstractContract end

@testset "closed-form contracts keep their own value" begin
    curve = FM.Yield.Constant(FC.Continuous(0.03))
    d1, d4 = exp(-0.03), exp(-0.12)
    c = ClosedFormPair()
    @test duration(Effective(), curve, c) ≈ (d1 + 4d4) / (d1 + d4)
    @test duration(Spread(), curve, c) ≈ (d1 + 4d4) / (d1 + d4)
    @test convexity(Effective(), curve, c) ≈ (d1 + 16d4) / (d1 + d4)
    @test convexity(Spread(), curve, c) ≈ (d1 + 16d4) / (d1 + d4)
    @test sensitivities(KeyRates([1.0, 4.0]), curve, c).value ≈ d1 + d4
    @test sensitivities(SecondOrder(), curve, c).convexity.discount.discount ≈ (d1 + 16d4) / (d1 + d4)
    # A contract with neither a cashflow projection nor a closed form has no value.
    @test_throws MethodError duration(Effective(), curve, UnvaluedContract())
end

@testset "composites of closed forms have risk" begin
    curve = FM.Yield.Constant(FC.Continuous(0.03))
    kr = KeyRates([1.0, 4.0])
    pair = ClosedFormPair()
    bond = FM.Bond.Fixed(0.05, FC.Periodic(1), 3.0)
    for (composite, parts) in ((FC.Composite(pair, pair), (pair, pair)), (FC.Composite(pair, bond), (pair, bond)))
        # A composite is worth the sum of its parts, under every bumped curve.
        value(c) = sum(FC.pv(c, p) for p in parts)
        @test duration(Effective(), curve, composite) ≈ duration(value, curve)
        @test duration(Spread(), curve, composite) ≈ duration(value, curve)
        @test convexity(Effective(), curve, composite) ≈ convexity(value, curve)
        @test duration(DV01(), curve, composite) ≈ sum(duration(DV01(), curve, p) for p in parts)
        s = sensitivities(kr, curve, composite)
        @test s.value ≈ value(curve)
        @test s.duration.discount .+ s.duration.index ≈ duration(value, kr, curve)
        @test duration(Effective(), curve, [composite, bond]) ≈ duration(c -> value(c) + FC.pv(c, bond), curve)
    end
end

@testset "Hull–White closed forms keep their pricing under bumps" begin
    curve = FM.Yield.Constant(FC.Continuous(0.03))
    hw = FM.ShortRate.HullWhite(0.1, 0.01, curve)
    kr = KeyRates([1.0, 2.0, 3.0, 6.0])
    # A bump moves the curve the model is calibrated to; mean reversion and volatility stay fixed.
    on(c) = FM.ShortRate.HullWhite(hw.a, hw.σ, c)
    cap, swaption = FM.Option.Cap(0.03, 4, 3.0), FM.Option.Swaption(1.0, 6.0, 0.035, 1)
    for target in (cap, swaption)
        value(c) = FC.pv(on(c), target)
        @test duration(Effective(), hw, target) ≈ duration(value, curve)
        @test duration(DV01(), hw, target) ≈ duration(value, DV01(), curve)
        @test duration(Effective(), hw, FC.Composite(target, target)) ≈ duration(value, curve)
        s = sensitivities(kr, hw, target)
        @test s.duration.discount .+ s.duration.index ≈ duration(value, kr, curve)
    end
    @test convexity(Effective(), hw, cap) ≈ convexity(c -> FC.pv(on(c), cap), curve)
    # FinanceModels differentiates the swaption's critical rate to first order only, so second-order
    # risk throws rather than falling back to first order.
    @test_throws "first-order ForwardDiff derivatives only" convexity(Effective(), hw, swaption)
    @test_throws "first-order ForwardDiff derivatives only" sensitivities(SecondOrder(), hw, swaption)
    @test_throws "first-order ForwardDiff derivatives only" sensitivities(SecondOrder(), kr, hw, swaption)
    @test_throws "first-order ForwardDiff derivatives only" convexity(DollarConvexity(), hw, swaption)
    @test_throws "first-order ForwardDiff derivatives only" convexity(DollarConvexity(), Spread(), kr, hw, swaption)
    @test sensitivities(hw, swaption).dv01.discount ≈ duration(DV01(), Spread(), hw, swaption)
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
    @test duration(Effective(), curve, floater) ≈ duration(reset, curve)
    @test duration(Spread(), curve, floater) ≈ duration(held, curve)
    # Rate risk ends at the reset; spread risk runs to payment.
    @test duration(Effective(), curve, floater) ≈ 1.0
    @test duration(Spread(), curve, floater) ≈ 2.0
    s = sensitivities(curve, floater)
    @test s.duration.discount + s.duration.index ≈ 1.0
    @test s.duration.discount ≈ 2.0
    @test s.duration.index ≈ -1.0
end
