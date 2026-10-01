@testset "contract measures project what FinanceModels declares" begin
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
        explicit(cs) = FC.pv(cs.credit, FM.Projection(c, Dict(:index => cs.index)))
        reference = sensitivities(KeyRates(tenors), explicit, (; index, credit))
        actual = sensitivities(KeyRates(tenors), c, index, credit)
        @test actual.value ≈ reference.value
        @test actual.forward_key_rate ≈ reference.key_rate.index
        @test actual.spread_key_rate ≈ reference.key_rate.credit
        @test actual.effective_dv01 ≈ reference.dv01.index + reference.dv01.credit atol = 1.0e-12
    end
    for c in (floater, swap, forward)
        spread = 0.007
        target = FC.pv(credit + FC.Continuous(spread), FM.Projection(c; index))
        solved = zspread(c, credit, target; forward = index)
        @test solved.zspread ≈ spread atol = 1.0e-10
        @test solved.zspread_dv01 ≈
            -ForwardDiff.derivative(s -> FC.pv(credit + FC.Continuous(s), FM.Projection(c; index)), spread) / 10_000
    end
    # One curve cannot supply an FX model.
    converted = FM.FX.Converted(floater, FM.FX.Pair(:EUR, :USD), :fx)
    @test_throws ArgumentError duration(Effective(), converted, index)
end

# Pays 1 at t = 1 and t = 4, valued in closed form without cashflows.
struct ClosedFormPair <: FC.AbstractContract end
FM.model_requirements(::ClosedFormPair) = ()
FC.present_value(m::FM.Yield.AbstractYieldModel, ::ClosedFormPair) = FC.discount(m, 1.0) + FC.discount(m, 4.0)
struct UndeclaredContract <: FC.AbstractContract end

@testset "contracts without projection requirements keep their own value" begin
    curve = FM.Yield.Constant(FC.Continuous(0.03))
    d1, d4 = exp(-0.03), exp(-0.12)
    c = ClosedFormPair()
    @test duration(Effective(), c, curve) ≈ (d1 + 4d4) / (d1 + d4)
    @test duration(Spread(), c, curve) ≈ (d1 + 4d4) / (d1 + d4)
    @test convexity(Effective(), c, curve) ≈ (d1 + 16d4) / (d1 + d4)
    @test sensitivities(KeyRates([1.0, 4.0]), c, curve).value ≈ d1 + d4
    @test_throws ArgumentError duration(Effective(), UndeclaredContract(), curve)
end

# At t = 2, pays principal plus the index forward rate from t = 1 to 2, valued in closed form.
struct ClosedFormFloater <: FC.AbstractContract
    key::Symbol
end
FM.model_requirements(c::ClosedFormFloater) = (c.key => FM.Yield.AbstractYieldModel,)
FC.present_value(m::FM.Yield.AbstractYieldModel, p::FM.Projection{ClosedFormFloater}) =
    FC.discount(m, 2.0) / FC.discount(p.model[p.contract.key], 1.0, 2.0)

@testset "a custom floater resets on the index curve" begin
    curve = FM.Yield.Constant(FC.Continuous(0.03))
    floater = ClosedFormFloater(:index)
    reset(c) = FC.pv(c, FM.Projection(floater, Dict(:index => c)))
    held(c) = FC.pv(c, FM.Projection(floater, Dict(:index => curve)))
    @test duration(Effective(), floater, curve) ≈ duration(curve, reset)
    @test duration(Spread(), floater, curve) ≈ duration(curve, held)
    # Rate risk ends at the reset; spread risk runs to payment.
    @test duration(Effective(), floater, curve) ≈ 1.0
    @test duration(Spread(), floater, curve) ≈ 2.0
end
