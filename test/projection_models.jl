@testset "FinanceModels-owned projection wiring" begin
    index = FM.Yield.Spline(FM.Spline.Linear(), [1.0, 2.0, 5.0, 10.0], [0.02, 0.025, 0.04, 0.045])
    credit = FM.Yield.Constant(FC.Continuous(0.06))
    tenors = [1.0, 2.0, 5.0, 10.0]
    fixed = FM.Bond.Fixed(0.05, FC.Periodic(2), 5.0)
    floater = FM.Bond.Floating(0.005, FC.Periodic(2), 5.0, :index)
    swap = FM.InterestRateSwap(index, 5.0; model_key = :index)
    forward = FM.Forward(1.0, floater)
    # The swap's floating leg is an Eduction; exercise it as a root too.
    transformed = swap.b
    portfolio = [fixed, swap, forward]
    store = Dict(:index => index)
    for c in (fixed, floater, swap, forward, transformed, portfolio)
        projected = reproject(c, index)
        @test projected isa FM.Projection
        @test collect(projected) == collect(FM.Projection(c, store))
        @test FC.pv(credit, projected) ≈ FC.pv(credit, FM.Projection(c, store))
        @test reproject(c, store).model === store
    end
    for c in (floater, swap, forward, portfolio)
        explicit(cs) = FC.pv(cs.credit, FM.Projection(c, Dict(:index => cs.index)))
        reference = sensitivities(explicit, (; index, credit); tenors)
        actual = sensitivities(c, index, credit, tenors)
        @test actual.value ≈ reference.value
        @test actual.forward_key_rate ≈ reference.key_rate.index
        @test actual.spread_key_rate ≈ reference.key_rate.credit
        @test actual.effective_dv01 ≈ reference.dv01.index + reference.dv01.credit atol = 1.0e-12
    end
    for c in (floater, swap, forward)
        spread = 0.007
        disc = credit + FC.Continuous(spread)
        target = FC.pv(disc, FM.Projection(c, store))
        solved = zspread(c, credit, target; forward = index)
        @test solved.zspread ≈ spread atol = 1.0e-10
        @test solved.zspread_dv01 ≈
            -ForwardDiff.derivative(s -> FC.pv(credit + FC.Continuous(s), FM.Projection(c, store)), spread) / 10_000
    end

    other = FM.Bond.Floating(0.0, FC.Periodic(2), 5.0, :other)
    pair = FC.Composite(floater, other)
    models = Dict(:index => index, :other => credit)
    @test FC.pv(credit, reproject(pair, models)) ≈
        FC.pv(credit, FM.Projection(floater, models)) + FC.pv(credit, FM.Projection(other, models))
    fx_pair = FM.FX.Pair(:EUR, :USD)
    converted = FM.FX.Converted(floater, fx_pair, :fx)
    fx = FM.FX.Forwards(fx_pair, 1.08, credit, index)
    @test_throws "explicit model store" reproject(converted, index)
    fx_models = Dict(:index => index, :fx => fx)
    @test FC.pv(credit, reproject(converted, fx_models)) ≈ 1.08 * FC.pv(index, reproject(floater, index))
end
