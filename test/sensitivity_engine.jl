struct GradientOnlyContract <: FC.AbstractContract
    depths::Vector{Int}
end
_dual_depth(::Type) = 0
_dual_depth(::Type{ForwardDiff.Dual{Tag, V, N}}) where {Tag, V, N} = 1 + _dual_depth(V)
function FC.present_value(curve::FM.Yield.AbstractYieldModel, c::GradientOnlyContract)
    v = FC.discount(curve, 1.0) + FC.discount(curve, 4.0)
    push!(c.depths, _dual_depth(typeof(v)))
    return v
end

@testset "Shared sensitivity engine" begin
    tenors = [1.0, 3.0, 7.0]
    kr = KeyRates(tenors)
    base = FM.Yield.Constant(FC.Continuous(0.03))
    credit = FM.Yield.Constant(FC.Continuous(0.01))
    v(c) = 5FC.discount(c, 2.0) + 105FC.discount(c, 6.0)
    v2(b, c) = v(b + c)
    single = sensitivities(kr, v, base)
    named = sensitivities(c -> v(c.base), (; base); tenors)
    @test named.value ≈ single.value
    @test named.key_rate.base ≈ single.durations
    @test sensitivities(c -> v(c.base), (; base); tenors = Real[1, 3.0, 7]).key_rate.base ≈ single.durations
    pair = sensitivities(kr, v2, base, credit)
    named_pair = sensitivities(c -> v2(c.base, c.credit), (; base, credit); tenors)
    @test pair.base_durations ≈ named_pair.key_rate.base
    @test pair.credit_durations ≈ named_pair.key_rate.credit
    engine = ActuaryUtilities.FinancialMath._ncurve_ad
    r = engine(c -> v2(c.base, c.credit), (; base, credit), tenors; order = 2)
    @test r.hessian.base.base ./ r.value ≈ pair.convexities.base
    @test r.hessian.credit.credit ./ r.value ≈ pair.convexities.credit
    @test r.hessian.base.credit ./ r.value ≈ pair.convexities.cross
    @test r.hessian.credit.base ≈ transpose(r.hessian.base.credit)
    three = engine(c -> v(c.base + c.credit + c.liquidity), (; base, credit, liquidity = credit), tenors; order = 2)
    @test three.gradient.base ≈ three.gradient.liquidity
    @test three.hessian.base.credit ≈ three.hessian.liquidity.credit

    c = GradientOnlyContract(Int[])
    bundle = sensitivities(c, base, tenors)
    @test maximum(c.depths) == 1
    for (metric, dur, dollars) in (
            (Effective(), bundle.effective_duration, bundle.effective_dv01),
            (Spread(), bundle.spread_duration, bundle.spread_dv01),
        )
        empty!(c.depths)
        @test duration(metric, c, base, tenors) ≈ dur
        @test dv01(metric, c, base, tenors) ≈ dollars
        @test maximum(c.depths) == 1
    end
    floater = FM.Bond.Floating(0.005, FC.Periodic(2), 5.0, :index)
    sb = sensitivities(floater, base, base + credit, tenors)
    @test duration(Effective(), floater, base, base + credit, tenors) ≈ sb.effective_duration atol = 1.0e-12
    @test duration(Spread(), floater, base, base + credit, tenors) ≈ sb.spread_duration
    @test dv01(Effective(), floater, base, base + credit, tenors) ≈ sb.effective_dv01 atol = 1.0e-12

    for bad in (Float64[], [2.0, 1.0], [1.0, 1.0], [0.0, 1.0], [1.0, Inf], [1.0, NaN])
        @test_throws ArgumentError KeyRates(bad)
        @test_throws ArgumentError sensitivities(c -> v(c.base), (; base); tenors = bad)
        @test_throws ArgumentError sensitivities(floater, base, bad)
    end
    @test_throws ArgumentError engine(identity, (;), tenors)
    @test_throws ArgumentError engine(identity, (; base = 0.03), tenors)
    @test_throws ArgumentError engine(c -> v(c.base), (; base), tenors; order = 3)
    grid = KeyRates(copy(tenors))
    grid.tenors[2] = grid.tenors[1]
    @test_throws ArgumentError duration(grid, v, base)
end
