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

@testset "Derivative bundles reuse AD results and preserve numeric types" begin
    engine = ActuaryUtilities.FinancialMath._ncurve_ad
    tenors = [1.0, 3.0, 7.0]
    curve(r) = FM.Yield.Constant(FC.Continuous(r))
    depths = Int[]
    tracked(c) = begin
        value = FC.discount(c.base, 2.0)
        push!(depths, _dual_depth(typeof(value)))
        value
    end
    result = engine(tracked, (; base = curve(0.04)), tenors; order = 2)
    @test count(iszero, depths) == 1 # establish the valuation's buffer type
    @test maximum(depths) == 2
    @test 1 ∉ depths # no standalone gradient pass before computing the Hessian
    @test sum(result.gradient.base) ≈ -2result.value
    @test sum(result.hessian.base.base) ≈ 4result.value

    for order in (1, 2)
        result = engine(c -> FC.discount(c.base, 2.0), (; base = curve(big"0.04")), tenors; order)
        @test result.value isa BigFloat
        @test eltype(result.gradient.base) == BigFloat
        @test sum(result.gradient.base) ≈ -2result.value
        if order == 2
            @test eltype(result.hessian.base.base) == BigFloat
            @test sum(result.hessian.base.base) ≈ 4result.value
        end
        # Output types belong to the valuation, not just the supplied curves.
        constant = engine(_ -> big"3.0", (; base = curve(0.04)), tenors; order)
        @test constant.value isa BigFloat
        @test constant.value == big"3.0"
        @test all(iszero, constant.gradient.base)
        if order == 2
            @test all(iszero, constant.hessian.base.base)
        end
        first_order(r) = sum(engine(c -> FC.discount(c.base, 2.0), (; base = curve(r)), tenors; order).gradient.base)
        @test ForwardDiff.derivative(first_order, 0.04) ≈ 4exp(-0.08)
        @test ForwardDiff.derivative(r -> ForwardDiff.derivative(first_order, r), 0.04) ≈ -8exp(-0.08)
    end
    second_order(r) = sum(engine(c -> FC.discount(c.base, 2.0), (; base = curve(r)), tenors; order = 2).hessian.base.base)
    @test ForwardDiff.derivative(second_order, 0.04) ≈ -8exp(-0.08)

    # Internal views must not make the public normalized blocks alias each other.
    bundle = sensitivities(KeyRates(tenors), (b, c) -> FC.discount(b, 2.0) * FC.discount(c, 2.0), curve(0.03), curve(0.01))
    credit_block = copy(bundle.convexities.credit)
    cross_block = copy(bundle.convexities.cross)
    fill!(bundle.convexities.base, NaN)
    @test bundle.convexities.credit == credit_block
    @test bundle.convexities.cross == cross_block
end
