struct GradientOnlyContract <: FC.AbstractContract
    depths::Vector{Int}
end
_dual_depth(::Type) = 0
_dual_depth(::Type{ForwardDiff.Dual{Tag, V, N}}) where {Tag, V, N} = 1 + _dual_depth(V)
function FC.present_value(ctx, c::GradientOnlyContract)
    v = FC.discount(ctx, 1.0) + FC.discount(ctx, 4.0)
    push!(c.depths, _dual_depth(typeof(v)))
    return v
end

_npartials(::Type{<:ForwardDiff.Dual{T, V, N}}) where {T, V, N} = N
_npartials(::Type) = 0

@testset "Shared sensitivity engine" begin
    tenors = [1.0, 3.0, 7.0]
    kr = KeyRates(tenors)
    base = FM.Yield.Constant(FC.Continuous(0.03))
    credit = FM.Yield.Constant(FC.Continuous(0.01))
    v(c) = 5FC.discount(c, 2.0) + 105FC.discount(c, 6.0)
    v2(b, c) = v(b + c)
    single = sensitivities(v, SecondOrder(), kr, base)
    named = sensitivities(c -> v(c.base), SecondOrder(), kr, (; base))
    @test named.value ≈ single.value
    @test named.duration.base ≈ single.duration
    @test named.dv01.base ≈ single.dv01
    @test named.convexity.base.base ≈ single.convexity
    @test sensitivities(c -> v(c.base), KeyRates(Real[1, 3.0, 7]), (; base)).duration.base ≈ single.duration
    pair = sensitivities(v2, SecondOrder(), kr, base, credit)
    @test _same_sensitivity(pair, sensitivities(c -> v2(c.base, c.credit), SecondOrder(), kr, (; base, credit)))
    engine = ActuaryUtilities.FinancialMath._curve_ad
    r = engine(c -> v2(c.base, c.credit), (; base, credit), tenors, SecondOrder())
    @test r.hessian.base.base ./ r.value ≈ pair.convexity.base.base
    @test r.hessian.credit.credit ./ r.value ≈ pair.convexity.credit.credit
    @test r.hessian.base.credit ./ r.value ≈ pair.convexity.base.credit
    @test r.hessian.credit.base ≈ transpose(r.hessian.base.credit)
    @test pair.convexity.credit.base ≈ transpose(pair.convexity.base.credit)
    @test _same_sensitivity(convexity(v2, kr, base, credit), pair.convexity)
    three = engine(c -> v(c.base + c.credit + c.liquidity), (; base, credit, liquidity = credit), tenors, SecondOrder())
    @test three.gradient.base ≈ three.gradient.liquidity
    @test three.hessian.base.credit ≈ three.hessian.liquidity.credit

    # Without a grid, each role takes one parallel shock: the derivatives are numbers, ≈ the sums
    # of the key-rate ones, and the valuation sees one partial per role, not one per tenor.
    parallel = sensitivities(v2, SecondOrder(), base, credit)
    @test parallel.dv01.base ≈ sum(pair.dv01.base) && parallel.dv01.credit ≈ sum(pair.dv01.credit)
    @test parallel.duration.credit ≈ sum(pair.duration.credit)
    @test parallel.convexity.base.credit ≈ sum(pair.convexity.base.credit)
    @test convexity(v2, base, credit) == parallel.convexity
    @test duration(v2, IR01(), base, credit) ≈ parallel.dv01.base
    widths = Int[]
    counted(b, c) = (x = v2(b, c); push!(widths, _npartials(typeof(x))); x)
    sensitivities(counted, base, credit)
    @test maximum(widths) == 2
    empty!(widths)
    sensitivities(c -> counted(c, credit), base)
    @test maximum(widths) == 1
    empty!(widths)
    sensitivities(c -> counted(c, credit), kr, base)
    @test maximum(widths) == length(tenors)

    # First order takes no second derivatives; second order takes the value and both derivatives
    # from one pass, after the evaluation that sets the result type.
    depths = Int[]
    deep(c) = (x = v(c); push!(depths, _dual_depth(typeof(x))); x)
    for (order, depth) in ((FirstOrder(), 1), (SecondOrder(), 2))
        for curves in ((base,), (kr, base))
            empty!(depths)
            sensitivities(deep, order, curves...)
            @test maximum(depths) == depth
            @test length(depths) == 2
        end
    end

    c = GradientOnlyContract(Int[])
    bundle = sensitivities(kr, base, c)
    @test maximum(c.depths) == 1
    @test keys(bundle.dv01) == (:discount, :index)
    parallel = sensitivities(base, c)
    @test maximum(c.depths) == 1
    for (metric, dur, dollars, key_rate) in (
            (Effective(), parallel.duration.discount + parallel.duration.index, parallel.dv01.discount + parallel.dv01.index, bundle.duration.discount .+ bundle.duration.index),
            (Spread(), parallel.duration.discount, parallel.dv01.discount, bundle.duration.discount),
        )
        empty!(c.depths)
        @test duration(metric, base, c) ≈ dur
        @test duration(DV01(), metric, base, c) ≈ dollars
        @test duration(metric, kr, base, c) ≈ key_rate
        @test sum(key_rate) ≈ dur
        @test maximum(c.depths) == 1
    end
    floater = FM.Bond.Floating(0.005, FC.Periodic(2), 5.0, :index)
    sb = sensitivities(base + credit, floater; index = base)
    @test duration(Effective(), base + credit, floater; index = base) ≈ sb.duration.discount + sb.duration.index atol = 1.0e-12
    @test duration(Spread(), base + credit, floater; index = base) ≈ sb.duration.discount
    @test duration(DV01(), Effective(), base + credit, floater; index = base) ≈ sb.dv01.discount + sb.dv01.index atol = 1.0e-12

    for bad in (Float64[], [2.0, 1.0], [1.0, 1.0], [0.0, 1.0], [1.0, Inf], [1.0, NaN])
        @test_throws ArgumentError KeyRates(bad)
    end
    # A grid mutated after construction is caught where it is used.
    grid = KeyRates(copy(tenors))
    grid.tenors[2] = grid.tenors[1]
    @test_throws ArgumentError duration(v, grid, base)
    @test_throws ArgumentError sensitivities(c -> v(c.base), grid, (; base))
    @test_throws ArgumentError sensitivities(grid, base, floater)
    @test_throws ArgumentError duration(Effective(), grid, base, floater)
end

@testset "Derivative bundles reuse AD results and preserve numeric types" begin
    engine = ActuaryUtilities.FinancialMath._curve_ad
    tenors = [1.0, 3.0, 7.0]
    curve(r) = FM.Yield.Constant(FC.Continuous(r))
    depths = Int[]
    tracked(c) = begin
        value = FC.discount(c.base, 2.0)
        push!(depths, _dual_depth(typeof(value)))
        value
    end
    result = engine(tracked, (; base = curve(0.04)), tenors, SecondOrder())
    @test count(iszero, depths) == 1 # establish the valuation's buffer type
    @test maximum(depths) == 2
    @test 1 ∉ depths # no standalone gradient pass before computing the Hessian
    @test sum(result.gradient.base) ≈ -2result.value
    @test sum(result.hessian.base.base) ≈ 4result.value

    for order in (FirstOrder(), SecondOrder()), grid in (tenors, nothing)
        result = engine(c -> FC.discount(c.base, 2.0), (; base = curve(big"0.04")), grid, order)
        @test result.value isa BigFloat
        @test eltype(result.gradient.base) == BigFloat
        @test sum(result.gradient.base) ≈ -2result.value
        if order isa SecondOrder
            @test eltype(result.hessian.base.base) == BigFloat
            @test sum(result.hessian.base.base) ≈ 4result.value
        end
        # Allocate results using the valuation's numeric type.
        constant = engine(_ -> big"3.0", (; base = curve(0.04)), grid, order)
        @test constant.value isa BigFloat
        @test constant.value == big"3.0"
        @test all(iszero, constant.gradient.base)
        if order isa SecondOrder
            @test all(iszero, constant.hessian.base.base)
        end
        first_order(r) = sum(engine(c -> FC.discount(c.base, 2.0), (; base = curve(r)), grid, order).gradient.base)
        @test ForwardDiff.derivative(first_order, 0.04) ≈ 4exp(-0.08)
        @test ForwardDiff.derivative(r -> ForwardDiff.derivative(first_order, r), 0.04) ≈ -8exp(-0.08)
    end
    second_order(r) = sum(engine(c -> FC.discount(c.base, 2.0), (; base = curve(r)), tenors, SecondOrder()).hessian.base.base)
    @test ForwardDiff.derivative(second_order, 0.04) ≈ -8exp(-0.08)

    # Public blocks must own independent arrays.
    bundle = sensitivities((b, c) -> FC.discount(b, 2.0) * FC.discount(c, 2.0), SecondOrder(), KeyRates(tenors), curve(0.03), curve(0.01))
    credit_block = copy(bundle.convexity.credit.credit)
    cross_block = copy(bundle.convexity.base.credit)
    duration_block = copy(bundle.duration.credit)
    fill!(bundle.convexity.base.base, NaN)
    fill!(bundle.duration.base, NaN)
    fill!(bundle.dv01.base, NaN)
    @test bundle.convexity.credit.credit == credit_block
    @test bundle.convexity.base.credit == cross_block
    @test bundle.duration.credit == duration_block
    @test all(isfinite, bundle.dv01.credit)
end
