@testset "Zero cashflow streams" begin
    curve = ZeroCashflowTestCurve()
    tenors = [1.0, 3.0, 7.0]
    kr = KeyRates(tenors)
    z, zz = zeros(3), zeros(3, 3)
    kernel = ActuaryUtilities.FinancialMath._keyrate_analytic
    positive_zero(x) = isequal(x, zero(x))

    @testset "Scalar measures and legacy key rates" begin
        for (cfs, times) in (
                (Float64[], Float64[]), ([], []),
                (Float64[], [1.0, 2.0]),
                ([0.0, -0.0], [1.0, 2.0]),
                ([0.0, -0.0], [1.0, 2.0, NaN]),
                (FC.Cashflow{Float64, Float64}[], Float64[]),
                (FC.Cashflow[], Float64[]),
                (FC.Cashflow.([0.0, -0.0], [1.0, 2.0]), [1.0, 2.0]),
            )
            for yield in (0.04, FC.Periodic(0.04, 2), FC.Continuous(0.04), FM.Yield.Constant(0.04), curve)
                @test positive_zero(duration(yield, cfs, times))
                @test positive_zero(duration(yield, cfs))
                for measure in (Macaulay(), Modified(), DV01())
                    @test positive_zero(duration(measure, yield, cfs, times))
                    @test positive_zero(duration(measure, yield, cfs))
                end
                @test positive_zero(convexity(yield, cfs, times))
                @test positive_zero(convexity(yield, cfs))
            end
            @test isequal(present_values(curve, cfs, times), zeros(length(cfs)))
        end
    end

    @testset "Key-rate results" begin
        for (cfs, times) in ((Float64[], Float64[]), (Float64[], [1.0, 2.0]), ([0.0, -0.0], [0.0, 2.0]), ([0.0, -0.0], [0.0, 2.0, NaN]))
            @test isequal(duration(kr, curve, cfs, times), z)
            @test isequal(duration(DV01(), kr, curve, cfs, times), z)
            @test isequal(convexity(kr, curve, cfs, times), zz)
            @test positive_zero(duration(curve, cfs, times))
            @test positive_zero(convexity(curve, cfs, times))
            @test isequal(sensitivities(kr, curve, cfs, times), (; value = 0.0, duration = z, dv01 = z))
            @test isequal(sensitivities(SecondOrder(), kr, curve, cfs, times), (; value = 0.0, duration = z, dv01 = z, convexity = zz, dollar_convexity = zz))
            @test isequal(sensitivities(curve, cfs, times), (; value = 0.0, duration = 0.0, dv01 = 0.0))
            @test isequal(sensitivities(SecondOrder(), curve, cfs, times), (; value = 0.0, duration = 0.0, dv01 = 0.0, convexity = 0.0, dollar_convexity = 0.0))
        end
        cfs = FC.Cashflow.([0.0, -0.0], [0.0, 2.0])
        @test isequal(duration(kr, curve, cfs), z)
        @test isequal(sensitivities(SecondOrder(), kr, curve, cfs), (; value = 0.0, duration = z, dv01 = z, convexity = zz, dollar_convexity = zz))
    end

    @testset "Input types and validation" begin
        for T in (Float32, BigFloat)
            for cfs in (T[], zeros(T, 2))
                times = T.(eachindex(cfs))
                @test duration(curve, cfs, times) isa T
                @test convexity(curve, cfs, times) isa T
                @test eltype(present_values(curve, cfs, times)) == T
                typed = sensitivities(SecondOrder(), KeyRates(T.(tenors)), curve, cfs, times)
                @test typed.value isa T
                @test eltype(typed.duration) == eltype(typed.dv01) == eltype(typed.convexity) == T
                parallel = sensitivities(SecondOrder(), curve, cfs, times)
                @test parallel.value isa T && parallel.duration isa T && parallel.dv01 isa T && parallel.convexity isa T
            end
        end
        @test duration(curve, FC.Cashflow{BigFloat, Float64}[]) isa BigFloat
        @test duration(curve, Real[0, big"0.0"], [1.0, 2.0]) isa BigFloat
        # a zero stream's result has the type a nonempty stream's would: the curve counts
        bigcurve = FM.Yield.Constant(FC.Continuous(big"0.04"))
        @test sensitivities(kr, bigcurve, zeros(2), [1.0, 2.0]).value isa BigFloat
        @test sensitivities(kr, bigcurve, [1.0, 0.0], [1.0, 2.0]).value isa BigFloat
        for measure in (Macaulay(), Modified(), DV01())
            @test_throws DimensionMismatch duration(measure, curve, [0.0], Float64[])
        end
        @test_throws DimensionMismatch convexity(curve, [0.0], Float64[])
        @test_throws DimensionMismatch present_values(curve, [0.0], Float64[])
        @test_throws DimensionMismatch sensitivities(kr, curve, [0.0], Float64[])
        @test_throws DimensionMismatch sensitivities(curve, [0.0], Float64[])
        @test_throws ArgumentError KeyRates(Float64[])
        @test_throws ArgumentError KeyRates([3.0, 1.0])
    end

    @testset "Zero amounts differ from zero net value" begin
        flat = FM.Yield.Constant(FC.Continuous(0.0))
        cfs, times = [100.0, -100.0], [1.0, 2.0]
        raw = kernel(flat, tenors, cfs, times, SecondOrder())
        result = sensitivities(SecondOrder(), kr, flat, cfs, times)
        @test iszero(raw.value)
        @test any(!iszero, raw.gradient) && any(!iszero, raw.hessian)
        @test any(!isfinite, result.duration) && any(!isfinite, result.convexity)
        @test duration(DV01(), kr, flat, cfs, times) == -raw.gradient ./ 10_000 == result.dv01
        @test !isfinite(duration(Macaulay(), flat, cfs, times))
        @test !isfinite(convexity(flat, cfs, times))
        # A zero-value position with dollar risk: the dollar measures are defined, the normalized not.
        parallel = sensitivities(SecondOrder(), flat, cfs, times)
        @test iszero(parallel.value)
        @test parallel.dv01 ≈ duration(DV01(), flat, cfs, times) ≈ -100 * exp(0.0) / 10_000
        @test !isfinite(parallel.duration) && !isfinite(parallel.convexity)
        tiny = sensitivities(kr, flat, [1.0e-200], [2.0])
        @test tiny.value == 1.0e-200
        @test sum(tiny.duration) ≈ 2.0
        # Valuation functions cannot be classified as zero streams from PV alone.
        @test all(isnan, duration(_ -> 0.0, kr, flat))
    end

    @testset "Automatic differentiation" begin
        flat = FM.Yield.Constant(FC.Continuous(0.04))
        # A zero primal amount with a nonzero partial still carries exposure.
        dollar(x) = sum(duration(DV01(), kr, flat, [x, zero(x)], [2.0, 3.0]))
        @test ForwardDiff.derivative(dollar, 0.0) ≈ 2 * FC.discount(flat, 2.0) / 10_000
        value(x) = kernel(flat, tenors, [x], [2.0], FirstOrder()).value
        @test ForwardDiff.derivative(value, 0.0) ≈ FC.discount(flat, 2.0)
        squared_value(x) = value(x * x)
        @test ForwardDiff.derivative(x -> ForwardDiff.derivative(squared_value, x), 0.0) ≈ 2 * FC.discount(flat, 2.0)
        zero_krd(rs) = duration(kr, FM.ZeroRateCurve(rs, tenors), zeros(2), [0.0, 2.0])
        @test ForwardDiff.jacobian(zero_krd, [0.02, 0.03, 0.04]) == zz
    end

    @testset "Hull-White skips simulation" begin
        hw = FM.ShortRate.HullWhite(0.1, 0.01, FM.Yield.Constant(0.04))
        for (cfs, times) in ((Float64[], Float64[]), (Float64[], [1.0, 2.0]), (zeros(2), [1.0, 2.0]), (zeros(2), [1.0, 2.0, NaN]))
            rng = MersenneTwister(123)
            s = Scenarios(hw; horizon = 3.0, rng)
            untouched = copy(rng)
            @test isequal(sensitivities(kr, s, cfs, times), (; value = 0.0, duration = z, dv01 = z))
            @test isequal(sensitivities(SecondOrder(), kr, s, cfs, times), (; value = 0.0, duration = z, dv01 = z, convexity = zz, dollar_convexity = zz))
            @test isequal(sensitivities(s, cfs, times), (; value = 0.0, duration = 0.0, dv01 = 0.0))
            @test rand(rng) == rand(untouched)
        end
    end
end

@testset "Trailing times are unused" begin
    cfs, times = [5.0, 105.0], [1.0, 2.0]
    extra = [1.0, 2.0, 50.0, NaN]
    tenors = [1.0, 2.0, 3.0]
    kr = KeyRates(tenors)
    curve = FM.ZeroRateCurve([0.02, 0.03, 0.04], tenors)
    flat = FM.Yield.Constant(FC.Continuous(0.04))
    for yield in (0.04, FC.Periodic(0.04, 2), FC.Continuous(0.04), flat, curve)
        for measure in (Macaulay(), Modified(), DV01())
            @test duration(measure, yield, cfs, extra) ≈ duration(measure, yield, cfs, times)
            @test_throws DimensionMismatch duration(measure, yield, cfs, [1.0])
        end
        @test convexity(yield, cfs, extra) ≈ convexity(yield, cfs, times)
        @test_throws DimensionMismatch convexity(yield, cfs, [1.0])
        @test present_values(yield, cfs, extra) ≈ present_values(yield, cfs, times)
    end
    for args in ((kr, curve), (SecondOrder(), kr, curve), (curve,), (SecondOrder(), flat))
        @test isequal(sensitivities(args..., cfs, extra), sensitivities(args..., cfs, times))
        @test_throws DimensionMismatch sensitivities(args..., cfs, [1.0])
    end
    @test convexity(kr, curve, cfs, extra) == convexity(kr, curve, cfs, times)
    @test convexity(curve, cfs, extra) == convexity(curve, cfs, times)

    # A trailing time, even past the horizon, is not valued and does not change RNG use.
    hw = FM.ShortRate.HullWhite(0.1, 0.01, flat)
    for m in ((), (SecondOrder(),))
        rng_short, rng_long = MersenneTwister(42), MersenneTwister(42)
        short = sensitivities(m..., kr, Scenarios(hw; n_scenarios = 8, timestep = 0.25, horizon = 3.0, rng = rng_short), cfs, times)
        long = sensitivities(m..., kr, Scenarios(hw; n_scenarios = 8, timestep = 0.25, horizon = 3.0, rng = rng_long), cfs, extra)
        @test isequal(short, long)
        @test rand(rng_short) == rand(rng_long)
        @test_throws DimensionMismatch sensitivities(m..., kr, Scenarios(hw; horizon = 3.0), cfs, [1.0])
    end
end

@testset "Zero streams take the type of a nonempty result" begin
    # Linearity forces the value zero. Its type is a convention: the type the same measure
    # gives for a nonempty stream of the same amount, time and rate types. When the element
    # type says nothing (`Any[]`, `Cashflow[]`, `()`), the rate or curve sets it.
    kr = KeyRates([1.0, 5.0])
    yields = (
        0.03, 0.03f0, big"0.03", FC.Periodic(0.03f0, 1), FC.Continuous(0.03f0), FC.Periodic(0.03, 2),
        FM.Yield.Constant(0.03), FM.Yield.Constant(big"0.03"),
    )
    for y in yields, amounts in ([1.0, 2.0], Float32[1, 2], [1, 2]), times in (Float32[1, 2], [1.0, 2.0])
        measures = Any[
            (a, t) -> duration(Macaulay(), y, a, t), (a, t) -> duration(y, a, t),
            (a, t) -> duration(DV01(), y, a, t), (a, t) -> convexity(y, a, t), (a, t) -> present_values(y, a, t),
        ]
        push!(measures, (a, t) -> sensitivities(SecondOrder(), y, a, t))
        if y isa FM.Yield.AbstractYieldModel
            push!(measures, (a, t) -> duration(kr, y, a, t), (a, t) -> duration(DV01(), kr, y, a, t), (a, t) -> sensitivities(SecondOrder(), kr, y, a, t))
        end
        for m in measures
            T = typeof(m(amounts, times))
            @test typeof(m(zero(amounts), times)) == T
            @test typeof(m(similar(amounts, 0), similar(times, 0))) == T
        end
    end
    for (y, T) in ((0.03, Float64), (big"0.03", BigFloat), (FC.Continuous(0.03f0), Float32), (FM.Yield.Constant(big"0.03"), BigFloat))
        for cfs in (Any[], FC.Cashflow[], ())
            @test duration(Macaulay(), y, cfs) isa T
            @test isequal(duration(Macaulay(), y, cfs), zero(T))
        end
    end
    @test duration(kr, FM.Yield.Constant(big"0.03"), Any[], Float64[]) isa Vector{BigFloat}
    # the zero is exact even where the origin query is not finite
    @test isequal(duration(Macaulay(), FC.Continuous(Inf), Float64[], Float64[]), 0.0)

    # a dual-number rate gives dual zeros, with a zero derivative
    types = Ref{Any}()
    d = ForwardDiff.derivative(0.03) do r
        types[] = (typeof(duration(DV01(), r, [1.0, 2.0], [1.0, 2.0])), typeof(duration(DV01(), r, Float64[], Float64[])))
        duration(DV01(), r, Float64[], Float64[])
    end
    @test iszero(d)
    @test types[][1] == types[][2]
end
