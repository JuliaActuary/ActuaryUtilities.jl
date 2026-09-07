@testset "Analytic sensitivities preserve valuation types" begin
    kernel = ActuaryUtilities.FinancialMath._ncurve_analytic
    tenors = [1.0, 3.0, 7.0]
    cfs = [5.0, 8.0, 105.0]
    times = [0.5, 2.0, 6.0]
    kr = KeyRates(tenors)
    makecurve(r) = FM.Yield.Constant(FC.Continuous(r))
    value(c) = sum(cf * FC.discount(c, t) for (cf, t) in zip(cfs, times))

    for r in (0.04, big"0.04")
        curve = makecurve(r)
        an = sensitivities(kr, curve, cfs, times)
        ad = sensitivities(kr, value, curve)
        @test an.value ≈ ad.value
        @test an.durations ≈ ad.durations
        @test an.convexities ≈ ad.convexities
        @test eltype(an.durations) == typeof(an.value)
        @test eltype(an.convexities) == typeof(an.value)
        layers = (; base = makecurve(0.03), credit = curve)
        multi = sensitivities(kr, layers, cfs, times)
        two = sensitivities(kr, layers.base, layers.credit, cfs, times)
        @test multi.durations.credit ≈ two.credit_durations
        @test multi.convexities.base.credit ≈ two.convexities.cross
        @test eltype(multi.durations.base) == typeof(multi.value)
    end

    analytic(r) = sensitivities(kr, makecurve(r), cfs, times)
    automatic(r) = sensitivities(kr, value, makecurve(r))
    for field in (:durations, :convexities)
        fa(r) = sum(getproperty(analytic(r), field))
        fd(r) = sum(getproperty(automatic(r), field))
        @test ForwardDiff.derivative(fa, 0.04) ≈ ForwardDiff.derivative(fd, 0.04)
        @test ForwardDiff.derivative(r -> ForwardDiff.derivative(fa, r), 0.04) ≈
            ForwardDiff.derivative(r -> ForwardDiff.derivative(fd, r), 0.04)
    end
    big_grid = kernel((; curve = makecurve(0.04)), big.(tenors), cfs, times; order = 2)
    @test eltype(big_grid.gradient) == BigFloat
    @test eltype(big_grid.hessian) == BigFloat
    @test_throws ArgumentError duration(kr, makecurve(0.04), Float64[], Float64[])
    @test_throws DimensionMismatch kernel((; curve = makecurve(0.04)), tenors, cfs, [1.0])
end
