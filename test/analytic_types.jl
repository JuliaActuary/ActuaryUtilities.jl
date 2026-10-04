@testset "Analytic sensitivities preserve valuation types" begin
    kernel = ActuaryUtilities.FinancialMath._keyrate_analytic
    tenors = [1.0, 3.0, 7.0]
    cfs = [5.0, 8.0, 105.0]
    times = [0.5, 2.0, 6.0]
    kr = KeyRates(tenors)
    makecurve(r) = FM.Yield.Constant(FC.Continuous(r))
    value(c) = sum(cf * FC.discount(c, t) for (cf, t) in zip(cfs, times))

    for r in (0.04, big"0.04")
        curve = makecurve(r)
        an = sensitivities(kr, curve, cfs, times)
        ad = sensitivities(value, kr, curve)
        @test an.value ≈ ad.value
        @test an.durations ≈ ad.durations
        @test an.convexities ≈ ad.convexities
        @test eltype(an.durations) == typeof(an.value)
        @test eltype(an.convexities) == typeof(an.value)
        base = makecurve(0.03)
        combined = sensitivities(kr, base + curve, cfs, times)
        two = sensitivities((b, c) -> value(b + c), kr, base, curve)
        @test combined.durations ≈ two.credit_durations
        @test combined.convexities ≈ two.convexities.cross
        @test eltype(combined.durations) == typeof(combined.value)
    end

    analytic(r) = sensitivities(kr, makecurve(r), cfs, times)
    automatic(r) = sensitivities(value, kr, makecurve(r))
    for field in (:durations, :convexities)
        fa(r) = sum(getproperty(analytic(r), field))
        fd(r) = sum(getproperty(automatic(r), field))
        @test ForwardDiff.derivative(fa, 0.04) ≈ ForwardDiff.derivative(fd, 0.04)
        @test ForwardDiff.derivative(r -> ForwardDiff.derivative(fa, r), 0.04) ≈
            ForwardDiff.derivative(r -> ForwardDiff.derivative(fd, r), 0.04)
    end
    big_grid = kernel(makecurve(0.04), big.(tenors), cfs, times; order = 2)
    @test eltype(big_grid.gradient) == BigFloat
    @test eltype(big_grid.hessian) == BigFloat
    @test_throws DimensionMismatch kernel(makecurve(0.04), tenors, cfs, [1.0])

    @testset "ZeroRateCurve knot Jacobian with time-zero cashflows" begin
        rates = [0.02, 0.03, 0.04]
        for ts in ([0.0, 2.0, 6.0], [0.5, 0.0, 6.0])
            krd(rs) = duration(kr, FM.ZeroRateCurve(rs, tenors), cfs, ts)
            jac = ForwardDiff.jacobian(krd, rates)
            h = 1.0e-6
            fd = hcat(
                map(eachindex(rates)) do i
                    up, down = copy(rates), copy(rates)
                    up[i] += h
                    down[i] -= h
                    (krd(up) - krd(down)) / (2h)
                end...
            )
            @test all(isfinite, jac)
            @test jac ≈ fd rtol = 1.0e-6 atol = 1.0e-8
        end
    end
end

@testset "Empty key-rate cashflows have zero value and risk" begin
    kernel = ActuaryUtilities.FinancialMath._keyrate_analytic
    tenors = [1.0, 3.0, 7.0]
    kr = KeyRates(tenors)
    curve = ZeroCashflowTestCurve()
    z = zeros(length(tenors))
    zz = zeros(length(tenors), length(tenors))

    for (cfs, times) in ((Float64[], Float64[]), (BigFloat[], Float32[]), ([], []))
        for order in (1, 2)
            raw = kernel(curve, tenors, cfs, times; order)
            @test iszero(raw.value)
            @test raw.gradient == z
            order == 2 && @test raw.hessian == zz
        end
        @test duration(kr, curve, cfs, times) == z
        @test duration(DV01(), kr, curve, cfs, times) == z
        @test convexity(kr, curve, cfs, times) == zz
        @test iszero(duration(curve, cfs, times))
        @test iszero(duration(DV01(), curve, cfs, times))
        @test iszero(convexity(curve, cfs, times))

        single = sensitivities(kr, curve, cfs, times)
        @test single == (; value = 0.0, durations = z, convexities = zz)
        dollar = sensitivities(DV01(), kr, curve, cfs, times)
        @test dollar == (; value = 0.0, dv01s = z, convexities = zz)
    end
    big_grid = kernel(curve, big.(tenors), Float64[], Float64[]; order = 2)
    @test big_grid.value isa Float64
    @test eltype(big_grid.gradient) == eltype(big_grid.hessian) == BigFloat
    big_cfs = sensitivities(kr, curve, BigFloat[], Float64[])
    @test big_cfs.value isa BigFloat
    @test eltype(big_cfs.durations) == eltype(big_cfs.convexities) == BigFloat
    empty_krd(rs) = duration(kr, FM.ZeroRateCurve(rs, tenors), Float64[], Float64[])
    @test ForwardDiff.jacobian(empty_krd, [0.02, 0.03, 0.04]) == zz

    for cfs in (FC.Cashflow{Float64, Float64}[], FC.Cashflow[])
        @test duration(kr, curve, cfs) == z
        @test convexity(kr, curve, cfs) == zz
        @test sensitivities(kr, curve, cfs) == (; value = 0.0, durations = z, convexities = zz)
        @test iszero(convexity(curve, cfs))
    end

    # Zero net value alone does not imply an empty portfolio or zero exposure.
    flat = FM.Yield.Constant(FC.Continuous(0.0))
    cfs, times = [100.0, -100.0], [1.0, 2.0]
    result = sensitivities(kr, flat, cfs, times)
    @test iszero(result.value)
    @test any(x -> !isfinite(x), result.durations)
    @test any(x -> !isfinite(x), result.convexities)
    @test any(x -> !iszero(x), duration(DV01(), kr, flat, cfs, times))
    @test_throws DimensionMismatch duration(kr, flat, [1.0], Float64[])
end
