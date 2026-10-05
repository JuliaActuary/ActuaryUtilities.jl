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
        for grid in ((kr,), ())
            an = sensitivities(SecondOrder(), grid..., curve, cfs, times)
            ad = sensitivities(value, SecondOrder(), grid..., curve)
            @test _same_sensitivity(an, ad)
            @test eltype(an.duration) == eltype(an.dv01) == eltype(an.convexity) == typeof(an.value)
            base = makecurve(0.03)
            combined = sensitivities(SecondOrder(), grid..., base + curve, cfs, times)
            two = sensitivities((b, c) -> value(b + c), SecondOrder(), grid..., base, curve)
            @test combined.duration ≈ two.duration.credit
            @test combined.convexity ≈ two.convexity.base.credit
            @test eltype(combined.duration) == typeof(combined.value)
        end
    end

    analytic(r) = sensitivities(SecondOrder(), kr, makecurve(r), cfs, times)
    automatic(r) = sensitivities(value, SecondOrder(), kr, makecurve(r))
    for field in (:duration, :dv01, :convexity, :dollar_convexity)
        fa(r) = sum(getproperty(analytic(r), field))
        fd(r) = sum(getproperty(automatic(r), field))
        @test ForwardDiff.derivative(fa, 0.04) ≈ ForwardDiff.derivative(fd, 0.04)
        @test ForwardDiff.derivative(r -> ForwardDiff.derivative(fa, r), 0.04) ≈
            ForwardDiff.derivative(r -> ForwardDiff.derivative(fd, r), 0.04)
    end
    big_grid = kernel(makecurve(0.04), big.(tenors), cfs, times, SecondOrder())
    @test eltype(big_grid.gradient) == BigFloat
    @test eltype(big_grid.hessian) == BigFloat
    @test_throws DimensionMismatch duration(kr, makecurve(0.04), cfs, [1.0])

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
        for order in (FirstOrder(), SecondOrder())
            raw = kernel(curve, tenors, cfs, times, order)
            @test iszero(raw.value)
            @test raw.gradient == z
            order isa SecondOrder && @test raw.hessian == zz
        end
        @test duration(kr, curve, cfs, times) == z
        @test duration(DV01(), kr, curve, cfs, times) == z
        @test convexity(kr, curve, cfs, times) == zz
        @test iszero(duration(curve, cfs, times))
        @test iszero(duration(DV01(), curve, cfs, times))
        @test iszero(convexity(curve, cfs, times))

        @test sensitivities(kr, curve, cfs, times) == (; value = 0.0, duration = z, dv01 = z)
        @test sensitivities(SecondOrder(), kr, curve, cfs, times) == (; value = 0.0, duration = z, dv01 = z, convexity = zz, dollar_convexity = zz)
        @test sensitivities(SecondOrder(), curve, cfs, times) == (; value = 0.0, duration = 0.0, dv01 = 0.0, convexity = 0.0, dollar_convexity = 0.0)
    end
    big_grid = kernel(curve, big.(tenors), Float64[], Float64[], SecondOrder())
    @test big_grid.value isa Float64
    @test eltype(big_grid.gradient) == eltype(big_grid.hessian) == BigFloat
    big_cfs = sensitivities(SecondOrder(), kr, curve, BigFloat[], Float64[])
    @test big_cfs.value isa BigFloat
    @test eltype(big_cfs.duration) == eltype(big_cfs.dv01) == eltype(big_cfs.convexity) == eltype(big_cfs.dollar_convexity) == BigFloat
    empty_krd(rs) = duration(kr, FM.ZeroRateCurve(rs, tenors), Float64[], Float64[])
    @test ForwardDiff.jacobian(empty_krd, [0.02, 0.03, 0.04]) == zz

    for cfs in (FC.Cashflow{Float64, Float64}[], FC.Cashflow[])
        @test duration(kr, curve, cfs) == z
        @test convexity(kr, curve, cfs) == zz
        @test sensitivities(SecondOrder(), kr, curve, cfs) == (; value = 0.0, duration = z, dv01 = z, convexity = zz, dollar_convexity = zz)
        @test iszero(convexity(curve, cfs))
    end

    # Zero net value alone does not imply an empty portfolio or zero exposure.
    flat = FM.Yield.Constant(FC.Continuous(0.0))
    cfs, times = [100.0, -100.0], [1.0, 2.0]
    result = sensitivities(SecondOrder(), kr, flat, cfs, times)
    @test iszero(result.value)
    @test any(x -> !isfinite(x), result.duration)
    @test any(x -> !isfinite(x), result.convexity)
    @test any(x -> !iszero(x), result.dv01)
    @test result.dv01 == duration(DV01(), kr, flat, cfs, times)
    @test_throws DimensionMismatch duration(kr, flat, [1.0], Float64[])
end
