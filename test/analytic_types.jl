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
    @test_throws DimensionMismatch kernel((; curve = makecurve(0.04)), tenors, cfs, [1.0])

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

struct EmptyCashflowTestCurve <: FM.Yield.AbstractYieldModel end
FC.discount(::EmptyCashflowTestCurve, t) = error("empty streams must not evaluate the curve")

@testset "Empty key-rate cashflows have zero value and risk" begin
    kernel = ActuaryUtilities.FinancialMath._ncurve_analytic
    tenors = [1.0, 3.0, 7.0]
    kr = KeyRates(tenors)
    curve = EmptyCashflowTestCurve()
    layers = (; base = curve, credit = curve, liquidity = curve)
    z = zeros(length(tenors))
    zz = zeros(length(tenors), length(tenors))

    for (cfs, times) in ((Float64[], Float64[]), (BigFloat[], Float32[]), ([], []))
        for order in (1, 2)
            raw = kernel(layers, tenors, cfs, times; order)
            @test iszero(raw.value)
            @test raw.gradient == z
            order == 2 && @test raw.hessian == zz
        end
        @test duration(kr, curve, cfs, times) == z
        @test duration(DV01(), kr, curve, cfs, times) == z
        @test duration(IR01(), kr, curve, curve, cfs, times) == z
        @test duration(CS01(), kr, curve, curve, cfs, times) == z
        @test convexity(kr, curve, cfs, times) == zz
        @test convexity(kr, curve, curve, cfs, times) == (; base = zz, credit = zz, cross = zz)
        @test iszero(duration(curve, tenors, cfs, times))
        @test iszero(duration(DV01(), curve, tenors, cfs, times))
        @test iszero(convexity(curve, tenors, cfs, times))
        @test convexity(curve, curve, tenors, cfs, times) == (; base = 0.0, credit = 0.0, cross = 0.0)

        single = sensitivities(kr, curve, cfs, times)
        @test single == (; value = 0.0, durations = z, convexities = zz)
        dollar = sensitivities(DV01(), kr, curve, cfs, times)
        @test dollar == (; value = 0.0, dv01s = z, convexities = zz)
        two = sensitivities(kr, curve, curve, cfs, times)
        @test iszero(two.value)
        @test two.base_durations == two.credit_durations == z
        @test two.convexities == (; base = zz, credit = zz, cross = zz)
        two_dollar = sensitivities(DV01(), kr, curve, curve, cfs, times)
        @test iszero(two_dollar.value)
        @test two_dollar.base_dv01s == two_dollar.credit_dv01s == z
        @test two_dollar.convexities == two.convexities
        multi = sensitivities(kr, layers, cfs, times)
        @test iszero(multi.value)
        @test all(==(z), values(multi.durations))
        @test all(blocks -> all(==(zz), values(blocks)), values(multi.convexities))
        @test convexity(kr, layers, cfs, times) == multi.convexities
    end
    big_grid = kernel(layers, big.(tenors), Float64[], Float64[]; order = 2)
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
        @test iszero(convexity(curve, tenors, cfs))
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
