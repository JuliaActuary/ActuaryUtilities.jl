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

@testset "Cashflow moments are formed in the valuation's arithmetic" begin
    # A unit payment at year 20 on a zero curve has dollar convexity t² = 400.
    # Squaring the maturity in Int8 would wrap to -112.
    kernel = ActuaryUtilities.FinancialMath._keyrate_analytic
    curve = FM.Yield.Constant(FC.Continuous(0.0))
    kr = KeyRates([10.0, 30.0])
    H(x...) = convexity(DollarConvexity(), x...)
    for T in (Int8, UInt8, Int16, Int, Float64)
        cfs, times = [1.0], T[20]
        @test convexity(curve, cfs, times) == 400
        @test H(curve, cfs, times) == 400
        @test H(curve, [FC.Cashflow(1.0, T(20))]) == 400
        s = sensitivities(SecondOrder(), curve, cfs, times)
        @test s.convexity == 400 && s.dollar_convexity == 400
        @test s.duration == 20 && s.dv01 == 20 / 10_000
        @test sum(convexity(kr, curve, cfs, times)) ≈ 400
        @test sum(H(kr, curve, cfs, times)) ≈ 400
        @test sum(H(kr, curve, [FC.Cashflow(1.0, T(20))])) ≈ 400
        k = sensitivities(SecondOrder(), kr, curve, cfs, times)
        @test sum(k.convexity) ≈ 400 && sum(k.dollar_convexity) ≈ 400
        @test sum(k.duration) ≈ 20
        # the result has the valuation's type, whatever the time type
        @test H(curve, cfs, times) isa Float64
        @test eltype(k.dollar_convexity) == Float64
        # each shock coordinate keeps its own factor: t(t + 1/m)
        @test H(0.0, cfs, times) == 420
        @test H(FC.Periodic(0.0, 2), cfs, times) == 410
        @test H(FC.Continuous(0.0), cfs, times) == 400
    end
    # Fractional years need the same promotion before t². Squaring 31/365 in
    # Rational{Int16} overflows its denominator, and Float16 rounds 50.25² to 2526.
    for t in (Int16(31) // Int16(365), Float16(50.25))
        payment = [FC.Cashflow(1.0, t)]
        expected = Float64(t)^2
        for grid in ((), (kr,))
            @test sum(H(grid..., curve, payment)) ≈ expected
            @test sum(sensitivities(SecondOrder(), grid..., curve, payment).convexity) ≈ expected
        end
    end

    # the widest times of a narrow type
    @test H(0.0, [1.0], Int8[127]) == 127 * 128
    @test H(0.0, [1.0], UInt8[255]) == 255 * 256
    @test H(curve, [1.0], Int8[-128]) == 128^2

    # At a nonzero rate, narrow integer times agree with the same times as floating point.
    cfs, times = [5.0, 5.0, 105.0], [3, 12, 25]
    flat = FM.Yield.Constant(FC.Continuous(0.04))
    for yield in (0.04, FC.Periodic(0.04, 2), FC.Continuous(0.04), flat), T in (Int8, UInt8)
        @test convexity(yield, cfs, T.(times)) ≈ convexity(yield, cfs, float.(times)) rtol = 1.0e-14
        @test H(yield, cfs, T.(times)) ≈ H(yield, cfs, float.(times)) rtol = 1.0e-14
        @test duration(yield, cfs, T.(times)) ≈ duration(yield, cfs, float.(times)) rtol = 1.0e-14
    end
    for T in (Int8, UInt8)
        narrow = kernel(flat, kr.tenors, cfs, T.(times), SecondOrder())
        wide = kernel(flat, kr.tenors, cfs, float.(times), SecondOrder())
        @test narrow.gradient ≈ wide.gradient rtol = 1.0e-14
        @test narrow.hessian ≈ wide.hessian rtol = 1.0e-14
    end

    # Narrow floating-point valuations stay narrow.
    @test H(FC.Continuous(0.0f0), Float32[1], Int8[20]) === 400.0f0
    @test H(FC.Continuous(big"0.0"), [1.0], Int8[20]) isa BigFloat
end

@testset "Key-rate buffers hold every payment's type" begin
    # Every payment contributes its type, including a dual amount or time after a Float64.
    kernel = ActuaryUtilities.FinancialMath._keyrate_analytic
    kr = KeyRates([1.0, 2.0])
    curve = FM.Yield.Constant(FC.Continuous(0.03))
    measures = (
        (cfs, times...) -> duration(kr, curve, cfs, times...),
        (cfs, times...) -> duration(DV01(), kr, curve, cfs, times...),
        (cfs, times...) -> convexity(kr, curve, cfs, times...),
        (cfs, times...) -> convexity(DollarConvexity(), kr, curve, cfs, times...),
        (cfs, times...) -> sensitivities(SecondOrder(), kr, curve, cfs, times...).value,
        (cfs, times...) -> sensitivities(SecondOrder(), kr, curve, cfs, times...).dollar_convexity,
        (cfs, times...) -> sensitivities(kr, curve, cfs, times...).dv01,
        (cfs, times...) -> duration(DV01(), curve, cfs, times...),
        (cfs, times...) -> convexity(DollarConvexity(), curve, cfs, times...),
    )
    D = ForwardDiff.derivative

    # Payments of 1 at year 1 and x at year 2, with the dual amount last and first. The reference
    # promotes the amounts to one type.
    for measure in measures
        reference = D(x -> measure([one(x), x], [1.0, 2.0]), 3.0)
        @test !iszero(sum(reference))
        @test D(x -> measure(Real[1.0, x], [1.0, 2.0]), 3.0) ≈ reference rtol = 1.0e-14
        @test D(x -> measure(Real[x, 1.0], [2.0, 1.0]), 3.0) ≈ D(x -> measure([x, one(x)], [2.0, 1.0]), 3.0) rtol = 1.0e-14
        @test D(x -> measure(Any[1, x], [1.0, 2.0]), 3.0) ≈ reference rtol = 1.0e-14
        # abstract fields in a concrete `Cashflow` type, and a `Cashflow` among numbers
        wrapped(x) = [FC.Cashflow{Real, Real}(1.0, 1.0), FC.Cashflow{Real, Real}(x, 2.0)]
        @test D(x -> measure(wrapped(x)), 3.0) ≈ reference rtol = 1.0e-14
        @test D(x -> measure(Any[FC.Cashflow(1.0, 1.0), x], [7.0, 2.0]), 3.0) ≈ reference rtol = 1.0e-14

        # a dual time among plain ones
        by_time = D(t -> measure([1.0, 3.0], [one(t), t]), 2.0)
        @test !iszero(sum(by_time))
        @test D(t -> measure([1.0, 3.0], Real[1.0, t]), 2.0) ≈ by_time rtol = 1.0e-14
        @test D(t -> measure([3.0, 1.0], Real[t, 1.0]), 2.0) ≈ D(t -> measure([3.0, 1.0], [t, one(t)]), 2.0) rtol = 1.0e-14
    end

    # Without derivatives: the buffers take the promoted type of the payments, in either order.
    homogeneous = kernel(curve, kr.tenors, [1.0, 3.0], [1.0, 2.0], SecondOrder())
    for (cfs, times) in ((Real[1, 3.0], [1.0, 2.0]), (Real[3.0, 1], [2.0, 1.0]), ([1.0, 3.0], Real[1, 2.0]), (Any[1, 3.0], Any[1.0, 2]))
        mixed = kernel(curve, kr.tenors, cfs, times, SecondOrder())
        @test mixed.value == homogeneous.value
        @test mixed.gradient == homogeneous.gradient && mixed.gradient isa Vector{Float64}
        @test mixed.hessian == homogeneous.hessian && mixed.hessian isa Matrix{Float64}
    end
    for cfs in (Real[1.0, big"3.0"], Real[big"3.0", 1.0])
        times = cfs[1] == 1 ? [1.0, 2.0] : [2.0, 1.0]
        mixed = kernel(curve, kr.tenors, cfs, times, SecondOrder())
        @test mixed.value isa BigFloat
        @test mixed.gradient isa Vector{BigFloat} && mixed.gradient ≈ homogeneous.gradient
        @test mixed.hessian isa Matrix{BigFloat} && mixed.hessian ≈ homogeneous.hessian
    end

    # Payments of one type keep the seeded buffers: the kernel stays inferred.
    for order in (FirstOrder(), SecondOrder())
        @test (@inferred kernel(curve, kr.tenors, [1.0, 3.0], [1.0, 2.0], order)).gradient == homogeneous.gradient
        @test (@inferred kernel(curve, kr.tenors, Float32[1, 3], Int8[1, 2], order)).gradient ≈ homogeneous.gradient
        @test (@inferred kernel(curve, big.(kr.tenors), [1.0, 3.0], [1.0, 2.0], order)).gradient isa Vector{BigFloat}
    end
end


@testset "Mixed precision preserves a net payment in either order" begin
    curve = FM.Yield.Constant(FC.Continuous(0.0f0))
    kr = KeyRates(Float32[1, 2])
    amounts = Real[100_000_000.0f0, 1.0f0, -100_000_000.0]
    times = fill(2.0f0, 3)
    for order in (1:3, 3:-1:1)
        for args in ((amounts[order], times), (FC.Cashflow[FC.Cashflow(amounts[i], times[i]) for i in order],))
            # All three payments fall at year two and net to one dollar.
            for grid in ((), (kr,))
                bundle = sensitivities(SecondOrder(), grid..., curve, args...)
                @test bundle.value == 1
                @test sum(bundle.duration) == sum(duration(grid..., curve, args...)) == 2
                @test sum(bundle.dv01) == sum(duration(DV01(), grid..., curve, args...)) == 0.0002
                @test sum(bundle.dollar_convexity) == sum(convexity(DollarConvexity(), grid..., curve, args...)) == 4
            end
        end
    end
end
