@testset "Market-input sensitivities" begin
    tenors = [1.0, 2.0, 3.0, 5.0]
    zeros_ = [0.02, 0.025, 0.03, 0.035]
    cfs = [4.0, 4.0, 4.0, 4.0, 104.0]
    times = [1.0, 2.0, 3.0, 4.0, 5.0]
    linear(z) = FM.ZeroRateCurve(z, tenors, FM.Spline.Linear())
    value(z) = FC.present_value(linear(z), cfs, times)

    @testset "zero-rate inputs match key rates on a linear curve" begin
        # Linear interpolation of zero rates is exactly the triangular key-rate bump.
        r = sensitivities((; z = zeros_)) do m
            value(m.z)
        end
        kr = sensitivities(SecondOrder(), KeyRates(tenors), linear(zeros_), cfs, times)
        @test r.value ≈ kr.value rtol = 1.0e-14
        @test r.duration.z ≈ kr.duration rtol = 1.0e-12
        @test r.dv01.z ≈ duration(DV01(), KeyRates(tenors), linear(zeros_), cfs, times) rtol = 1.0e-12
        @test sum(r.duration.z) ≈ duration(linear(zeros_), cfs, times) rtol = 1.0e-12
        @test sum(r.dv01.z) ≈ duration(DV01(), linear(zeros_), cfs, times) rtol = 1.0e-12
        second = sensitivities(m -> value(m.z), SecondOrder(), (; z = zeros_))
        @test _same_sensitivity((; second.value, second.duration, second.dv01), r)
        @test second.convexity.z.z ≈ kr.convexity rtol = 1.0e-10
    end

    @testset "independent central differences across named inputs" begin
        spread = [0.01]
        valuation(m) = FC.present_value(linear(m.z) + FM.Yield.Constant(FC.Continuous(only(m.spread))), cfs, times)
        r = sensitivities(valuation, (; z = zeros_, spread))
        h = 1.0e-6
        for (name, x) in pairs((; z = zeros_, spread)), i in eachindex(x)
            bump(d) = merge((; z = zeros_, spread), NamedTuple{(name,)}((setindex!(copy(x), x[i] + d, i),)))
            fd = (valuation(bump(-h)) - valuation(bump(h))) / (2h) / 10_000
            @test getproperty(r.dv01, name)[i] ≈ fd rtol = 1.0e-7
        end
        # A parallel shift of the continuous spread equals the curve DV01.
        @test only(r.dv01.spread) ≈ duration(DV01(), linear(zeros_) + FM.Yield.Constant(FC.Continuous(0.01)), cfs, times) rtol = 1.0e-12
        @test sum(r.dv01.z) ≈ only(r.dv01.spread) rtol = 1.0e-12
    end

    @testset "sign, zero value, and numeric types" begin
        liability = sensitivities((; z = zeros_)) do m
            -value(m.z)
        end
        asset = sensitivities((; z = zeros_)) do m
            value(m.z)
        end
        @test liability.dv01.z ≈ -asset.dv01.z
        @test liability.duration.z ≈ asset.duration.z
        hedge = [100.0, -100.0 * FC.discount(linear(zeros_), 1.0) / FC.discount(linear(zeros_), 5.0)]
        flat = sensitivities((; z = zeros_)) do m
            FC.present_value(linear(m.z), hedge, [1.0, 5.0])
        end
        @test abs(flat.value) < 1.0e-12
        @test all(isfinite, flat.dv01.z)
        @test sum(flat.dv01.z) != 0
        integer_inputs = sensitivities(m -> sum(m.x .^ 2), (; x = [1, 2, 3]))
        @test integer_inputs.dv01.x ≈ -[2.0, 4.0, 6.0] ./ 10_000
        big_inputs = sensitivities(m -> sum(m.x .^ 2), (; x = big.([1.0, 2.0])))
        @test big_inputs.value isa BigFloat
    end

    @testset "one pass up to 64 inputs; views of the input shapes" begin
        calls = Ref(0)
        n = 40
        r = sensitivities((; x = collect(1.0:n))) do m
            calls[] += 1
            @test m.x isa AbstractVector
            @test length(m.x) == n
            sum(abs2, m.x)
        end
        @test calls[] == 2 # one primal value, one forward pass
        @test r.dv01.x ≈ -2 .* collect(1.0:n) ./ 10_000
    end

    @testset "offset inputs are read by position" begin
        x, y = [1.0, 2.0], [0.5, -1.0, 3.0]
        v(m) = sum(abs2, m.x) * (1 + sum(m.y)) + m.x[1] * m.y[end]
        w(m) = sum(abs2, m.x)
        seen = Ref{Any}(nothing)
        for order in (FirstOrder(), SecondOrder())
            one = sensitivities(w, order, (; x = OffsetArray(x, 0:1)))
            @test isequal(one, sensitivities(w, order, (; x)))
            @test one.dv01.x isa Vector{Float64}
            two = sensitivities(order, (; x = OffsetArray(x, 0:1), y = OffsetArray(y, -5:-3))) do m
                seen[] = map(r -> axes(r, 1), m)
                v(m)
            end
            @test isequal(two, sensitivities(v, order, (; x, y)))
            @test seen[] == (; x = Base.OneTo(2), y = Base.OneTo(3))
            @inferred sensitivities(w, order, (; x = OffsetArray(x, 0:1)))
        end
    end

    @testset "named curves and named inputs dispatch separately" begin
        curve = linear(zeros_)
        by_curve = sensitivities(c -> FC.present_value(c.curve, cfs, times), KeyRates(tenors), (; curve))
        @test by_curve.duration.curve ≈ duration(KeyRates(tenors), curve, cfs, times) rtol = 1.0e-12
        @test by_curve.dv01.curve ≈ duration(DV01(), KeyRates(tenors), curve, cfs, times) rtol = 1.0e-12
        @test_throws MethodError sensitivities(c -> 0.0, KeyRates(tenors), (; rate = 0.03))
        @test_throws MethodError sensitivities(m -> 0.0, (; rate = 0.03))
    end

    @testset "par quotes: calibration, AD, and finite bumps" begin
        par_tenors = [1.0, 2.0, 3.0, 5.0, 10.0]
        par_yields = [0.02, 0.024, 0.028, 0.032, 0.035]
        payments = [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 104.0]
        payment_times = collect(1.0:7.0)
        quotes(y) = FM.ParYield.(y, par_tenors; frequency = FC.Periodic(2))
        curve_from_par(y) = FM.fit(FM.Spline.Linear(), quotes(y), FM.Fit.Bootstrap())
        par_value(y) = FC.pv(curve_from_par(y), payments, payment_times)
        curve = curve_from_par(par_yields)
        @test all(q -> FC.pv(curve, q.instrument) ≈ q.price, quotes(par_yields))

        ad = sensitivities(m -> par_value(m.par_yields), (; par_yields))
        @test ad.value ≈ par_value(par_yields)
        @test ad.dv01.par_yields ≈ ad.duration.par_yields .* ad.value ./ 10_000

        # Independent primal refits at 10 bp and 1 bp. Compare both dollar and
        # normalized measures; the central difference has second-order bump error.
        fd = map((10.0, 1.0)) do bump_bps
            h = bump_bps / 10_000
            map(eachindex(par_yields)) do i
                up, down = copy(par_yields), copy(par_yields)
                up[i] += h
                down[i] -= h
                vup, vdown = par_value(up), par_value(down)
                (; dv01 = (vdown - vup) / (2 * bump_bps), duration = (vdown - vup) / (2 * h * ad.value))
            end
        end
        coarse_error = maximum(abs.([x.dv01 for x in fd[1]] .- ad.dv01.par_yields))
        fine_error = maximum(abs.([x.dv01 for x in fd[2]] .- ad.dv01.par_yields))
        @test fine_error < coarse_error / 20
        @test [x.dv01 for x in fd[2]] ≈ ad.dv01.par_yields rtol = 1.0e-6
        @test [x.duration for x in fd[2]] ≈ ad.duration.par_yields rtol = 1.0e-6
        h = 1.0e-4
        parallel_dv01 = (par_value(par_yields .- h) - par_value(par_yields .+ h)) / 2
        @test parallel_dv01 ≈ sum(ad.dv01.par_yields) rtol = 1.0e-6

        # Calibration identity: hold a par bond's payments fixed at its original
        # coupon. Other par quotes cannot change its price, while its own quote
        # sensitivity is minus the discounted coupon annuity.
        j = 4
        par_bond = quotes(par_yields)[j].instrument
        bond_risk = sensitivities((; par_yields)) do m
            FC.pv(curve_from_par(m.par_yields), par_bond)
        end
        annuity = sum(FC.discount(curve, t) / 2 for t in 0.5:0.5:par_tenors[j])
        expected = zeros(length(par_yields))
        expected[j] = annuity
        @test bond_risk.value ≈ 1.0
        @test bond_risk.duration.par_yields ≈ expected atol = 1.0e-9
        @test bond_risk.dv01.par_yields ≈ expected ./ 10_000 atol = 1.0e-13
    end
end
