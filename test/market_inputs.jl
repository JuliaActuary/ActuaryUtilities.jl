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
        kr = sensitivities(KeyRates(tenors), linear(zeros_), cfs, times)
        @test r.value ≈ kr.value rtol = 1.0e-14
        @test r.key_rate.z ≈ kr.durations rtol = 1.0e-12
        @test r.key_rate_dv01.z ≈ duration(DV01(), KeyRates(tenors), linear(zeros_), cfs, times) rtol = 1.0e-12
        @test r.duration.z ≈ duration(linear(zeros_), cfs, times) rtol = 1.0e-12
        @test r.dv01.z ≈ duration(DV01(), linear(zeros_), cfs, times) rtol = 1.0e-12
    end

    @testset "independent central differences across named inputs" begin
        spread = [0.01]
        valuation(m) = FC.present_value(linear(m.z) + FM.Yield.Constant(FC.Continuous(only(m.spread))), cfs, times)
        r = sensitivities(valuation, (; z = zeros_, spread))
        h = 1.0e-6
        for (name, x) in pairs((; z = zeros_, spread)), i in eachindex(x)
            bump(d) = merge((; z = zeros_, spread), NamedTuple{(name,)}((setindex!(copy(x), x[i] + d, i),)))
            fd = (valuation(bump(-h)) - valuation(bump(h))) / (2h) / 10_000
            @test getproperty(r.key_rate_dv01, name)[i] ≈ fd rtol = 1.0e-7
        end
        # A parallel shift of the continuous spread equals the curve DV01.
        @test r.dv01.spread ≈ duration(DV01(), linear(zeros_) + FM.Yield.Constant(FC.Continuous(0.01)), cfs, times) rtol = 1.0e-12
        @test r.dv01.z ≈ r.dv01.spread rtol = 1.0e-12
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
        @test all(isfinite, flat.key_rate_dv01.z)
        @test flat.dv01.z != 0
        integer_inputs = sensitivities(m -> sum(m.x .^ 2), (; x = [1, 2, 3]))
        @test integer_inputs.key_rate_dv01.x ≈ -[2.0, 4.0, 6.0] ./ 10_000
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
        @test r.key_rate_dv01.x ≈ -2 .* collect(1.0:n) ./ 10_000
    end

    @testset "named curves and named inputs dispatch separately" begin
        curve = linear(zeros_)
        by_curve = sensitivities(c -> FC.present_value(c.curve, cfs, times), (; curve); tenors)
        @test by_curve.key_rate.curve ≈ duration(KeyRates(tenors), curve, cfs, times) rtol = 1.0e-12
        @test by_curve.key_rate_dv01.curve ≈ duration(DV01(), KeyRates(tenors), curve, cfs, times) rtol = 1.0e-12
        @test_throws MethodError sensitivities(c -> 0.0, (; rate = 0.03); tenors)
        @test_throws MethodError sensitivities(m -> 0.0, (; rate = 0.03))
    end
end
