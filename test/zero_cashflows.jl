struct ZeroCashflowTestCurve <: FM.Yield.AbstractYieldModel end
FC.discount(::ZeroCashflowTestCurve, t) = error("zero cashflows do not require a curve query")

@testset "Zero cashflow streams" begin
    curve = ZeroCashflowTestCurve()
    tenors = [1.0, 3.0, 7.0]
    kr = KeyRates(tenors)
    z, zz = zeros(3), zeros(3, 3)
    kernel = ActuaryUtilities.FinancialMath._ncurve_analytic
    positive_zero(x) = isequal(x, zero(x))

    @testset "Scalar measures and legacy key rates" begin
        for (cfs, times) in (
                (Float64[], Float64[]), ([], []),
                ([0.0, -0.0], [1.0, 2.0]),
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
            for measure in (IR01(), CS01())
                @test positive_zero(duration(measure, curve, curve, cfs, times))
                @test positive_zero(duration(measure, curve, curve, cfs))
            end
            for measure in (KeyRateZero(1), KeyRatePar(1))
                @test positive_zero(duration(measure, curve, cfs))
                @test positive_zero(duration(measure, curve, cfs, times))
                @test positive_zero(duration(measure, curve, cfs, times, tenors))
                @test_throws ArgumentError duration(measure, curve, cfs, times, [3.0, 7.0])
            end
            @test isequal(present_values(curve, cfs, times), zeros(length(cfs)))
        end
    end

    @testset "One, two, and named discount layers" begin
        layers = (; base = curve, credit = curve, liquidity = curve)
        for (cfs, times) in ((Float64[], Float64[]), ([0.0, -0.0], [0.0, 2.0]))
            @test isequal(duration(kr, curve, cfs, times), z)
            @test isequal(duration(DV01(), kr, curve, cfs, times), z)
            @test isequal(duration(IR01(), kr, curve, curve, cfs, times), z)
            @test isequal(duration(CS01(), kr, curve, curve, cfs, times), z)
            @test isequal(convexity(kr, curve, cfs, times), zz)
            @test isequal(convexity(kr, curve, curve, cfs, times), (; base = zz, credit = zz, cross = zz))
            @test positive_zero(duration(curve, tenors, cfs, times))
            @test positive_zero(convexity(curve, tenors, cfs, times))
            @test isequal(sensitivities(kr, curve, cfs, times), (; value = 0.0, durations = z, convexities = zz))
            @test isequal(sensitivities(DV01(), kr, curve, cfs, times), (; value = 0.0, dv01s = z, convexities = zz))
            two = sensitivities(kr, curve, curve, cfs, times)
            @test positive_zero(two.value)
            @test isequal(two.base_durations, z) && isequal(two.credit_durations, z)
            @test isequal(two.convexities, (; base = zz, credit = zz, cross = zz))
            @test two.convexities.base !== two.convexities.credit
            dollar = sensitivities(DV01(), kr, curve, curve, cfs, times)
            @test isequal(dollar.base_dv01s, z) && isequal(dollar.credit_dv01s, z)
            @test isequal(dollar.convexities, two.convexities)
            multi = sensitivities(kr, layers, cfs, times)
            @test positive_zero(multi.value)
            @test all(v -> isequal(v, z), values(multi.durations))
            @test all(blocks -> all(v -> isequal(v, zz), values(blocks)), values(multi.convexities))
            @test isequal(convexity(kr, layers, cfs, times), multi.convexities)
        end
        cfs = FC.Cashflow.([0.0, -0.0], [0.0, 2.0])
        @test isequal(duration(kr, curve, cfs), z)
        @test isequal(sensitivities(kr, curve, cfs), (; value = 0.0, durations = z, convexities = zz))
    end

    @testset "Input types and validation" begin
        for T in (Float32, BigFloat)
            for cfs in (T[], zeros(T, 2))
                times = T.(eachindex(cfs))
                @test duration(curve, cfs, times) isa T
                @test convexity(curve, cfs, times) isa T
                @test eltype(present_values(curve, cfs, times)) == T
                typed = sensitivities(KeyRates(T.(tenors)), curve, cfs, times)
                @test typed.value isa T
                @test eltype(typed.durations) == eltype(typed.convexities) == T
            end
        end
        @test duration(curve, FC.Cashflow{BigFloat, Float64}[]) isa BigFloat
        @test duration(curve, Real[0, big"0.0"], [1.0, 2.0]) isa BigFloat
        bigcurve = FM.Yield.Constant(FC.Continuous(big"0.04"))
        @test sensitivities(kr, bigcurve, zeros(2), [1.0, 2.0]).value isa Float64
        for measure in (Macaulay(), Modified(), DV01(), KeyRateZero(1), KeyRatePar(1))
            @test_throws DimensionMismatch duration(measure, curve, [0.0], Float64[])
        end
        @test_throws DimensionMismatch convexity(curve, [0.0], Float64[])
        @test_throws DimensionMismatch present_values(curve, [0.0], Float64[])
        @test_throws DimensionMismatch sensitivities(kr, curve, [0.0], Float64[])
        @test_throws ArgumentError KeyRates(Float64[])
        @test_throws ArgumentError KeyRates([3.0, 1.0])
    end

    @testset "Zero amounts differ from zero net value" begin
        flat = FM.Yield.Constant(FC.Continuous(0.0))
        cfs, times = [100.0, -100.0], [1.0, 2.0]
        raw = kernel((; flat), tenors, cfs, times; order = 2)
        result = sensitivities(kr, flat, cfs, times)
        @test iszero(raw.value)
        @test any(!iszero, raw.gradient) && any(!iszero, raw.hessian)
        @test any(!isfinite, result.durations) && any(!isfinite, result.convexities)
        @test duration(DV01(), kr, flat, cfs, times) == -raw.gradient ./ 10_000
        @test !isfinite(duration(Macaulay(), flat, cfs, times))
        @test !isfinite(convexity(flat, cfs, times))
        tiny = sensitivities(kr, flat, [1.0e-200], [2.0])
        @test tiny.value == 1.0e-200
        @test sum(tiny.durations) ≈ 2.0
        # Valuation functions cannot be classified as zero streams from PV alone.
        @test all(isnan, duration(kr, _ -> 0.0, flat))
    end

    @testset "Automatic differentiation" begin
        flat = FM.Yield.Constant(FC.Continuous(0.04))
        # A zero primal amount with a nonzero partial still carries exposure.
        dollar(x) = sum(duration(DV01(), kr, flat, [x, zero(x)], [2.0, 3.0]))
        @test ForwardDiff.derivative(dollar, 0.0) ≈ 2 * FC.discount(flat, 2.0) / 10_000
        value(x) = kernel((; flat), tenors, [x], [2.0]).value
        @test ForwardDiff.derivative(value, 0.0) ≈ FC.discount(flat, 2.0)
        zero_krd(rs) = duration(kr, FM.ZeroRateCurve(rs, tenors), zeros(2), [0.0, 2.0])
        @test ForwardDiff.jacobian(zero_krd, [0.02, 0.03, 0.04]) == zz
    end

    @testset "Hull-White skips simulation" begin
        hw = FM.ShortRate.HullWhite(0.1, 0.01, FM.Yield.Constant(0.04))
        for (cfs, times) in ((Float64[], Float64[]), (zeros(2), [1.0, 2.0]))
            rng = MersenneTwister(123)
            untouched = copy(rng)
            @test isequal(sensitivities(kr, hw, cfs, times; rng), (; value = 0.0, durations = z, convexities = zz))
            @test isequal(sensitivities(DV01(), kr, hw, cfs, times; rng), (; value = 0.0, dv01s = z, convexities = zz))
            @test rand(rng) == rand(untouched)
        end
    end
end
