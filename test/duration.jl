@testset "duration and convexity" begin

    @testset "Macaulay weighted sum and allocations" begin
        times = [0.5, 1.0, 2.0, 4.0]
        yields = (
            0.04, FC.Periodic(0.04, 2), FC.Continuous(0.04),
            FM.Yield.Constant(0.04),
            FM.ZeroRateCurve([0.02, 0.03, 0.04, 0.05], times, FM.Spline.Linear()),
        )
        for y in yields, amounts in ([5.0, 5.0, 5.0, 105.0], [-5.0, -5.0, -5.0, -105.0], [-100.0, 5.0, 5.0, 110.0])
            weights = FC.present_value.(y, amounts, times)
            expected = sum(times .* weights) / sum(weights)
            @test duration(Macaulay(), y, amounts, times) ≈ expected
            @test duration(Macaulay(), y, reshape(amounts, :, 1), times) ≈ expected
            @test duration(Macaulay(), y, FC.Cashflow.(amounts, times)) ≈ expected
        end

        amounts = [5.0, 5.0, 5.0, 105.0]
        allocations(y, cfs, ts) = @allocated duration(Macaulay(), y, cfs, ts)
        for y in yields[1:4]
            allocations(y, amounts, times) # warm up each specialization
            @test allocations(y, amounts, times) == 0
        end
        @test_throws DimensionMismatch duration(Macaulay(), 0.04, amounts, times[1:3])
    end

    # per issue #74
    @testset "generators" begin
        g = (10 for t in 1:10)
        v = collect(g)
        i = FM.Yield.Constant(0.04)
        @test duration(0.04, g) ≈ duration(0.04, v)
        @test duration(i, g) ≈ duration(i, v)
        @test convexity(0.04, g) ≈ convexity(0.04, v)
    end

    @testset "wikipedia example" begin
        times = [0.5, 1, 1.5, 2]
        cfs = [10, 10, 10, 110]
        cfo = FC.Cashflow.(cfs, times)
        V = present_value(0.04, cfs, times)

        @test duration(Macaulay(), 0.04, cfs, times) ≈ 1.777570320376649
        @test duration(Modified(), 0.04, cfs, times) ≈ 1.777570320376649 / (1 + 0.04)
        @test duration(0.04, cfs, times) ≈ 1.777570320376649 / (1 + 0.04)

        @test duration(Macaulay(), 0.04, cfo) ≈ 1.777570320376649
        @test duration(Modified(), 0.04, cfo) ≈ 1.777570320376649 / (1 + 0.04)
        @test duration(0.04, cfo) ≈ 1.777570320376649 / (1 + 0.04)

        # wikipedia example defines DV01 as a per point change, but industry practice is per basis point. Ref Issue #96
        @test duration(DV01(), 0.04, cfs, times) ≈ 1.777570320376649 / (1 + 0.04) * V / 10000
        @test duration(DV01(), 0.04, cfo) ≈ 1.777570320376649 / (1 + 0.04) * V / 10000

        # test with a Rate
        r = FC.Periodic(0.04, 1)
        @test duration(Macaulay(), r, cfs, times) ≈ 1.777570320376649
        @test duration(Modified(), r, cfs, times) ≈ 1.777570320376649 / (1 + 0.04)
        @test duration(r, cfs, times) ≈ 1.777570320376649 / (1 + 0.04)
        @test duration(DV01(), r, cfs, times) ≈ 1.777570320376649 / (1 + 0.04) * V / 10000

        # Macaulay duration: genuinely mixed-sign cashflows use signed PVs.
        # For [-100, 110] at t = [1, 2], the signed-PV ratio is exactly 58/3.
        @test duration(Macaulay(), 0.04, [-100.0, 110.0], [1.0, 2.0]) ≈ 58 / 3

        #test without times
        r = FC.Periodic(0.04, 1)
        @test duration(Macaulay(), r, cfs) ≈ duration(Macaulay(), r, cfs, 1:4)
        @test duration(Modified(), r, cfs) ≈ duration(Modified(), r, cfs, 1:4)
        @test duration(r, cfs) ≈ duration(r, cfs, 1:4)
        @test duration(DV01(), r, cfs) ≈ duration(DV01(), r, cfs, 1:4)

        # FM v5 CompositeYield operates in continuous zero-rate space,
        # so bump-and-reprice gives continuous modified duration (= Macaulay duration)
        @test duration(FM.Yield.Constant(0.04), cfs, times) ≈ 1.777570320376649
        @test duration(FM.Yield.Constant(0.04), -1 .* cfs, times) ≈ 1.777570320376649 atol = 0.00001
        @test duration(FM.fit(FM.Spline.Linear(), FM.ForwardYield([0.04, 0.04]), FM.Fit.Bootstrap()), cfs, times) ≈ 1.777570320376649 atol = 0.00001

        # test that dispatch resolves the ambiguity between duration(FM.Yield,vec) and duration(FM.Yield, function)
        @test duration(FM.Yield.Constant(0.03), cfs) > 0
        @test convexity(FM.Yield.Constant(0.03), cfs) > 0
    end

    @testset "finpipe example" begin
        # from https://www.finpipe.com/duration-macaulay-and-modified-duration-convexity/

        cfs = zeros(10) .+ 3.75
        cfs[10] += 100

        times = 0.5:0.5:5.0
        int = (1 + 0.075 / 2)^2 - 1 # convert bond yield to effective yield

        @test isapprox(present_value(int, cfs, times), 100.0, atol = 1.0e-2)
        @test isapprox(duration(Macaulay(), int, cfs, times), 4.26, atol = 1.0e-2)
    end

    @testset "Primer example" begin
        # from https://math.illinoisstate.edu/krzysio/Primer.pdf
        # the duration tests are commented out because I think the paper is wrong on the duration?
        cfs = [0, 0, 0, 0, 1.0e6]
        times = 1:5
        cfo = FC.Cashflow.(cfs, times)

        @test isapprox(present_value(0.04, cfs, times), 821927.11, atol = 1.0e-2)
        # @test isapprox(duration(0.04,cfs,times),4.76190476,atol=1e-6)
        @test isapprox(convexity(0.04, cfs, times), 27.7366864, atol = 1.0e-6)
        @test isapprox(convexity(0.04, cfs), 27.7366864, atol = 1.0e-6)
        @test isapprox(convexity(0.04, cfo), 27.7366864, atol = 1.0e-6)

        # the same, but with a functional argument
        value(i) = present_value(i, cfs, times)
        # @test isapprox(duration(0.04,value),4.76190476,atol=1e-6)
        @test isapprox(convexity(0.04, value), 27.7366864, atol = 1.0e-6)
    end

    @testset "Quantlib" begin
        # https://mhittesdorf.wordpress.com/2013/03/12/introduction-to-quantlib-duration-and-convexity/
        cfs = [5, 5, 105]
        times = 1:3
        @test present_value(0.03, cfs, times) ≈ 105.6572227097894
        @test duration(Macaulay(), 0.03, cfs, times) ≈ 2.863504670671131
        @test duration(0.03, cfs, times) ≈ 2.780101622010806
        @test convexity(0.03, cfs, times) ≈ 10.62580548268594

        # test omitting the times argument
        @test duration(Macaulay(), 0.03, cfs) ≈ 2.863504670671131
        @test duration(0.03, cfs) ≈ 2.780101622010806
        @test convexity(0.03, cfs) ≈ 10.62580548268594


        # test a single matrix dimension
        cfs = [
            5 0 0
            0 5 105
        ]

        @test duration(0.03, sum(cfs, dims = 1), times) ≈ 2.780101622010806

        cfs = [
            5 0
            5 0
            0 105
        ]

        @test duration(0.03, sum(cfs, dims = 2), times) ≈ 2.780101622010806

        @test duration(Macaulay(), 0.03, sum(cfs, dims = 2), times) ≈ 2.863504670671131

    end
end

@testset "IR01 and CS01" begin
    @testset "flat rates: IR01 ≈ CS01 ≈ DV01" begin
        cfs = [10, 10, 10, 110]
        times = [0.5, 1, 1.5, 2]
        base_rate = 0.03
        credit_spread = 0.02
        total_rate = base_rate + credit_spread

        dv01 = duration(DV01(), total_rate, cfs, times)
        ir01 = duration(IR01(), base_rate, credit_spread, cfs, times)
        cs01 = duration(CS01(), base_rate, credit_spread, cfs, times)

        @test ir01 ≈ dv01
        @test cs01 ≈ dv01
        @test ir01 ≈ cs01
    end

    @testset "with Rate objects" begin
        cfs = [10, 10, 10, 110]
        times = [0.5, 1, 1.5, 2]
        base_r = FC.Periodic(0.03, 1)
        spread_r = FC.Periodic(0.02, 1)

        ir01 = duration(IR01(), base_r, spread_r, cfs, times)
        cs01 = duration(CS01(), base_r, spread_r, cfs, times)

        @test ir01 > 0
        @test cs01 > 0
        @test ir01 ≈ cs01
    end

    @testset "without explicit times" begin
        cfs = [5, 5, 5, 105]

        dv01 = duration(DV01(), 0.05, cfs)
        ir01 = duration(IR01(), 0.03, 0.02, cfs)
        cs01 = duration(CS01(), 0.03, 0.02, cfs)

        @test ir01 ≈ dv01
        @test cs01 ≈ dv01
    end

    @testset "with Cashflow objects" begin
        cfs = [5, 5, 5, 105]
        times = 1:4
        cfo = FC.Cashflow.(cfs, times)

        ir01_cfo = duration(IR01(), 0.03, 0.02, cfo)
        ir01_raw = duration(IR01(), 0.03, 0.02, cfs, times)

        @test ir01_cfo ≈ ir01_raw

        cs01_cfo = duration(CS01(), 0.03, 0.02, cfo)
        cs01_raw = duration(CS01(), 0.03, 0.02, cfs, times)

        @test cs01_cfo ≈ cs01_raw
    end

    @testset "with yield curve" begin
        rates = [0.01, 0.02, 0.03, 0.04]
        mats = [1, 2, 3, 5]
        y = FM.fit(FM.Spline.Linear(), FM.CMTYield.(rates, mats), FM.Fit.Bootstrap())
        credit_spread = FC.Periodic(0.02, 1)
        cfs = [5, 5, 5, 105]
        times = 1:4

        ir01 = duration(IR01(), y, credit_spread, cfs, times)
        cs01 = duration(CS01(), y, credit_spread, cfs, times)

        @test ir01 > 0
        @test cs01 > 0
    end
end

@testset "do-block with AbstractYieldModel" begin
    c = FM.Yield.Constant(0.04)
    cfs = [5, 5, 5, 105]
    times = 1:4

    # duration with do-block (function-first argument order)
    d = duration(c) do i
        price(i, cfs, times)
    end
    @test d ≈ duration(c, cfs, times)

    # convexity with do-block
    cv = convexity(c) do i
        price(i, cfs, times)
    end
    @test cv ≈ convexity(c, cfs, times)
    @test cv ≈ convexity(FC.Continuous(log1p(0.04)), cfs, times)
end

@testset "Scalar do-block on ZRC uses continuous curve shocks" begin
    # No-tenor do-block calls use the same continuous-zero parallel shift as
    # the tenor-aware and key-rate APIs.
    rates = [0.04, 0.04, 0.04, 0.04, 0.04]
    tenors = [1.0, 2.0, 3.0, 4.0, 5.0]
    zrc = FM.ZeroRateCurve(rates, tenors, FM.Spline.Linear())
    cfs = [5.0, 5.0, 5.0, 5.0, 105.0]
    times = [1.0, 2.0, 3.0, 4.0, 5.0]

    vf_dur = duration(zrc) do curve
        sum(cf * curve(t) for (cf, t) in zip(cfs, times))
    end
    @test vf_dur isa Real
    @test vf_dur ≈ duration(zrc, tenors, cfs, times) atol = 1.0e-12

    vf_conv = convexity(zrc) do curve
        sum(cf * curve(t) for (cf, t) in zip(cfs, times))
    end
    @test vf_conv isa Real
    @test vf_conv ≈ convexity(zrc, tenors, cfs, times) atol = 1.0e-12
end
