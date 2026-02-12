using ActuaryUtilities

using Dates
using Test
using Distributions
using StatsBase

const FM = ActuaryUtilities.FinanceModels
const FC = ActuaryUtilities.FinanceCore


include("risk_measures.jl")

@testset "Temporal functions" begin
    @testset "years_between" begin
        @test years_between(Date(2018, 9, 30), Date(2018, 9, 30)) == 0
        @test years_between(Date(2022, 10, 30), Date(2022, 10, 30)) == 0
        @test years_between(Date(2021, 10, 30), Date(2021, 10, 30)) == 0
        @test years_between(Date(2021, 11, 30), Date(2021, 10, 30)) == -1
        @test years_between(Date(2022, 10, 30), Date(2021, 10, 30)) == -1
        @test years_between(Date(2018, 9, 30), Date(2018, 9, 30), true) == 0
        @test years_between(Date(2018, 9, 30), Date(2019, 9, 30), false) == 0
        @test years_between(Date(2018, 9, 30), Date(2019, 9, 30), true) == 1
        @test years_between(Date(2018, 9, 30), Date(2019, 10, 1), true) == 1
        @test years_between(Date(2018, 9, 30), Date(2019, 10, 1), false) == 1
    end

    @testset "duration tests" begin
        @test duration(Date(2018, 9, 30), Date(2019, 9, 30)) == 2
        @test duration(Date(2018, 9, 30), Date(2018, 9, 30)) == 1
        @test duration(Date(2018, 9, 30), Date(2018, 10, 1)) == 1
        @test duration(Date(2018, 9, 30), Date(2019, 10, 1)) == 2
        @test duration(Date(2018, 9, 30), Date(2018, 6, 30)) == 0
        @test duration(Date(2018, 9, 30), Date(2017, 6, 30)) == -1
        @test duration(Date(2018, 10, 15), Date(2019, 9, 30)) == 1
        @test duration(Date(2018, 10, 15), Date(2019, 10, 30)) == 2
        @test duration(Date(2018, 10, 15), Date(2019, 10, 15)) == 2
        @test duration(Date(2018, 10, 15), Date(2019, 10, 14)) == 1
    end
end


@testset "accum_offset" begin
    @test all(accum_offset([0.9, 0.8, 0.7]) .== [1.0, 0.9, 1.0 * 0.9 * 0.8])
    @test all(accum_offset([0.9, 0.8, 0.7], op = +) .== [1.0, 1.9, 2.7])
    @test all(accum_offset([0.9, 0.8, 0.7], op = +, init = 2) .== [2.0, 2.9, 3.7])

    @test all(accum_offset(1:5, op = +) .== [1, 2, 4, 7, 11])
    @test all(accum_offset(1:5) .== [1, 1, 2, 6, 24])
    @test all(accum_offset([1, 2, 3]) .== [1, 1, 2])
end

@testset "financial calcs" begin

    @testset "price and present_value" begin
        cf = [100, 100]

        @test price(0.05, cf) ≈ pv(0.05, cf)


        cfs = ones(3)
        @test present_values(FM.Yield.Constant(0.0), cfs) == [3, 2, 1]
        pvs = present_values(FM.Yield.Constant(0.1), cfs)
        @test pvs[3] ≈ 1 / 1.1
        @test pvs[2] ≈ (1 / 1.1 + 1) / 1.1


        @test all(present_values(0.0, [1, 1, 1]) .≈ [3, 2, 1])
        @test all(present_values(0.0, [1, 1, 1], [0, 1, 2]) .≈ [3, 2, 1])
        @test all(present_values(0.0, [1, 1, 1], [1, 2, 3]) .≈ [3, 2, 1])
        @test all(present_values(0.0, [1, 1, 1], [1, 2, 3]) .≈ [3, 2, 1])
        @test all(present_values(0.01, [1, 2, 3]) .≈ [5.862461552497766, 4.921086168022744, 2.9702970297029707])

        cf = [100, 100]

        ts = [0.5, 1]

        @test pv(0.05, cf, ts) ≈ 100 / 1.05^0.5 + 100 / 1.05^1

        @test price(0.05, cf, ts) ≈ pv(0.05, cf, ts)
        @test price(0.05, -1 .* cf, ts) ≈ abs(pv(0.05, cf, ts))

        @test pv(0.05, FC.Cashflow.(cf, ts)) ≈ pv(0.05, cf, ts)
        @test price(0.05, FC.Cashflow.(cf, ts)) ≈ price(0.05, cf, ts)


    end


end

@testset "Breakeven time" begin

    @testset "basic" begin
        @test breakeven(0.1, [-10, 1, 2, 3, 4, 8]) == 5
        @test breakeven(0.1, [-10, 15, 2, 3, 4, 8]) == 1
        @test breakeven(0.1, [-10, 15, 2, 3, 4, 8]) == 1
        @test breakeven(0.1, [10, 15, 2, 3, 4, 8]) == 0
        @test isnothing(breakeven(0.1, [-10, -15, 2, 3, 4, 8]))
        @test breakeven(0.1, FC.Cashflow.([-10, 1, 2, 3, 4, 8], 0:5)) == 5
    end

    @testset "timepoints" begin
        times = [t for t in 0:5]
        @test breakeven(0.1, [-10, 1, 2, 3, 4, 8], times) == 5
        @test breakeven(0.1, [-10, 15, 2, 3, 4, 8], times) == 1
        @test breakeven(0.1, [-10, 15, 2, 3, 4, 8], times) == 1
        @test isnothing(breakeven(0.1, [-10, -15, 2, 3, 4, 8], times))
    end
end

@testset "moic" begin

    # https://bankingprep.com/multiple-on-invested-capital/
    ex1 = [-100; [t == 200 ? 100 * 1.067^t : 0 for t in 1:200]]
    @test moic(ex1) ≈ 429421.59914697794
    @test moic(FC.Cashflow.(ex1, 0:200)) ≈ 429421.59914697794


    ex2 = ex1[end] *= 0.5
    @test moic(ex1) ≈ 429421.59914697794 * 0.5


end


@testset "duration and convexity" begin

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

        #test without times
        r = FC.Periodic(0.04, 1)
        @test duration(Macaulay(), r, cfs) ≈ duration(Macaulay(), r, cfs, 1:4)
        @test duration(Modified(), r, cfs) ≈ duration(Modified(), r, cfs, 1:4)
        @test duration(r, cfs) ≈ duration(r, cfs, 1:4)
        @test duration(DV01(), r, cfs) ≈ duration(DV01(), r, cfs, 1:4)

        @test duration(FM.Yield.Constant(0.04), cfs, times) ≈ 1.777570320376649 / (1 + 0.04)
        @test duration(FM.Yield.Constant(0.04), -1 .* cfs, times) ≈ 1.777570320376649 / (1 + 0.04) atol = 0.00001
        @test duration(FM.fit(FM.Spline.Linear(), FM.ForwardYields([0.04, 0.04]), FM.Fit.Bootstrap()), cfs, times) ≈ 1.777570320376649 / (1 + 0.04) atol = 0.00001

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


    end

    @testset "Key Rate Durations" begin
        default_shift = 0.001

        @test KeyRate(5) == KeyRateZero(5)
        @test KeyRate(5) == KeyRateZero(5, default_shift)
        @test KeyRatePar(5) == KeyRatePar(5, default_shift)

        c = FM.Yield.Constant(FC.Periodic(0.04, 2))

        cp = FinancialMath._krd_new_curve(KeyRatePar(5), c, 1:10)
        cz = FinancialMath._krd_new_curve(KeyRateZero(5), c, 1:10)

        # test some relationships between par and zero curve
        @test FM.par(cp, 5) ≈ FM.par(c, 5) + default_shift atol = 0.0002 # 0.001 is the default shift
        @test FM.par(cp, 4) ≈ FC.Periodic(0.04, 2) atol = 0.0001
        @test zero(cp, 5) > FM.par(cp, 5)
        @test zero(cp, 6) < FM.par(cp, 6)

        @testset "FEH123" begin
            # http://www.financialexamhelp123.com/key-rate-duration/

            #test some curve properties


            bond = (
                cfs = [0.02 for t in 1:10],
                times = collect(0.5:0.5:5),
            )
            bond.cfs[end] += 1.0

            @test duration(KeyRatePar(1), c, bond.cfs, bond.times) ≈ 0.0 atol = 0.01
            @test duration(KeyRatePar(2), c, bond.cfs, bond.times) ≈ 0.0 atol = 0.01
            @test duration(KeyRatePar(3), c, bond.cfs, bond.times) ≈ 0.0 atol = 0.01
            @test duration(KeyRatePar(4), c, bond.cfs, bond.times) ≈ 0.0 atol = 0.01
            @test duration(KeyRatePar(5), c, bond.cfs, bond.times) ≈ 4.45 atol = 0.05

            bond = (times = [1, 2, 3, 4, 5], cfs = [0, 0, 0, 0, 100])
            c = FC.Continuous(0.05)
            @test duration(KeyRateZero(1), c, bond.cfs, bond.times) ≈ 0.0 atol = 1.0e-10
            @test duration(KeyRateZero(2), c, bond.cfs, bond.times) ≈ 0.0 atol = 1.0e-10
            @test duration(KeyRateZero(3), c, bond.cfs, bond.times) ≈ 0.0 atol = 1.0e-10
            @test duration(KeyRateZero(4), c, bond.cfs, bond.times) ≈ 0.0 atol = 1.0e-10
            @test duration(KeyRateZero(5), c, bond.cfs, bond.times) ≈ duration(c, bond.cfs, bond.times) atol = 0.1

            cfo = FC.Cashflow.(bond.cfs, bond.times)
            @test duration(KeyRateZero(5), c, cfo) ≈ duration(c, bond.cfs, bond.times) atol = 0.1


        end
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

@testset "ZeroRateCurve duration" begin
    @testset "ZCB at a tenor: duration concentrated at that tenor" begin
        rates = [0.03, 0.03, 0.03]
        tenors = [1.0, 2.0, 5.0]
        zrc = FM.ZeroRateCurve(rates, tenors)
        face = 100.0

        # key rate durations via duration(zrc, cfs, times)
        krds = duration(zrc, [0.0, 0.0, face], tenors)

        # duration at the maturity tenor (index 3) should be ≈ t = 5.0
        @test krds[3] ≈ 5.0 atol = 1e-6

        # durations at other tenors should be zero
        @test krds[1] ≈ 0.0 atol = 1e-6
        @test krds[2] ≈ 0.0 atol = 1e-6
    end

    @testset "coupon bond flat curve: sum of KRDs ≈ Macaulay duration" begin
        rates = [0.04, 0.04, 0.04, 0.04, 0.04]
        tenors = [1.0, 2.0, 3.0, 4.0, 5.0]
        zrc = FM.ZeroRateCurve(rates, tenors)
        coupon = 5.0
        face = 100.0
        cfs = [coupon, coupon, coupon, coupon, coupon + face]

        krds = duration(zrc, cfs, tenors)

        # all key rate durations should be positive
        @test all(krds .> 0)

        # sum of KRDs ≈ Macaulay duration for a flat continuous curve
        dfs = [exp(-0.04 * t) for t in tenors]
        mac_dur = sum(t * cf * df for (t, cf, df) in zip(tenors, cfs, dfs)) / sum(cf * df for (cf, df) in zip(cfs, dfs))
        @test sum(krds) ≈ mac_dur atol = 1e-6
    end

    @testset "DV01 positive for standard bond" begin
        rates = [0.03, 0.03, 0.03]
        tenors = [1.0, 2.0, 3.0]
        zrc = FM.ZeroRateCurve(rates, tenors)
        cfs = [5.0, 5.0, 105.0]

        dv01s = duration(DV01(), zrc, cfs, tenors)
        @test all(dv01s .> 0)
    end

    @testset "do-block custom valuation (callable bond)" begin
        rates = [0.05, 0.05, 0.05, 0.05, 0.05]
        tenors = [1.0, 2.0, 3.0, 4.0, 5.0]
        zrc = FM.ZeroRateCurve(rates, tenors)
        coupon = 6.0
        face = 100.0
        call_price = 102.0
        cfs_noncallable = [coupon, coupon, coupon, coupon, coupon + face]

        callable_dur = duration(zrc) do curve
            ncv = sum(cf * curve(t) for (cf, t) in zip(cfs_noncallable, tenors))
            called_value = sum(cf * curve(t) for (cf, t) in zip(cfs_noncallable[1:3], tenors[1:3])) -
                           cfs_noncallable[3] * curve(3.0) + call_price * curve(3.0)
            min(ncv, called_value)
        end

        @test length(callable_dur) == 5
    end

    @testset "convexity matrix for ZCB" begin
        rates = [0.03, 0.03, 0.03]
        tenors = [1.0, 2.0, 5.0]
        zrc = FM.ZeroRateCurve(rates, tenors)
        face = 100.0

        conv = convexity(zrc, [0.0, 0.0, face], tenors)

        # diagonal at the maturity tenor should be t^2 = 25.0
        @test conv[3, 3] ≈ 25.0 atol = 1e-6

        # off-diagonal should be zero
        @test conv[1, 3] ≈ 0.0 atol = 1e-6
        @test conv[2, 3] ≈ 0.0 atol = 1e-6
    end

    @testset "two-curve IR01/CS01" begin
        base_rates = [0.03, 0.03, 0.03, 0.03, 0.03]
        credit_rates = [0.02, 0.02, 0.02, 0.02, 0.02]
        tenors = [1.0, 2.0, 3.0, 4.0, 5.0]
        base = FM.ZeroRateCurve(base_rates, tenors)
        credit = FM.ZeroRateCurve(credit_rates, tenors)
        cfs = [5.0, 5.0, 5.0, 5.0, 105.0]

        ir01s = duration(IR01(), base, credit, cfs, tenors)
        cs01s = duration(CS01(), base, credit, cfs, tenors)

        # For additive combination, IR01 ≈ CS01
        @test ir01s ≈ cs01s atol = 1e-10
        @test all(ir01s .> 0)
    end

    @testset "two-curve convexity" begin
        base_rates = [0.03, 0.03, 0.03]
        credit_rates = [0.02, 0.02, 0.02]
        tenors = [1.0, 2.0, 5.0]
        base = FM.ZeroRateCurve(base_rates, tenors)
        credit = FM.ZeroRateCurve(credit_rates, tenors)
        cfs = [5.0, 5.0, 105.0]

        conv = convexity(base, credit, cfs, tenors)

        @test !all(isapprox.(conv.cross, 0.0, atol = 1e-10))
        @test !all(isapprox.(conv.base, 0.0, atol = 1e-10))
        @test !all(isapprox.(conv.credit, 0.0, atol = 1e-10))
        # For symmetric additive combination, cross ≈ base
        @test conv.cross ≈ conv.base atol = 1e-10
    end

    @testset "cubic vs linear: same on flat curve" begin
        rates = [0.04, 0.04, 0.04, 0.04, 0.04]
        tenors = [1.0, 2.0, 3.0, 4.0, 5.0]
        cfs = [5.0, 5.0, 5.0, 5.0, 105.0]

        zrc_lin = FM.ZeroRateCurve(rates, tenors, FM.Spline.Linear())
        zrc_cub = FM.ZeroRateCurve(rates, tenors, FM.Spline.Cubic())

        dur_lin = duration(zrc_lin, cfs, tenors)
        dur_cub = duration(zrc_cub, cfs, tenors)

        @test dur_lin ≈ dur_cub atol = 1e-4
    end
end

@testset "ZeroRateCurve sensitivities" begin
    @testset "ZCB analytical" begin
        rates = [0.03, 0.03, 0.03]
        tenors = [1.0, 2.0, 5.0]
        zrc = FM.ZeroRateCurve(rates, tenors)
        face = 100.0

        result = sensitivities(zrc, [0.0, 0.0, face], tenors)

        @test result.value ≈ face * exp(-0.03 * 5.0) atol = 1e-6
        @test result.durations[3] ≈ 5.0 atol = 1e-6
        @test result.durations[1] ≈ 0.0 atol = 1e-6
        @test result.convexities[3, 3] ≈ 25.0 atol = 1e-6
    end

    @testset "coupon bond" begin
        rates = [0.04, 0.04, 0.04, 0.04, 0.04]
        tenors = [1.0, 2.0, 3.0, 4.0, 5.0]
        zrc = FM.ZeroRateCurve(rates, tenors)
        cfs = [5.0, 5.0, 5.0, 5.0, 105.0]

        result = sensitivities(zrc, cfs, tenors)

        @test result.value > 0
        @test all(result.durations .> 0)
        @test all(result.dv01s .> 0)

        # sensitivities returns same durations as calling duration separately
        @test result.durations ≈ duration(zrc, cfs, tenors) atol = 1e-12
    end

    @testset "do-block" begin
        rates = [0.03, 0.03, 0.03]
        tenors = [1.0, 2.0, 3.0]
        zrc = FM.ZeroRateCurve(rates, tenors)
        cfs = [5.0, 5.0, 105.0]

        result = sensitivities(zrc) do curve
            sum(cf * curve(t) for (cf, t) in zip(cfs, tenors))
        end

        @test result.value > 0
        @test all(result.durations .> 0)
    end

    @testset "two-curve additive: IR01 ≈ CS01" begin
        base_rates = [0.03, 0.03, 0.03, 0.03, 0.03]
        credit_rates = [0.02, 0.02, 0.02, 0.02, 0.02]
        tenors = [1.0, 2.0, 3.0, 4.0, 5.0]
        base = FM.ZeroRateCurve(base_rates, tenors)
        credit = FM.ZeroRateCurve(credit_rates, tenors)
        cfs = [5.0, 5.0, 5.0, 5.0, 105.0]

        result = sensitivities(base, credit, cfs, tenors)

        @test result.base_durations ≈ result.credit_durations atol = 1e-10
        @test result.base_dv01s ≈ result.credit_dv01s atol = 1e-12

        # Macaulay duration for flat continuous rate 0.05
        total_rate = 0.05
        dfs = [exp(-total_rate * t) for t in tenors]
        mac_dur = sum(t * cf * df for (t, cf, df) in zip(tenors, cfs, dfs)) / sum(cf * df for (cf, df) in zip(cfs, dfs))
        @test sum(result.base_durations) ≈ mac_dur atol = 1e-6
    end

    @testset "two-curve non-additive: base ≠ credit" begin
        base_rates = [0.03, 0.03, 0.03]
        credit_rates = [0.02, 0.02, 0.02]
        tenors = [1.0, 2.0, 5.0]
        base = FM.ZeroRateCurve(base_rates, tenors)
        credit = FM.ZeroRateCurve(credit_rates, tenors)
        face = 100.0

        result = sensitivities(base, credit) do base_curve, credit_curve
            face * (2.0 * base_curve(5.0) + 0.5 * credit_curve(5.0))
        end

        @test !isapprox(result.base_durations, result.credit_durations, atol = 1e-6)
    end

    @testset "chapter VGH test case" begin
        zero_rates = [0.01, 0.02, 0.02, 0.03, 0.05, 0.055]
        times = [1.0, 2.0, 3.0, 5.0, 10.0, 20.0]
        zrc = FM.ZeroRateCurve(zero_rates, times, FM.Spline.Cubic())

        # 10-year fixed bond, 9% coupon, semi-annual, par=1.0
        coupon = 0.09
        cfs_times = collect(0.5:0.5:10.0)
        cfs = [coupon / 2 + (t == 10.0 ? 1.0 : 0.0) for t in cfs_times]

        result = sensitivities(zrc, cfs, cfs_times)

        @test result.value > 0
        # durations at tenors within the bond's maturity are positive
        @test all(result.durations[1:5] .> 0)
        @test sum(result.durations) > 0  # total duration is positive

        # convexity matrix is symmetric
        @test result.convexities ≈ result.convexities' atol = 1e-10
    end

    @testset "portfolio linearity" begin
        zero_rates = [0.03, 0.03, 0.03, 0.03, 0.03]
        tenors = [1.0, 2.0, 3.0, 4.0, 5.0]
        zrc = FM.ZeroRateCurve(zero_rates, tenors)

        bond1_cfs = [0.05, 0.05, 1.05, 0.0, 0.0]
        bond1_times = [1.0, 2.0, 3.0, 4.0, 5.0]
        bond2_cfs = [0.03, 0.03, 0.03, 1.03, 0.0]
        bond2_times = [1.0, 2.0, 3.0, 5.0, 5.0]

        # Portfolio valuation — single AD pass over sum
        portfolio_valuation = curve -> begin
            sum(cf * curve(t) for (cf, t) in zip(bond1_cfs, bond1_times)) +
            sum(cf * curve(t) for (cf, t) in zip(bond2_cfs, bond2_times))
        end
        portfolio_dv01 = duration(DV01(), portfolio_valuation, zrc)

        # Individual DV01s
        dv01_1 = duration(DV01(), zrc, bond1_cfs, bond1_times)
        dv01_2 = duration(DV01(), zrc, bond2_cfs, bond2_times)

        # DV01 is additive (not value-weighted like modified duration)
        @test portfolio_dv01 ≈ dv01_1 .+ dv01_2 atol = 1e-10
    end
end

@testset "spread" begin
    cfs = fill(10, 10)
    cfo = FC.Cashflow.(cfs, 1:10)

    @test spread(0.04, 0.05, cfs) ≈ FC.Periodic(0.01, 1) atol = 1.0e-6
    @test spread(0.04, 0.05, cfo) ≈ FC.Periodic(0.01, 1) atol = 1.0e-6

    @test spread(FC.Continuous(0.04), FC.Continuous(0.05), cfs) ≈ FC.Periodic(1)(FC.Continuous(0.05) - FC.Continuous(0.04)) atol = 1.0e-6

    # 2021-03-31 rates from Treasury.gov
    rates = [0.01, 0.01, 0.03, 0.05, 0.07, 0.16, 0.35, 0.92, 1.4, 1.74, 2.31, 2.41] ./ 100
    mats = [1 / 12, 2 / 12, 3 / 12, 6 / 12, 1, 2, 3, 5, 7, 10, 20, 30]

    y = FM.fit(FM.Spline.Linear(), FM.CMTYield.(rates, mats), FM.Fit.Bootstrap())

    y2 = y + FC.Periodic(0.01, 1)

    s = spread(y, y2, cfs)

    @test s ≈ FC.Periodic(0.01, 1) atol = 1.0e-6
end
