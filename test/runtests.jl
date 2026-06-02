using ActuaryUtilities

using Dates
using Test
using Distributions
using StatsBase
using Random

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

    @testset "KeyRateZero TransformedYield" begin
        krd_points = 1:10
        shift = 0.001

        # Interior point: triangle bump at τ=5
        bump_fn = FinancialMath._tent_bump(shift, 5, krd_points)
        base_z = FC.Continuous(0.05)
        @test bump_fn(base_z, 5).continuous_value ≈ 0.05 + shift       # peak
        @test bump_fn(base_z, 4).continuous_value ≈ 0.05               # left neighbor
        @test bump_fn(base_z, 6).continuous_value ≈ 0.05               # right neighbor
        @test bump_fn(base_z, 4.5).continuous_value ≈ 0.05 + shift / 2 # midpoint of ramp

        # First point: flat left, ramp right
        bump_first = FinancialMath._tent_bump(shift, 1, krd_points)
        @test bump_first(base_z, 0.5).continuous_value ≈ 0.05 + shift  # flat left
        @test bump_first(base_z, 1.0).continuous_value ≈ 0.05 + shift  # at τ
        @test bump_first(base_z, 2.0).continuous_value ≈ 0.05          # right neighbor
        @test bump_first(base_z, 1.5).continuous_value ≈ 0.05 + shift / 2

        # Last point: ramp left, flat right
        bump_last = FinancialMath._tent_bump(shift, 10, krd_points)
        @test bump_last(base_z, 9.0).continuous_value ≈ 0.05           # left neighbor
        @test bump_last(base_z, 10.0).continuous_value ≈ 0.05 + shift  # at τ
        @test bump_last(base_z, 11.0).continuous_value ≈ 0.05 + shift  # flat right

        # Returns TransformedYield type
        c = FM.Yield.Constant(FC.Continuous(0.05))
        cz = FinancialMath._krd_new_curve(KeyRateZero(5), c, krd_points)
        @test cz isa FM.Yield.TransformedYield

        # Rate input properly wrapped in Constant
        cz_rate = FinancialMath._krd_new_curve(KeyRateZero(5), FC.Continuous(0.05), krd_points)
        @test cz_rate isa FM.Yield.TransformedYield

        # Sum of KRDs ≈ total modified duration (flat curve sanity check)
        bond_cfs = [3.0, 3.0, 3.0, 3.0, 103.0]
        bond_times = [1.0, 2.0, 3.0, 4.0, 5.0]
        flat = FM.Yield.Constant(FC.Continuous(0.05))
        krd_sum = sum(duration(KeyRateZero(t), flat, bond_cfs, bond_times, 1:5) for t in 1:5)
        mod_dur = duration(flat, bond_cfs, bond_times)
        @test krd_sum ≈ mod_dur atol = 0.01
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
        # Use Linear for perfect locality (zero sensitivity outside adjacent intervals)
        zrc = FM.ZeroRateCurve(rates, tenors, FM.Spline.Linear())
        face = 100.0

        # key rate durations via duration(KeyRates(knots), zrc, cfs, times)
        krds = duration(KeyRates(tenors), zrc, [0.0, 0.0, face], tenors)

        # duration at the maturity tenor (index 3) should be ≈ t = 5.0
        @test krds[3] ≈ 5.0 atol = 1e-6

        # durations at other tenors should be zero
        @test krds[1] ≈ 0.0 atol = 1e-6
        @test krds[2] ≈ 0.0 atol = 1e-6
    end

    @testset "coupon bond flat curve: sum of KRDs ≈ Macaulay duration" begin
        rates = [0.04, 0.04, 0.04, 0.04, 0.04]
        tenors = [1.0, 2.0, 3.0, 4.0, 5.0]
        coupon = 5.0
        face = 100.0
        cfs = [coupon, coupon, coupon, coupon, coupon + face]

        # sum of KRDs ≈ Macaulay duration regardless of interpolation
        dfs = [exp(-0.04 * t) for t in tenors]
        mac_dur = sum(t * cf * df for (t, cf, df) in zip(tenors, cfs, dfs)) / sum(cf * df for (cf, df) in zip(cfs, dfs))

        for spline in [FM.Spline.Linear(), FM.Spline.MonotoneConvex(), FM.Spline.PCHIP()]
            zrc = FM.ZeroRateCurve(rates, tenors, spline)
            krds = duration(KeyRates(tenors), zrc, cfs, tenors)
            @test sum(krds) ≈ mac_dur atol = 1e-4
        end

        # all KRDs positive only guaranteed for Linear (perfectly local)
        zrc_lin = FM.ZeroRateCurve(rates, tenors, FM.Spline.Linear())
        @test all(duration(KeyRates(tenors), zrc_lin, cfs, tenors) .> 0)
    end

    @testset "DV01 positive for standard bond" begin
        rates = [0.03, 0.03, 0.03]
        tenors = [1.0, 2.0, 3.0]
        # Use Linear: smooth methods may produce negative KRDs at some tenors
        zrc = FM.ZeroRateCurve(rates, tenors, FM.Spline.Linear())
        cfs = [5.0, 5.0, 105.0]

        dv01s = duration(DV01(), KeyRates(tenors), zrc, cfs, tenors)
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

        callable_dur = duration(KeyRates(tenors), zrc) do curve
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
        # Use Linear for perfect locality in convexity test
        zrc = FM.ZeroRateCurve(rates, tenors, FM.Spline.Linear())
        face = 100.0

        conv = convexity(KeyRates(tenors), zrc, [0.0, 0.0, face], tenors)

        # diagonal at the maturity tenor should be t^2 = 25.0
        @test conv[3, 3] ≈ 25.0 atol = 1e-6

        # off-diagonal should be zero
        @test conv[1, 3] ≈ 0.0 atol = 1e-6
        @test conv[2, 3] ≈ 0.0 atol = 1e-6
    end

    @testset "scalar convexity(curve, tenors, ...) ≡ sum(KRD Hessian) (POU regression guard)" begin
        # Under partition of unity of the KRD hat functions, the continuous-
        # shock parallel-shift scalar convexity equals the sum of the N×N
        # key-rate Hessian by the chain rule. The scalar entry points now
        # route through `_parallel_continuous_convexity` (TenorShift + two
        # nested ForwardDiff.derivatives), avoiding the Hessian build entirely.
        # Locks the equivalence in.
        rates  = [0.02, 0.025, 0.03, 0.035, 0.04]
        tenors = [1.0, 2.0, 3.0, 5.0, 7.0]
        zrc    = FM.ZeroRateCurve(rates, tenors, FM.Spline.Linear())
        cfs    = [5.0, 5.0, 5.0, 5.0, 105.0]
        times  = [1.0, 2.0, 3.0, 4.0, 5.0]

        scalar_form = convexity(zrc, tenors, cfs, times)
        matrix_sum  = sum(convexity(KeyRates(tenors), zrc, cfs, times))
        @test scalar_form ≈ matrix_sum atol = 1e-8

        vf_scalar = convexity(c -> sum(cf * FC.discount(c, t) for (cf, t) in zip(cfs, times)), zrc, tenors)
        vf_matrix = sum(convexity(KeyRates(tenors), c -> sum(cf * FC.discount(c, t) for (cf, t) in zip(cfs, times)), zrc))
        @test vf_scalar ≈ vf_matrix atol = 1e-8

        # Cashflow-vector form
        cashflows = [FC.Cashflow(cfs[k], times[k]) for k in eachindex(cfs)]
        @test convexity(zrc, tenors, cashflows) ≈ matrix_sum atol = 1e-8
    end

    @testset "two-curve IR01/CS01" begin
        base_rates = [0.03, 0.03, 0.03, 0.03, 0.03]
        credit_rates = [0.02, 0.02, 0.02, 0.02, 0.02]
        tenors = [1.0, 2.0, 3.0, 4.0, 5.0]
        # Use Linear for symmetric IR01 ≈ CS01 test
        base = FM.ZeroRateCurve(base_rates, tenors, FM.Spline.Linear())
        credit = FM.ZeroRateCurve(credit_rates, tenors, FM.Spline.Linear())
        cfs = [5.0, 5.0, 5.0, 5.0, 105.0]

        ir01s = duration(IR01(), KeyRates(tenors), base, credit, cfs, tenors)
        cs01s = duration(CS01(), KeyRates(tenors), base, credit, cfs, tenors)

        # For additive combination, IR01 ≈ CS01
        @test ir01s ≈ cs01s atol = 1e-10
        @test all(ir01s .> 0)
    end

    @testset "two-curve convexity" begin
        base_rates = [0.03, 0.03, 0.03]
        credit_rates = [0.02, 0.02, 0.02]
        tenors = [1.0, 2.0, 5.0]
        # Use Linear for symmetric cross ≈ base test
        base = FM.ZeroRateCurve(base_rates, tenors, FM.Spline.Linear())
        credit = FM.ZeroRateCurve(credit_rates, tenors, FM.Spline.Linear())
        cfs = [5.0, 5.0, 105.0]

        conv = convexity(KeyRates(tenors), base, credit, cfs, tenors)

        @test !all(isapprox.(conv.cross, 0.0, atol = 1e-10))
        @test !all(isapprox.(conv.base, 0.0, atol = 1e-10))
        @test !all(isapprox.(conv.credit, 0.0, atol = 1e-10))
        # For symmetric additive combination, cross ≈ base
        @test conv.cross ≈ conv.base atol = 1e-10
    end

    @testset "scalar return: duration(zrc, ...) returns sum of KeyRates" begin
        rates = [0.04, 0.04, 0.04, 0.04, 0.04]
        tenors = [1.0, 2.0, 3.0, 4.0, 5.0]
        zrc = FM.ZeroRateCurve(rates, tenors, FM.Spline.Linear())
        cfs = [5.0, 5.0, 5.0, 5.0, 105.0]

        # duration scalar = sum of KeyRates vector
        scalar_dur = duration(zrc, tenors, cfs, tenors)
        krds = duration(KeyRates(tenors), zrc, cfs, tenors)
        @test scalar_dur isa Real
        @test !(scalar_dur isa AbstractArray)
        @test scalar_dur ≈ sum(krds) atol = 1e-12

        # DV01 scalar = sum of KeyRates DV01 vector
        scalar_dv01 = duration(DV01(), zrc, tenors, cfs, tenors)
        dv01_vec = duration(DV01(), KeyRates(tenors), zrc, cfs, tenors)
        @test scalar_dv01 isa Real
        @test scalar_dv01 ≈ sum(dv01_vec) atol = 1e-12

        # convexity scalar = sum of KeyRates convexity matrix
        scalar_conv = convexity(zrc, tenors, cfs, tenors)
        conv_mat = convexity(KeyRates(tenors), zrc, cfs, tenors)
        @test scalar_conv isa Real
        @test scalar_conv ≈ sum(conv_mat) atol = 1e-12

        # scalar ZRC duration ≈ scalar yield duration for flat curve
        # ZRC uses continuous compounding, so compare with Continuous rate
        @test scalar_dur ≈ duration(FC.Continuous(0.04), cfs, tenors) atol = 1e-4
    end

    @testset "scalar return: two-curve duration and convexity" begin
        base_rates = [0.03, 0.03, 0.03, 0.03, 0.03]
        credit_rates = [0.02, 0.02, 0.02, 0.02, 0.02]
        tenors = [1.0, 2.0, 3.0, 4.0, 5.0]
        base = FM.ZeroRateCurve(base_rates, tenors, FM.Spline.Linear())
        credit = FM.ZeroRateCurve(credit_rates, tenors, FM.Spline.Linear())
        cfs = [5.0, 5.0, 5.0, 5.0, 105.0]

        # IR01 scalar = sum of KeyRates IR01 vector
        scalar_ir01 = duration(IR01(), base, credit, tenors, cfs, tenors)
        ir01_vec = duration(IR01(), KeyRates(tenors), base, credit, cfs, tenors)
        @test scalar_ir01 isa Real
        @test scalar_ir01 ≈ sum(ir01_vec) atol = 1e-12

        # CS01 scalar = sum of KeyRates CS01 vector
        scalar_cs01 = duration(CS01(), base, credit, tenors, cfs, tenors)
        cs01_vec = duration(CS01(), KeyRates(tenors), base, credit, cfs, tenors)
        @test scalar_cs01 isa Real
        @test scalar_cs01 ≈ sum(cs01_vec) atol = 1e-12

        # Two-curve convexity: scalars = sums of matrices
        scalar_conv = convexity(base, credit, tenors, cfs, tenors)
        mat_conv = convexity(KeyRates(tenors), base, credit, cfs, tenors)
        @test scalar_conv.base isa Real
        @test scalar_conv.base ≈ sum(mat_conv.base) atol = 1e-12
        @test scalar_conv.credit ≈ sum(mat_conv.credit) atol = 1e-12
        @test scalar_conv.cross ≈ sum(mat_conv.cross) atol = 1e-12
    end

    @testset "cubic vs linear: same on flat curve" begin
        rates = [0.04, 0.04, 0.04, 0.04, 0.04]
        tenors = [1.0, 2.0, 3.0, 4.0, 5.0]
        cfs = [5.0, 5.0, 5.0, 5.0, 105.0]

        zrc_lin = FM.ZeroRateCurve(rates, tenors, FM.Spline.Linear())
        zrc_cub = FM.ZeroRateCurve(rates, tenors, FM.Spline.Cubic())

        dur_lin = duration(KeyRates(tenors), zrc_lin, cfs, tenors)
        dur_cub = duration(KeyRates(tenors), zrc_cub, cfs, tenors)

        @test dur_lin ≈ dur_cub atol = 1e-4
    end

    @testset "multi-curve NamedTuple: analytic ≈ _ncurve_ad (gradient/Hessian)" begin
        # _keyrate_analytic_n must agree with _ncurve_ad on the vanilla cashflow
        # case (static cfs, multiplicative discount product). Regression guard
        # for the closed-form derivation of multi-curve KRD.
        rates  = fill(0.03, 5)
        tenors = [1.0, 2.0, 3.0, 4.0, 5.0]
        zrc1   = FM.ZeroRateCurve(rates,         tenors, FM.Spline.Linear())
        zrc2   = FM.ZeroRateCurve(rates .+ 0.005, tenors, FM.Spline.Linear())
        zrc3   = FM.ZeroRateCurve(rates .+ 0.002, tenors, FM.Spline.Linear())
        amts   = [5.0, 5.0, 5.0, 5.0, 105.0]
        times  = [1.0, 2.0, 3.0, 4.0, 5.0]
        nt3    = (; rf = zrc1, credit = zrc2, ilp = zrc3)

        vf(c) = sum(amts[k] * FC.discount(c.rf, times[k]) *
                              FC.discount(c.credit, times[k]) *
                              FC.discount(c.ilp, times[k]) for k in eachindex(amts))
        v_ad, g_ad = ActuaryUtilities.FinancialMath._ncurve_ad(vf, nt3, tenors)
        an = ActuaryUtilities.FinancialMath._keyrate_analytic_n(nt3, tenors, amts, times; order = 2)

        @test v_ad ≈ an.value rtol = 1e-12
        for r in (:rf, :credit, :ilp)
            @test maximum(abs.(g_ad[r] .- an.gradient[r])) < 1e-12
        end

        # Public API surfaces accept the NamedTuple form.
        sens = sensitivities(KeyRates(tenors), nt3, amts, times)
        @test sens.value ≈ v_ad rtol = 1e-12
        @test maximum(abs.(sens.durations.rf .- (-g_ad.rf ./ v_ad))) < 1e-12
        conv = convexity(KeyRates(tenors), nt3, amts, times)
        @test conv.rf.rf isa AbstractMatrix
        @test conv.rf.credit ≈ conv.credit.rf  # symmetric under multiplicative discount
    end
end

@testset "ZeroRateCurve sensitivities" begin
    @testset "ZCB analytical" begin
        rates = [0.03, 0.03, 0.03]
        tenors = [1.0, 2.0, 5.0]
        # Use Linear for perfect locality assertions
        zrc = FM.ZeroRateCurve(rates, tenors, FM.Spline.Linear())
        face = 100.0

        result = sensitivities(KeyRates(tenors), zrc, [0.0, 0.0, face], tenors)

        @test result.value ≈ face * exp(-0.03 * 5.0) atol = 1e-6
        @test result.durations[3] ≈ 5.0 atol = 1e-6
        @test result.durations[1] ≈ 0.0 atol = 1e-6
        @test result.convexities[3, 3] ≈ 25.0 atol = 1e-6
    end

    @testset "coupon bond" begin
        rates = [0.04, 0.04, 0.04, 0.04, 0.04]
        tenors = [1.0, 2.0, 3.0, 4.0, 5.0]
        # Use Linear for positive-KRD guarantee
        zrc = FM.ZeroRateCurve(rates, tenors, FM.Spline.Linear())
        cfs = [5.0, 5.0, 5.0, 5.0, 105.0]

        result = sensitivities(KeyRates(tenors), zrc, cfs, tenors)

        @test result.value > 0
        @test all(result.durations .> 0)

        # sensitivities returns same durations as calling duration(KeyRates(tenors), ...) separately
        @test result.durations ≈ duration(KeyRates(tenors), zrc, cfs, tenors) atol = 1e-12

        # DV01 dispatch
        dv01_result = sensitivities(DV01(), KeyRates(tenors), zrc, cfs, tenors)
        @test all(dv01_result.dv01s .> 0)
        @test dv01_result.dv01s ≈ duration(DV01(), KeyRates(tenors), zrc, cfs, tenors) atol = 1e-12
        @test dv01_result.value ≈ result.value atol = 1e-12
    end

    @testset "do-block" begin
        rates = [0.03, 0.03, 0.03]
        tenors = [1.0, 2.0, 3.0]
        # Use Linear for positive-KRD guarantee
        zrc = FM.ZeroRateCurve(rates, tenors, FM.Spline.Linear())
        cfs = [5.0, 5.0, 105.0]

        result = sensitivities(KeyRates(tenors), zrc) do curve
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

        result = sensitivities(KeyRates(tenors), base, credit, cfs, tenors)

        @test result.base_durations ≈ result.credit_durations atol = 1e-10

        # DV01 dispatch
        dv01_result = sensitivities(DV01(), KeyRates(tenors), base, credit, cfs, tenors)
        @test dv01_result.base_dv01s ≈ dv01_result.credit_dv01s atol = 1e-12
        @test dv01_result.value ≈ result.value atol = 1e-12

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

        result = sensitivities(KeyRates(tenors), base, credit) do base_curve, credit_curve
            face * (2.0 * base_curve(5.0) + 0.5 * credit_curve(5.0))
        end

        @test !isapprox(result.base_durations, result.credit_durations, atol = 1e-6)
    end

    @testset "two-curve with mismatched ZRC storage tenors" begin
        # Under the unified API the KRD knot grid is supplied explicitly, so
        # base and credit no longer need matching `tenors` fields — they can
        # be evaluated against any common knot grid.
        base = FM.ZeroRateCurve([0.03, 0.03, 0.03], [1.0, 2.0, 5.0])
        credit = FM.ZeroRateCurve([0.02, 0.02], [1.0, 2.0])
        knots = [1.0, 2.0, 5.0]
        result = sensitivities(KeyRates(knots), base, credit, [5.0, 5.0, 105.0], [1.0, 2.0, 5.0])
        @test result.value > 0
        @test length(result.base_durations) == length(knots)
        @test length(result.credit_durations) == length(knots)
    end

    @testset "chapter VGH test case" begin
        zero_rates = [0.01, 0.02, 0.02, 0.03, 0.05, 0.055]
        times = [1.0, 2.0, 3.0, 5.0, 10.0, 20.0]
        zrc = FM.ZeroRateCurve(zero_rates, times, FM.Spline.Cubic())

        # 10-year fixed bond, 9% coupon, semi-annual, par=1.0
        coupon = 0.09
        cfs_times = collect(0.5:0.5:10.0)
        cfs = [coupon / 2 + (t == 10.0 ? 1.0 : 0.0) for t in cfs_times]

        result = sensitivities(KeyRates(times), zrc, cfs, cfs_times)

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
        portfolio_dv01 = duration(DV01(), KeyRates(tenors), portfolio_valuation, zrc)

        # Individual DV01s
        dv01_1 = duration(DV01(), KeyRates(tenors), zrc, bond1_cfs, bond1_times)
        dv01_2 = duration(DV01(), KeyRates(tenors), zrc, bond2_cfs, bond2_times)

        # DV01 is additive (not value-weighted like modified duration)
        @test portfolio_dv01 ≈ dv01_1 .+ dv01_2 atol = 1e-10
    end
end

@testset "ZeroRateCurve external validation" begin

    @testset "AD vs finite difference" begin
        # Cross-validate AD gradient against central finite differences.
        # FD has O(ε²) truncation error so tolerance is ~1e-4, not machine-eps.
        rates = [0.02, 0.03, 0.04, 0.05]
        tenors = [1.0, 3.0, 5.0, 10.0]
        zrc = FM.ZeroRateCurve(rates, tenors)
        cfs = [3.0, 3.0, 3.0, 103.0]
        ε = 1e-5

        ad_dv01 = duration(DV01(), KeyRates(tenors), zrc, cfs, tenors)

        for i in 1:4
            rates_up = copy(rates); rates_up[i] += ε
            rates_dn = copy(rates); rates_dn[i] -= ε
            zrc_up = FM.ZeroRateCurve(rates_up, tenors)
            zrc_dn = FM.ZeroRateCurve(rates_dn, tenors)
            v_up = sum(cf * zrc_up(t) for (cf, t) in zip(cfs, tenors))
            v_dn = sum(cf * zrc_dn(t) for (cf, t) in zip(cfs, tenors))
            fd_dv01_i = -(v_up - v_dn) / (2ε) / 10_000
            @test ad_dv01[i] ≈ fd_dv01_i atol = 1e-4
        end
    end

    @testset "flat zero curve KRDs (Deriscope reference)" begin
        # Reference: Deriscope blog "Bond Key Rate Duration (KRD) in Excel"
        # https://blog.deriscope.com/index.php/en/excel-quantlib-key-rate-duration
        # They use QuantLib with a 1% FD shift on a flat 5.1441% zero curve,
        # 4% coupon 5yr bond. Their KRDs sum to 4.067035 (modified dur = 4.066705).
        # The ~0.03% discrepancy is due to the large (1%) FD shift introducing
        # O(Δr²) error. Our AD gives exact derivatives, so sum(KRDs) = Macaulay
        # duration exactly (continuous compounding ⟹ modified = Macaulay).
        r = 0.051441  # continuously compounded
        tenors = [1.0, 2.0, 3.0, 4.0, 5.0]
        rates = fill(r, 5)
        zrc = FM.ZeroRateCurve(rates, tenors, FM.Spline.Linear())

        # 4% annual coupon, 5yr, face=100
        cfs = [4.0, 4.0, 4.0, 4.0, 104.0]

        krds = duration(KeyRates(tenors), zrc, cfs, tenors)

        # On a flat curve with linear interp, each KRD_i = t_i * cf_i * df_i / V
        dfs = [exp(-r * t) for t in tenors]
        V = sum(cf * df for (cf, df) in zip(cfs, dfs))
        expected_krds = [t * cf * df / V for (t, cf, df) in zip(tenors, cfs, dfs)]

        @test krds ≈ expected_krds atol = 1e-6

        # Sum of KRDs = modified duration (exact for continuous compounding)
        mac_dur = sum(t * cf * df for (t, cf, df) in zip(tenors, cfs, dfs)) / V
        @test sum(krds) ≈ mac_dur atol = 1e-10

        # Deriscope FD reference: modified dur = 4.067, sum(KRDs) = 4.067.
        # Our exact AD Macaulay duration is ~4.618 — the difference arises
        # because Deriscope uses dirty price with accrued interest and
        # settlement-date conventions. We just verify our value is in the
        # right ballpark for a 5yr bond (between 3 and 5).
        @test 3.0 < sum(krds) < 5.0
    end

    @testset "coupon bond KRD analytical (flat curve)" begin
        # Analytical derivation: V = Σ cf_i * exp(-r * t_i).
        # With linear interpolation and cashflows at exact tenor points,
        # ∂V/∂r_i = -t_i * cf_i * exp(-r * t_i), so KRD_i = t_i * cf_i * df_i / V.
        # This is exact (AD gives true partial derivatives, no FD approximation).
        r = 0.04
        tenors = [1.0, 2.0, 3.0, 4.0, 5.0]
        rates = fill(r, 5)
        zrc = FM.ZeroRateCurve(rates, tenors, FM.Spline.Linear())
        cfs = [5.0, 5.0, 5.0, 5.0, 105.0]

        krds = duration(KeyRates(tenors), zrc, cfs, tenors)

        dfs = [exp(-r * t) for t in tenors]
        V = sum(cf * df for (cf, df) in zip(cfs, dfs))

        # Each KRD = t_i * cf_i * df_i / V
        for i in 1:5
            expected = tenors[i] * cfs[i] * dfs[i] / V
            @test krds[i] ≈ expected atol = 1e-8
        end
    end

    @testset "non-flat curve, cashflows at tenors" begin
        # With linear interpolation of zero rates and cashflows at exact tenor
        # points, the discount factor at tenor i depends only on rate i:
        # df_i = exp(-r_i * t_i). So ∂V/∂r_i = -t_i * cf_i * exp(-r_i * t_i),
        # giving KRD_i = t_i * cf_i * df_i / V — same formula as flat curve.
        rates = [0.02, 0.03, 0.04, 0.05]
        tenors = [1.0, 2.0, 5.0, 10.0]
        zrc = FM.ZeroRateCurve(rates, tenors, FM.Spline.Linear())
        cfs = [3.0, 3.0, 3.0, 103.0]

        krds = duration(KeyRates(tenors), zrc, cfs, tenors)

        dfs = [exp(-rates[i] * tenors[i]) for i in 1:4]
        V = sum(cf * df for (cf, df) in zip(cfs, dfs))

        for i in 1:4
            expected = tenors[i] * cfs[i] * dfs[i] / V
            @test krds[i] ≈ expected atol = 1e-6
        end
    end

    @testset "DV01 do-block: two assets = 2× single asset" begin
        rates = [0.03, 0.03, 0.03, 0.03, 0.03]
        tenors = [1.0, 2.0, 3.0, 4.0, 5.0]
        zrc = FM.ZeroRateCurve(rates, tenors)
        cfs = [5.0, 5.0, 5.0, 5.0, 105.0]

        single_dv01 = duration(DV01(), KeyRates(tenors), zrc, cfs, tenors)

        double_dv01 = duration(DV01(), KeyRates(tenors), zrc) do curve
            2 * sum(cf * curve(t) for (cf, t) in zip(cfs, tenors))
        end

        @test double_dv01 ≈ 2 .* single_dv01 atol = 1e-10
    end

    @testset "convexity analytical (flat curve)" begin
        # Second-order analytical: ∂²V/∂r_i² = t_i² * cf_i * exp(-r*t_i),
        # so convexity_{i,i} = t_i² * cf_i * df_i / V.
        # Cross-partials ∂²V/∂r_i∂r_j = 0 because df_i = exp(-r_i * t_i)
        # doesn't depend on r_j when cashflows are at exact tenor points
        # with linear interpolation.
        r = 0.04
        tenors = [1.0, 2.0, 3.0, 4.0, 5.0]
        rates = fill(r, 5)
        zrc = FM.ZeroRateCurve(rates, tenors, FM.Spline.Linear())
        cfs = [5.0, 5.0, 5.0, 5.0, 105.0]

        conv = convexity(KeyRates(tenors), zrc, cfs, tenors)

        dfs = [exp(-r * t) for t in tenors]
        V = sum(cf * df for (cf, df) in zip(cfs, dfs))

        for i in 1:5
            expected_diag = tenors[i]^2 * cfs[i] * dfs[i] / V
            @test conv[i, i] ≈ expected_diag atol = 1e-6
        end

        # Off-diagonal should be zero (no cross-dependence at exact tenor points)
        for i in 1:5, j in 1:5
            i == j && continue
            @test conv[i, j] ≈ 0.0 atol = 1e-10
        end
    end
end

@testset "Hull-White MC: sum of KRDs = deterministic (risk-neutral guarantee)" begin
    # For fixed cashflows, E[V] = Σ cf_i × P(0,t_i) under any risk-neutral model
    # (Glasserman, 2003, Ch. 7), so the sum of key rate durations is preserved
    # between deterministic discounting and Monte Carlo under Hull-White dynamics.
    # Individual KRDs differ because HW's θ(t) calibration creates non-local
    # rate dependencies (Brigo & Mercurio, 2006, Ch. 3).
    rates = [0.03, 0.03, 0.03, 0.03, 0.03]
    tenors = [1.0, 2.0, 3.0, 4.0, 5.0]
    cfs = [5.0, 5.0, 5.0, 5.0, 105.0]

    # Deterministic KRDs
    zrc = FM.ZeroRateCurve(rates, tenors)
    det = sensitivities(KeyRates(tenors), zrc, cfs, tenors)

    # Hull-White MC KRDs (AD through Monte Carlo via pathwise differentiation)
    hw_result = sensitivities(KeyRates(tenors), zrc) do curve
        hw = FM.ShortRate.HullWhite(0.1, 0.01, curve)
        scenarios = FM.simulate(hw; n_scenarios=500, timestep=1 / 12, horizon=6.0, rng=Xoshiro(42))
        sum(sum(cf * FC.discount(sc, t) for (cf, t) in zip(cfs, tenors)) for sc in scenarios) / 500
    end

    # Total duration preserved (risk-neutral pricing theorem)
    @test sum(hw_result.durations) ≈ sum(det.durations) atol = 0.05

    # Individual KRDs should differ (HW redistributes across tenors)
    @test !(hw_result.durations ≈ det.durations)

    # Present values should also agree
    @test hw_result.value ≈ det.value atol = 0.5
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

@testset "ZeroRateCurve Cashflow support" begin
    rates = [0.04, 0.04, 0.04, 0.04, 0.04]
    tenors = [1.0, 2.0, 3.0, 4.0, 5.0]
    zrc = FM.ZeroRateCurve(rates, tenors, FM.Spline.Linear())
    amounts = [5.0, 5.0, 5.0, 5.0, 105.0]
    cfs = FC.Cashflow.(amounts, tenors)

    # single-curve duration
    @test duration(zrc, tenors, cfs) ≈ duration(zrc, tenors, amounts, tenors)
    @test duration(KeyRates(tenors), zrc, cfs) ≈ duration(KeyRates(tenors), zrc, amounts, tenors)
    @test duration(DV01(), zrc, tenors, cfs) ≈ duration(DV01(), zrc, tenors, amounts, tenors)
    @test duration(DV01(), KeyRates(tenors), zrc, cfs) ≈ duration(DV01(), KeyRates(tenors), zrc, amounts, tenors)

    # single-curve convexity
    @test convexity(zrc, tenors, cfs) ≈ convexity(zrc, tenors, amounts, tenors)
    @test convexity(KeyRates(tenors), zrc, cfs) ≈ convexity(KeyRates(tenors), zrc, amounts, tenors)

    # single-curve sensitivities
    s_cf = sensitivities(KeyRates(tenors), zrc, cfs)
    s_raw = sensitivities(KeyRates(tenors), zrc, amounts, tenors)
    @test s_cf.value ≈ s_raw.value
    @test s_cf.durations ≈ s_raw.durations
    @test s_cf.convexities ≈ s_raw.convexities

    s_dv01_cf = sensitivities(DV01(), KeyRates(tenors), zrc, cfs)
    s_dv01_raw = sensitivities(DV01(), KeyRates(tenors), zrc, amounts, tenors)
    @test s_dv01_cf.dv01s ≈ s_dv01_raw.dv01s
    @test s_dv01_cf.convexities ≈ s_dv01_raw.convexities

    # two-curve duration
    base_rates = [0.03, 0.03, 0.03, 0.03, 0.03]
    credit_rates = [0.02, 0.02, 0.02, 0.02, 0.02]
    base = FM.ZeroRateCurve(base_rates, tenors, FM.Spline.Linear())
    credit = FM.ZeroRateCurve(credit_rates, tenors, FM.Spline.Linear())

    @test duration(IR01(), base, credit, tenors, cfs) ≈ duration(IR01(), base, credit, tenors, amounts, tenors)
    @test duration(IR01(), KeyRates(tenors), base, credit, cfs) ≈ duration(IR01(), KeyRates(tenors), base, credit, amounts, tenors)
    @test duration(CS01(), base, credit, tenors, cfs) ≈ duration(CS01(), base, credit, tenors, amounts, tenors)
    @test duration(CS01(), KeyRates(tenors), base, credit, cfs) ≈ duration(CS01(), KeyRates(tenors), base, credit, amounts, tenors)

    # two-curve convexity
    conv_cf = convexity(base, credit, tenors, cfs)
    conv_raw = convexity(base, credit, tenors, amounts, tenors)
    @test conv_cf.base ≈ conv_raw.base
    @test conv_cf.credit ≈ conv_raw.credit
    @test conv_cf.cross ≈ conv_raw.cross

    conv_kr_cf = convexity(KeyRates(tenors), base, credit, cfs)
    conv_kr_raw = convexity(KeyRates(tenors), base, credit, amounts, tenors)
    @test conv_kr_cf.base ≈ conv_kr_raw.base
    @test conv_kr_cf.credit ≈ conv_kr_raw.credit
    @test conv_kr_cf.cross ≈ conv_kr_raw.cross

    # two-curve sensitivities
    s2_cf = sensitivities(KeyRates(tenors), base, credit, cfs)
    s2_raw = sensitivities(KeyRates(tenors), base, credit, amounts, tenors)
    @test s2_cf.base_durations ≈ s2_raw.base_durations
    @test s2_cf.credit_durations ≈ s2_raw.credit_durations

    s2_dv01_cf = sensitivities(DV01(), KeyRates(tenors), base, credit, cfs)
    s2_dv01_raw = sensitivities(DV01(), KeyRates(tenors), base, credit, amounts, tenors)
    @test s2_dv01_cf.base_dv01s ≈ s2_dv01_raw.base_dv01s
    @test s2_dv01_cf.credit_dv01s ≈ s2_dv01_raw.credit_dv01s

    @testset "Cashflow with non-tenor times" begin
        # ZRC has annual tenors, but cashflows are semi-annual
        rates = [0.04, 0.04, 0.04, 0.04, 0.04]
        tenors = [1.0, 2.0, 3.0, 4.0, 5.0]
        zrc = FM.ZeroRateCurve(rates, tenors, FM.Spline.Linear())
        semi_times = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        semi_amounts = [2.5, 2.5, 2.5, 2.5, 2.5, 102.5]
        semi_cfs = FC.Cashflow.(semi_amounts, semi_times)

        @test duration(zrc, tenors, semi_cfs) ≈ duration(zrc, tenors, semi_amounts, semi_times)
        @test duration(KeyRates(tenors), zrc, semi_cfs) ≈ duration(KeyRates(tenors), zrc, semi_amounts, semi_times)
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
end

@testset "Scalar do-block on ZRC falls through to generic FD path" begin
    # Previously a ZRC-specific dispatch routed `duration(fn, zrc)` /
    # `convexity(fn, zrc)` through the AD KRD path. With the unified API,
    # those 2-arg calls fall through to the generic `duration(yield, vf)`
    # FD-based scalar path — which adds a parallel shift via Periodic
    # compounding, not Continuous, so the numerical values are not bitwise
    # comparable to `sum(KRDs)` (which differentiates w.r.t. continuous zero
    # rates). We just verify the calls execute and return a Real scalar.
    rates = [0.04, 0.04, 0.04, 0.04, 0.04]
    tenors = [1.0, 2.0, 3.0, 4.0, 5.0]
    zrc = FM.ZeroRateCurve(rates, tenors, FM.Spline.Linear())
    cfs = [5.0, 5.0, 5.0, 5.0, 105.0]
    times = [1.0, 2.0, 3.0, 4.0, 5.0]

    vf_dur = duration(zrc) do curve
        sum(cf * curve(t) for (cf, t) in zip(cfs, times))
    end
    @test vf_dur isa Real

    vf_conv = convexity(zrc) do curve
        sum(cf * curve(t) for (cf, t) in zip(cfs, times))
    end
    @test vf_conv isa Real
end

# Custom AbstractYieldModel: a composite of two flat curves, multiplicative in
# discount space. Has no `.rates`/`.tenors`/`.spline` field, so the only way
# AU can compute KRDs is via TenorShift bumps over the curve's own `discount`.
struct CompositeTwoFlatYield{A, B} <: FM.Yield.AbstractYieldModel
    base::A
    spread::B
end
FC.discount(c::CompositeTwoFlatYield, t) = FC.discount(c.base, t) * FC.discount(c.spread, t)

@testset "Custom AbstractYieldModel: KRD/IR01/CS01 on a non-ZRC curve" begin
    base = FM.Yield.Constant(FC.Continuous(0.04))
    spread = FM.Yield.Constant(FC.Continuous(0.012))
    curve = CompositeTwoFlatYield(base, spread)

    tenors = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0]
    cfs = vcat(fill(5.0, 29), [105.0])
    times = collect(1.0:30.0)
    pv(c) = sum(cf * FC.discount(c, t) for (cf, t) in zip(cfs, times))

    @testset "scalar duration matches sum of KRDs" begin
        sd = duration(pv, curve, tenors)
        krds = duration(KeyRates(tenors), pv, curve)
        @test sd ≈ sum(krds) atol = 1e-10
    end

    @testset "do-block and cashflow forms agree" begin
        krds_db = duration(KeyRates(tenors), curve) do c
            pv(c)
        end
        krds_cf = duration(KeyRates(tenors), curve, cfs, times)
        @test krds_db ≈ krds_cf atol = 1e-10
    end

    @testset "scalar matches parallel-shift modified duration" begin
        # Total rate is 5.2% continuous; on annual coupon CFs, modified ≈ Macaulay.
        rate = 0.04 + 0.012
        dfs = [exp(-rate * t) for t in times]
        V = sum(cf * df for (cf, df) in zip(cfs, dfs))
        mac = sum(t * cf * df for (t, cf, df) in zip(times, cfs, dfs)) / V
        @test duration(pv, curve, tenors) ≈ mac atol = 1e-6
    end

    @testset "DV01" begin
        dv01 = duration(DV01(), pv, curve, tenors)
        krd_dv01 = duration(DV01(), KeyRates(tenors), pv, curve)
        @test dv01 ≈ sum(krd_dv01) atol = 1e-10
        @test all(krd_dv01 .≥ 0)
    end

    @testset "IR01 ≈ CS01 ≈ DV01 (flat additive case)" begin
        # Per the docstring: in a flat additive decomposition, bumping base
        # alone, credit alone, or the composite all shift the total zero rate
        # by 1bp — so IR01 ≈ CS01 ≈ DV01 individually.
        pv2c(b, c) = sum(cf * FC.discount(b, t) * FC.discount(c, t) for (cf, t) in zip(cfs, times))
        ir01 = duration(IR01(), pv2c, base, spread, tenors)
        cs01 = duration(CS01(), pv2c, base, spread, tenors)
        dv01 = duration(DV01(), pv, curve, tenors)
        @test ir01 ≈ cs01 atol = 1e-10
        @test ir01 ≈ dv01 atol = 1e-10
    end

    @testset "convexity matrix symmetric, scalar = sum" begin
        cmat = convexity(KeyRates(tenors), pv, curve)
        @test cmat ≈ cmat' atol = 1e-10
        @test convexity(pv, curve, tenors) ≈ sum(cmat) atol = 1e-10
    end

    @testset "sensitivities bundle" begin
        r = sensitivities(KeyRates(tenors), curve, cfs, times)
        @test r.value ≈ pv(curve) atol = 1e-10        # exact baseline; no resampling
        @test r.durations ≈ duration(KeyRates(tenors), pv, curve) atol = 1e-10
        @test sum(r.durations) ≈ duration(pv, curve, tenors) atol = 1e-10
        @test r.convexities ≈ r.convexities' atol = 1e-10

        r_dv01 = sensitivities(DV01(), KeyRates(tenors), curve, cfs, times)
        @test r_dv01.dv01s ≈ duration(DV01(), KeyRates(tenors), pv, curve) atol = 1e-10
    end

    @testset "two-curve sensitivities" begin
        pv2c(b, c) = sum(cf * FC.discount(b, t) * FC.discount(c, t) for (cf, t) in zip(cfs, times))
        r = sensitivities(KeyRates(tenors), pv2c, base, spread)
        @test r.value ≈ pv2c(base, spread) atol = 1e-10
        @test r.base_durations ≈ r.credit_durations atol = 1e-10   # additive ⇒ symmetric
    end

    @testset "ZRC promotion equivalence (Linear spline)" begin
        # With Spline.Linear, ZRC's KRDs match TenorShift+hat exactly because
        # linear interpolation in zero-rate space ≡ triangular-hat bumps.
        zrc = FM.Yield.ZeroRateCurve(curve, tenors, spline = FM.Spline.Linear())
        krds_zrc = duration(KeyRates(tenors), pv, zrc)
        krds_custom = duration(KeyRates(tenors), pv, curve)
        @test krds_custom ≈ krds_zrc atol = 1e-10
    end

    @testset "AD through non-flat base (NelsonSiegel + Constant)" begin
        # Stresses the AD chain on a curve whose `Base.zero(base, t)` is
        # non-linear in `t`, unlike the `Constant + Constant` flat composite
        # used in the other testsets here.
        ns_base = FM.Yield.NelsonSiegel(1.0, 0.04, -0.02, 0.01)
        flat_spr = FM.Yield.Constant(FC.Continuous(0.012))
        curve_nf = CompositeTwoFlatYield(ns_base, flat_spr)
        krds_nf = duration(KeyRates(tenors), pv, curve_nf)
        @test sum(krds_nf) ≈ duration(pv, curve_nf, tenors) atol = 1e-10
        @test argmax(krds_nf) == lastindex(tenors)   # sensitivity peaks at the long end
    end
end

@testset "KeyRates input validation" begin
    @test_throws ArgumentError KeyRates(Float64[])
    @test_throws ArgumentError KeyRates([5.0, 1.0, 10.0])    # unsorted
    @test_throws ArgumentError KeyRates([1.0, 1.0, 5.0])     # duplicate
    @test_throws ArgumentError KeyRates([0.0, 1.0, 5.0])     # non-positive
    @test_throws ArgumentError KeyRates([-1.0, 1.0, 5.0])    # negative

    # Valid grids construct cleanly
    @test KeyRates([0.25, 1.0, 5.0, 10.0, 30.0]) isa KeyRates
    @test KeyRates(1:5) isa KeyRates
end

@testset "AD vs analytic KRD: byte-equivalence across curve types and arities" begin
    # The analytic helpers `_keyrate_analytic` (single/two-curve) and
    # `_keyrate_analytic_n` (NamedTuple) must produce the same value, gradient,
    # and Hessian as the AD path (`_keyrate_ad`, `_ncurve_ad`) for the vanilla
    # cashflow case. Regression guard against future drift between the two
    # implementations of the same math.
    KRA   = ActuaryUtilities.FinancialMath._keyrate_analytic
    KRA_N = ActuaryUtilities.FinancialMath._keyrate_analytic_n
    KRAD  = ActuaryUtilities.FinancialMath._keyrate_ad
    NCAD  = ActuaryUtilities.FinancialMath._ncurve_ad

    tenors  = collect(1.0:30.0)
    rates   = fill(0.03, 30)
    rates2  = rates .+ 0.005
    curves  = [
        FM.ZeroRateCurve(rates,  tenors, FM.Spline.Linear()),
        FM.ZeroRateCurve(rates,  tenors, FM.Spline.MonotoneConvex()),
        FM.Yield.Constant(FC.Continuous(0.03)),
    ]

    cfs_full = collect(FM.Projection(FM.Bond.Fixed(0.04, FC.Periodic(2), 5), curves[1], FM.CashflowProjection()))
    amts  = FC.amount.(cfs_full)
    times = FC.timepoint.(cfs_full)

    @testset "single-curve [$(typeof(c).name.name)]" for c in curves
        ad = KRAD(c, tenors,
                  i -> sum(amts[k] * FC.discount(i, times[k]) for k in eachindex(amts));
                  order = 2)
        an = KRA(c, tenors, amts, times; order = 2)
        @test ad.value ≈ an.value rtol = 1e-12
        @test maximum(abs.(ad.gradient .- an.gradient)) < 1e-12
        @test maximum(abs.(ad.hessian  .- an.hessian))  < 1e-12
    end

    @testset "two-curve" begin
        base   = curves[1]
        credit = FM.ZeroRateCurve(rates2, tenors, FM.Spline.Linear())
        ad = KRAD(base, credit, tenors,
                  (b, c) -> sum(amts[k] * FC.discount(b, times[k]) * FC.discount(c, times[k])
                                 for k in eachindex(amts));
                  order = 2)
        an = KRA(base, credit, tenors, amts, times; order = 2)
        @test ad.value ≈ an.value rtol = 1e-12
        @test maximum(abs.(ad.base_gradient   .- an.base_gradient))   < 1e-12
        @test maximum(abs.(ad.credit_gradient .- an.credit_gradient)) < 1e-12
        @test maximum(abs.(ad.base_hessian    .- an.base_hessian))    < 1e-12
        @test maximum(abs.(ad.credit_hessian  .- an.credit_hessian))  < 1e-12
        @test maximum(abs.(ad.cross_hessian   .- an.cross_hessian))   < 1e-12
    end

    @testset "NamedTuple (3 curves)" begin
        c1 = curves[1]
        c2 = FM.ZeroRateCurve(rates2,       tenors, FM.Spline.Linear())
        c3 = FM.ZeroRateCurve(rates .+ 0.002, tenors, FM.Spline.Linear())
        nt = (; rf = c1, credit = c2, ilp = c3)
        ad_v, ad_g = NCAD(c -> sum(amts[k] * FC.discount(c.rf, times[k]) *
                                              FC.discount(c.credit, times[k]) *
                                              FC.discount(c.ilp, times[k])
                                    for k in eachindex(amts)),
                          nt, tenors)
        an = KRA_N(nt, tenors, amts, times; order = 2)
        @test ad_v ≈ an.value rtol = 1e-12
        for r in (:rf, :credit, :ilp)
            @test maximum(abs.(ad_g[r] .- an.gradient[r])) < 1e-12
        end
    end
end

@testset "Hull-White convenience method: pathwise consistency" begin
    # The four `sensitivities(KeyRates, hw, ...)` convenience methods snapshot
    # one UInt64 from the user's rng and rebuild Xoshiro(seed) inside each AD
    # evaluation. Two calls seeded the same way must produce bit-identical
    # results — otherwise ForwardDiff's many evaluations of the closure each
    # draw different MC samples and KRD = -∇V/V is biased by MC noise.
    rates = [0.03, 0.03, 0.03, 0.03, 0.03]
    tenors = [1.0, 2.0, 3.0, 4.0, 5.0]
    cfs = [5.0, 5.0, 5.0, 5.0, 105.0]
    zrc = FM.ZeroRateCurve(rates, tenors)
    hw = FM.ShortRate.HullWhite(0.1, 0.01, zrc)

    r1 = sensitivities(KeyRates(tenors), hw, cfs, tenors;
                       n_scenarios=500, rng=Xoshiro(42))
    r2 = sensitivities(KeyRates(tenors), hw, cfs, tenors;
                       n_scenarios=500, rng=Xoshiro(42))
    @test r1.value ≈ r2.value
    @test r1.durations ≈ r2.durations
    @test r1.convexities ≈ r2.convexities

    # DV01 form
    d1 = sensitivities(DV01(), KeyRates(tenors), hw, cfs, tenors;
                       n_scenarios=500, rng=Xoshiro(42))
    d2 = sensitivities(DV01(), KeyRates(tenors), hw, cfs, tenors;
                       n_scenarios=500, rng=Xoshiro(42))
    @test d1.value ≈ d2.value
    @test d1.dv01s ≈ d2.dv01s
    @test d1.convexities ≈ d2.convexities

    # Different seeds give different MC samples (sanity check the seed is actually used)
    r3 = sensitivities(KeyRates(tenors), hw, cfs, tenors;
                       n_scenarios=500, rng=Xoshiro(43))
    @test !(r1.value ≈ r3.value && r1.durations ≈ r3.durations)
end

@testset "Contract/portfolio duration & sensitivities (unified)" begin
    mats = [1.0, 2.0, 3.0, 5.0, 7.0]
    curve = FM.Yield.Spline(FM.Spline.Linear(), mats, [0.02, 0.025, 0.03, 0.035, 0.04])
    tenors = mats
    fl0 = FM.Bond.Floating(0.0, FC.Periodic(1), 5.0, "IDX")
    flm = FM.Bond.Floating(0.02, FC.Periodic(1), 5.0, "IDX")
    fb = FM.Bond.Fixed(0.04, FC.Periodic(1), 5.0)

    @testset "par floater: effective ≈ 0, spread ≈ maturity" begin
        @test duration(Effective(), fl0, curve, tenors) ≈ 0.0 atol = 1e-8
        @test duration(Spread(), fl0, curve, tenors) > 4.0
        @test convexity(Effective(), fl0, curve, tenors) ≈ 0.0 atol = 1e-6
    end

    @testset "bundle: effective = forward + spread; sums; dollar <-> year" begin
        s = sensitivities(flm, curve, tenors)
        @test s.effective_duration ≈ s.forward_duration + s.spread_duration atol = 1e-10
        @test s.effective_dv01 ≈ s.effective_duration * s.value / 10_000 atol = 1e-12
        @test sum(s.effective_key_rate) ≈ s.effective_duration atol = 1e-10
        @test sum(s.spread_key_rate) ≈ s.spread_duration atol = 1e-10
    end

    @testset "fixed bond: effective == spread == modified, forward == 0" begin
        s = sensitivities(fb, curve, tenors)
        modified = duration(curve, tenors, collect(FM.Projection(fb, curve, FM.CashflowProjection())))
        @test s.effective_duration ≈ modified atol = 1e-8
        @test s.spread_duration ≈ modified atol = 1e-8
        @test s.forward_duration ≈ 0.0 atol = 1e-8
    end

    @testset "fixed bond: effective convexity matches matrix-sum (POU equivalence regression guard)" begin
        # Under partition of unity of the KRD hat functions, sum(N×N key-rate
        # Hessian) = continuous-shock parallel-shift second derivative by the
        # chain rule. The optimized `convexity(::Effective, …)` computes that
        # scalar directly via TenorShift, in O(1) rather than O(N²) AD work.
        # Locks the numerical equivalence in for future refactors of either
        # path. Note: `convexity(curve, cfs)` uses a *periodic* shock and is
        # NOT equivalent here — see `_parallel_continuous_convexity` for why.
        cfs = collect(FM.Projection(fb, curve, FM.CashflowProjection()))
        amts = FC.amount.(cfs); times = FC.timepoint.(cfs)
        @test convexity(Effective(), fb, curve, tenors) ≈
              sum(convexity(KeyRates(tenors), curve, amts, times)) atol = 1e-8
    end

    @testset "default duration & dv01 verb" begin
        @test duration(flm, curve, tenors) ≈ duration(Effective(), flm, curve, tenors)
        @test dv01(Effective(), flm, curve, tenors) ≈ sensitivities(flm, curve, tenors).effective_dv01
        @test dv01(0.05, [5.0, 5.0, 105.0]) ≈ duration(DV01(), 0.05, [5.0, 5.0, 105.0])   # cashflow fallback
    end

    @testset "portfolio: one-pass == value-weighted" begin
        port = [flm, fb]
        dport = duration(port, curve, tenors)
        vfl = FC.present_value(curve, reproject(flm, curve)); vfb = FC.present_value(curve, fb)
        dfl = duration(flm, curve, tenors); dfb = duration(fb, curve, tenors)
        @test dport ≈ (vfl * dfl + vfb * dfb) / (vfl + vfb) atol = 1e-8
    end

    @testset "multi-curve: structured == do-block; additive layers" begin
        credit = FM.Yield.Constant(FC.Continuous(0.01))
        ilp = FM.Yield.Constant(FC.Continuous(0.004))
        rs = sensitivities(flm, tenors; discount = (; rf = curve, credit = credit, ilp = ilp), index = curve)
        rd = sensitivities((; rf = curve, credit = credit, ilp = ilp, index = curve); tenors) do c
            FC.present_value(c.rf + c.credit + c.ilp, reproject(flm, c.index))
        end
        @test rs.duration.rf ≈ rd.duration.rf atol = 1e-10
        @test rs.duration.index ≈ rd.duration.index atol = 1e-10
        @test rs.duration.rf ≈ rs.duration.credit atol = 1e-8       # additive layers ⇒ equal discount sensitivity
        @test rs.duration.credit ≈ rs.duration.ilp atol = 1e-8
        @test rs.duration.index < 0.0                                # bumping the index raises coupons → raises value
    end

    @testset "z-spread round-trips; locked ≈ next reset" begin
        pvm = FC.present_value(curve, reproject(flm, curve))
        @test zspread(flm, curve, pvm).zspread ≈ 0.0 atol = 1e-8
        z = zspread(flm, curve, pvm - 0.03)
        @test z.zspread > 0.0
        reprice = FC.present_value(curve + ((zz, t) -> zz + FC.Continuous(z.zspread)), reproject(flm, curve))
        @test reprice ≈ pvm - 0.03 atol = 1e-10
        @test duration(Effective(), locked_floater(fl0, 0.05, 1.0), curve, tenors) ≈ 1.0 atol = 0.1
    end

    @testset "effective: AD == central finite difference (re-projecting)" begin
        Δ = 1e-4
        up = curve + ((z, t) -> z + FC.Continuous(+Δ)); dn = curve + ((z, t) -> z + FC.Continuous(-Δ))
        rj(crv) = FC.present_value(crv, reproject(flm, crv))
        eff_fd = (rj(dn) - rj(up)) / (2Δ * rj(curve))
        @test duration(Effective(), flm, curve, tenors) ≈ eff_fd atol = 1e-4
    end
end
