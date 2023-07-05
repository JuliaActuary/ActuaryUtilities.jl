using ActuaryUtilities

using Dates
using Test

const FM = ActuaryUtilities.FinanceModels
const FC = ActuaryUtilities.FinanceCore



include("risk_measures.jl")

#convenience function to wrap scalar into default Rate type
p(rate) = Yields.Periodic(rate, 1)

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
    @test all(accum_offset([0.9, 0.8, 0.7], op=+) .== [1.0, 1.9, 2.7])
    @test all(accum_offset([0.9, 0.8, 0.7], op=+, init=2) .== [2.0, 2.9, 3.7])

    @test all(accum_offset(1:5, op=+) .== [1, 2, 4, 7, 11])
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



        @test all(present_values(0.00, [1, 1, 1]) .≈ [3, 2, 1])
        @test all(present_values(0.00, [1, 1, 1], [0, 1, 2]) .≈ [3, 2, 1])
        @test all(present_values(0.00, [1, 1, 1], [1, 2, 3]) .≈ [3, 2, 1])
        @test all(present_values(0.00, [1, 1, 1], [1, 2, 3]) .≈ [3, 2, 1])
        @test all(present_values(0.01, [1, 2, 3]) .≈ [5.862461552497766, 4.921086168022744, 2.9702970297029707])

        cf = [100, 100]

        ts = [0.5, 1]

        @test pv(0.05, cf, ts) ≈ 100 / 1.05^0.5 + 100 / 1.05^1

        @test price(0.05, cf, ts) ≈ pv(0.05, cf, ts)
        @test price(0.05, -1 .* cf, ts) ≈ abs(pv(0.05, cf, ts))


    end






end

@testset "Breakeven time" begin

    @testset "basic" begin
        @test breakeven(0.10, [-10, 1, 2, 3, 4, 8]) == 5
        @test breakeven(0.10, [-10, 15, 2, 3, 4, 8]) == 1
        @test breakeven(0.10, [-10, 15, 2, 3, 4, 8]) == 1
        @test breakeven(0.10, [10, 15, 2, 3, 4, 8]) == 0
        @test isnothing(breakeven(0.10, [-10, -15, 2, 3, 4, 8]))
    end

    @testset "timepoints" begin
        times = [t for t in 0:5]
        @test breakeven(0.10, [-10, 1, 2, 3, 4, 8], times) == 5
        @test breakeven(0.10, [-10, 15, 2, 3, 4, 8], times) == 1
        @test breakeven(0.10, [-10, 15, 2, 3, 4, 8], times) == 1
        @test isnothing(breakeven(0.10, [-10, -15, 2, 3, 4, 8], times))
    end
end

@testset "moic" begin

    # https://bankingprep.com/multiple-on-invested-capital/
    ex1 = [-100; [t == 200 ? 100 * 1.067^t : 0 for t in 1:200]]
    @test moic(ex1) ≈ 429421.59914697794


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
        V = present_value(0.04, cfs, times)

        @test duration(Macaulay(), 0.04, cfs, times) ≈ 1.777570320376649
        @test duration(Modified(), 0.04, cfs, times) ≈ 1.777570320376649 / (1 + 0.04)
        @test duration(0.04, cfs, times) ≈ 1.777570320376649 / (1 + 0.04)

        # wikipedia example defines DV01 as a per point change, but industry practice is per basis point. Ref Issue #96
        @test duration(DV01(), 0.04, cfs, times) ≈ 1.777570320376649 / (1 + 0.04) * V / 10000

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

        @test isapprox(present_value(int, cfs, times), 100.00, atol=1e-2)
        @test isapprox(duration(Macaulay(), int, cfs, times), 4.26, atol=1e-2)
    end

    @testset "Primer example" begin
        # from https://math.illinoisstate.edu/krzysio/Primer.pdf
        # the duration tests are commented out because I think the paper is wrong on the duration?
        cfs = [0, 0, 0, 0, 1.0e6]
        times = 1:5

        @test isapprox(present_value(0.04, cfs, times), 821927.11, atol=1e-2)
        # @test isapprox(duration(0.04,cfs,times),4.76190476,atol=1e-6)
        @test isapprox(convexity(0.04, cfs, times), 27.7366864, atol=1e-6)
        @test isapprox(convexity(0.04, cfs), 27.7366864, atol=1e-6)
        # the same, but with a functional argument
        value(i) = present_value(i, cfs, times)
        # @test isapprox(duration(0.04,value),4.76190476,atol=1e-6)
        @test isapprox(convexity(0.04, value), 27.7366864, atol=1e-6)
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
        cfs = [5 0 0
            0 5 105]

        @test duration(0.03, sum(cfs, dims=1), times) ≈ 2.780101622010806

        cfs = [5 0
            5 0
            0 105]

        @test duration(0.03, sum(cfs, dims=2), times) ≈ 2.780101622010806


    end

    @testset "Key Rate Durations" begin
        default_shift = 0.001

        @test KeyRate(5) == KeyRateZero(5)
        @test KeyRate(5) == KeyRateZero(5, default_shift)
        @test KeyRatePar(5) == KeyRatePar(5, default_shift)

        c = FM.Yield.Constant(FC.Periodic(0.04, 2))

        cp = ActuaryUtilities._krd_new_curve(KeyRatePar(5), c, 1:10)
        cz = ActuaryUtilities._krd_new_curve(KeyRateZero(5), c, 1:10)

        # test some relationships between par and zero curve
        @test FM.par(cp, 5) ≈ FM.par(c, 5) + default_shift atol = 0.0002 # 0.001 is the default shift
        @test FM.par(cp, 4) ≈ FC.Periodic(0.04, 2) atol = 0.0001
        @test zero(cp, 5) > FM.par(cp, 5)
        @test zero(cp, 6) < FM.par(cp, 6)

        @testset "FEH123" begin
            # http://www.financialexamhelp123.com/key-rate-duration/

            #test some curve properties


            bond = (
                cfs=[0.02 for t in 1:10],
                times=collect(0.5:0.5:5)
            )
            bond.cfs[end] += 1.0

            @test duration(KeyRatePar(1), c, bond.cfs, bond.times) ≈ 0.0 atol = 0.01
            @test duration(KeyRatePar(2), c, bond.cfs, bond.times) ≈ 0.0 atol = 0.01
            @test duration(KeyRatePar(3), c, bond.cfs, bond.times) ≈ 0.0 atol = 0.01
            @test duration(KeyRatePar(4), c, bond.cfs, bond.times) ≈ 0.0 atol = 0.01
            @test duration(KeyRatePar(5), c, bond.cfs, bond.times) ≈ 4.45 atol = 0.05

            bond = (times=[1, 2, 3, 4, 5], cfs=[0, 0, 0, 0, 100])
            c = FC.Continuous(0.05)
            @test duration(KeyRateZero(1), c, bond.cfs, bond.times) ≈ 0.0 atol = 1e-10
            @test duration(KeyRateZero(2), c, bond.cfs, bond.times) ≈ 0.0 atol = 1e-10
            @test duration(KeyRateZero(3), c, bond.cfs, bond.times) ≈ 0.0 atol = 1e-10
            @test duration(KeyRateZero(4), c, bond.cfs, bond.times) ≈ 0.0 atol = 1e-10
            @test duration(KeyRateZero(5), c, bond.cfs, bond.times) ≈ duration(c, bond.cfs, bond.times) atol = 0.1




        end
    end

end

@testset "spread" begin
    cfs = fill(10, 10)
    @test spread(0.04, 0.05, cfs) ≈ FC.Periodic(0.01, 1)
    @test spread(FC.Continuous(0.04), FC.Continuous(0.05), cfs) ≈ FC.Periodic(1)(FC.Continuous(0.05)) - FC.Periodic(1)(FC.Continuous(0.04))

    # 2021-03-31 rates from Treasury.gov
    rates = [0.01, 0.01, 0.03, 0.05, 0.07, 0.16, 0.35, 0.92, 1.40, 1.74, 2.31, 2.41] ./ 100
    mats = [1 / 12, 2 / 12, 3 / 12, 6 / 12, 1, 2, 3, 5, 7, 10, 20, 30]

    y = FM.fit(FM.Spline.Linear(), FM.CMTYield.(rates, mats), FM.Fit.Bootstrap())

    y2 = y + FC.Periodic(0.01, 1)

    s = spread(y, y2, cfs)

    @test s ≈ FC.Periodic(0.01, 1) atol = 0.002
end