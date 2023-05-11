using ActuaryUtilities

using Dates
using Test

const Yields = ActuaryUtilities.Yields
include("test_utils.jl")
include("risk_measures.jl")
include("derivatives.jl")

#convenience function to wrap scalar into default Rate type
p(rate) = Yields.Periodic(rate,1)

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
    @test all(accum_offset([0.9, 0.8, 0.7]) .== [1.0,0.9,1.0 * 0.9 * 0.8])
    @test all(accum_offset([0.9, 0.8, 0.7],op=+) .== [1.0,1.9,2.7])
    @test all(accum_offset([0.9, 0.8, 0.7],op=+,init=2) .== [2.0,2.9,3.7])

    @test all(accum_offset(1:5,op=+) .== [1,2,4,7,11])
    @test all(accum_offset(1:5) .== [1,1,2,6,24])
    @test all(accum_offset([1, 2, 3]) .== [1,1,2])
end

@testset "financial calcs" begin

    @testset "pv" begin
        cf = [100, 100]
        
        @test pv(0.05, cf) ≈ cf[1] / 1.05 + cf[2] / 1.05^2
        @test price(0.05, cf) ≈ pv(0.05, cf)

        # this vector came from Numpy Financial's test suite with target of 122.89, but that assumes payments are begin of period
        # 117.04 comes from Excel verification with NPV function
        @test isapprox(pv(0.05, [-15000, 1500, 2500, 3500, 4500, 6000]), 117.04, atol = 1e-2)

        cfs = ones(3)
        @test present_values(Yields.Constant(0.0),cfs) == [3,2,1]
        pvs = present_values(Yields.Constant(0.1),cfs) 
        @test pvs[3] ≈ 1 / 1.1
        @test pvs[2] ≈ (1 / 1.1 + 1) / 1.1



        @test all(present_values(0.00, [1,1,1]) .≈ [3,2,1])
        @test all(present_values(0.00, [1,1,1],[0,1,2]) .≈ [3,2,1])
        @test all(present_values(0.00, [1,1,1],[1,2,3]) .≈ [3,2,1])
        @test all(present_values(0.00, [1,1,1],[1,2,3]) .≈ [3,2,1])
        @test all(present_values(0.01, [1,2,3]) .≈ [ 5.862461552497766,4.921086168022744,2.9702970297029707])
        @test all(present_values(Yields.Forward([0.1,0.2]), [10,20],[0,1]) ≈  [28.18181818181818,18.18181818181818])
        @test all(present_values([0.1,0.2], [10,20],[0,1]) ≈  [28.18181818181818,18.18181818181818])

        # issue #58
        r = Yields.Periodic(0.02,1)
        @test present_value(r,[1,2]) ≈ 1 / 1.02 + 2 / 1.02^2

    end

    @testset "pv with timepoints" begin
        cf = [100, 100]

        @test pv(0.05, cf, [1,2]) ≈ cf[1] / 1.05 + cf[2] / 1.05^2
    end


end

    @testset "Breakeven time" begin

    @testset "basic" begin
        @test breakeven(0.10, [-10,1,2,3,4,8]) == 5
        @test breakeven(0.10, [-10,15,2,3,4,8]) == 1
        @test breakeven(0.10, [-10,15,2,3,4,8]) == 1
        @test breakeven(0.10, [10,15,2,3,4,8]) == 0
        @test isnothing(breakeven(0.10, [-10,-15,2,3,4,8]))
    end

    @testset "basic with vector interest" begin
        @test breakeven(0.0,[-10,1,2,3,4], [1,2,3,4,5]) == 5
        # 
        @test isnothing(breakeven([0.0,0.0,0.0,0.0,0.1], [-10,1,2,3,4], [1,2,3,4,5]))
        @test breakeven([0.0,0.0,0.0,0.0,-0.5], [-10,1,2,3,4], [1,2,3,4,5]) == 5
        @test breakeven([0.0,0.0,0.0,-0.9,-0.5],[-10,1,2,3,4], [1,2,3,4,5]) == 4
        @test breakeven([0.1,0.1,0.2,0.1,0.1], [-10,1,12,3,4], [1,2,3,4,5]) == 3
    end

    @testset "timepoints" begin
        times = [t for t in 0:5]
        @test breakeven(0.10,[-10,1,2,3,4,8], times) == 5
        @test breakeven(0.10,[-10,15,2,3,4,8], times) == 1
        @test breakeven(0.10,[-10,15,2,3,4,8], times) == 1
        @test isnothing(breakeven(0.10,[-10,-15,2,3,4,8], times))
    end
end

@testset "moic" begin

    # https://bankingprep.com/multiple-on-invested-capital/
    ex1 = [-100;[t == 200 ? 100 * 1.067^t : 0 for t in 1:200]]
    @test moic(ex1) ≈ 429421.59914697794
    

    ex2 = ex1[end] *= 0.5
    @test moic(ex1) ≈ 429421.59914697794 * 0.5


end


@testset "duration and convexity" begin

    # per issue #74
    @testset "generators" begin
        g = (10 for t in 1:10)
        v = collect(g)
        i = Yields.Constant(0.04)
        @test pv(i,g) ≈ pv(i,[10 for t in 1:10])
        @test duration(0.04,g) ≈ duration(0.04,v)
        @test duration(i,g) ≈ duration(i,v)
        @test convexity(0.04,g) ≈ convexity(0.04,v)
    end
    
    @testset "wikipedia example" begin
        times = [0.5,1,1.5,2]
        cfs = [10,10,10,110]
        V = present_value(0.04, cfs, times)

        @test duration(Macaulay(), 0.04, cfs, times) ≈ 1.777570320376649
        @test duration(Modified(), 0.04, cfs, times) ≈ 1.777570320376649 / (1 + 0.04)
        @test duration(0.04, cfs, times) ≈ 1.777570320376649 / (1 + 0.04)

        # wikipedia example defines DV01 as a per point change, but industry practice is per basis point. Ref Issue #96
        @test duration(DV01(), 0.04, cfs, times) ≈ 1.777570320376649 / (1 + 0.04) * V / 10000
        
        # test with a Rate
        r = Yields.Periodic(0.04,1)
        @test duration(Macaulay(), r, cfs, times) ≈ 1.777570320376649
        @test duration(Modified(), r, cfs, times) ≈ 1.777570320376649 / (1 + 0.04)
        @test duration(r, cfs, times) ≈ 1.777570320376649 / (1 + 0.04)
        @test duration(DV01(), r, cfs, times) ≈ 1.777570320376649 / (1 + 0.04) * V / 10000
        
        #test without times
        r = Yields.Periodic(0.04,1)
        @test duration(Macaulay(), r, cfs) ≈ duration(Macaulay(), r, cfs, 1:4)
        @test duration(Modified(), r, cfs) ≈ duration(Modified(), r, cfs,1:4)
        @test duration(r, cfs) ≈ duration(r, cfs,1:4)
        @test duration(DV01(), r, cfs) ≈ duration(DV01(), r, cfs,1:4)

        @test duration(Yields.Constant(0.04), cfs, times) ≈ 1.777570320376649 / (1 + 0.04)
        @test duration(Yields.Constant(0.04), -1 .* cfs, times) ≈ 1.777570320376649 / (1 + 0.04) atol=0.00001
        @test duration(Yields.Forward([0.04,0.04]), cfs, times) ≈ 1.777570320376649 / (1 + 0.04) atol=0.00001

        # test that dispatch resolves the ambiguity between duration(Yield,vec) and duration(Yield, function)
        @test duration(Yields.Constant(0.03),cfs) > 0
        @test convexity(Yields.Constant(0.03),cfs) > 0
    end

    @testset "finpipe example" begin
        # from https://www.finpipe.com/duration-macaulay-and-modified-duration-convexity/

        cfs = zeros(10) .+ 3.75
        cfs[10] += 100

        times = 0.5:0.5:5.0
        int = (1 + 0.075 / 2)^2 - 1 # convert bond yield to effective yield

        @test isapprox(present_value(int, cfs, times), 100.00, atol = 1e-2)
        @test isapprox(duration(Macaulay(), int, cfs, times), 4.26, atol = 1e-2)
    end

    @testset "Primer example" begin
        # from https://math.illinoisstate.edu/krzysio/Primer.pdf
        # the duration tests are commented out because I think the paper is wrong on the duration?
        cfs = [0,0,0,0,1.0e6]
        times = 1:5

        @test isapprox(present_value(0.04, cfs, times), 821927.11, atol = 1e-2)
        # @test isapprox(duration(0.04,cfs,times),4.76190476,atol=1e-6)
        @test isapprox(convexity(0.04, cfs, times), 27.7366864, atol = 1e-6)
        @test isapprox(convexity(0.04, cfs), 27.7366864, atol = 1e-6)
        # the same, but with a functional argument
        value(i) = ActuaryUtilities.present_value_differentiable(i, cfs, times)
        # @test isapprox(duration(0.04,value),4.76190476,atol=1e-6)
        @test isapprox(convexity(0.04, value), 27.7366864, atol = 1e-6)
    end

    @testset "Quantlib" begin
    # https://mhittesdorf.wordpress.com/2013/03/12/introduction-to-quantlib-duration-and-convexity/
        cfs = [5,5,105]
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

        @test duration(0.03, sum(cfs,dims=1), times) ≈ 2.780101622010806

        cfs = [5 0
               5 0 
               0 105]

        @test duration(0.03, sum(cfs,dims=2), times) ≈ 2.780101622010806


    end

    @testset "Key Rate Durations" begin
        default_shift = 0.001

        @test KeyRate(5) == KeyRateZero(5)
        @test KeyRate(5) == KeyRateZero(5,default_shift)
        @test KeyRatePar(5) == KeyRatePar(5,default_shift)
        
        c = Yields.Constant(Yields.Periodic(0.04,2))

        cp = ActuaryUtilities._krd_new_curve(KeyRatePar(5),c,1:10)
        cz = ActuaryUtilities._krd_new_curve(KeyRateZero(5),c,1:10)

        # test some relationships between par and zero curve
        @test Yields.par(cp,5) ≈ Yields.par(c,5) + default_shift atol = 0.0002 # 0.001 is the default shift
        @test Yields.par(cp,4) ≈ Yields.Periodic(0.04,2) atol = 0.0001           
        @test Yields.zero(cp,5) > Yields.par(cp,5)
        @test Yields.zero(cp,6) < Yields.par(cp,6)

        @testset "FEH123" begin
            # http://www.financialexamhelp123.com/key-rate-duration/

            #test some curve properties


            bond = parbond(0.04,5)

            @test duration(KeyRatePar(1),c,bond.cfs,bond.times) ≈ 0.0 atol = 0.01
            @test duration(KeyRatePar(2),c,bond.cfs,bond.times) ≈ 0.0 atol = 0.01
            @test duration(KeyRatePar(3),c,bond.cfs,bond.times) ≈ 0.0 atol = 0.01
            @test duration(KeyRatePar(4),c,bond.cfs,bond.times) ≈ 0.0 atol = 0.01
            @test duration(KeyRatePar(5),c,bond.cfs,bond.times) ≈ 4.45 atol = 0.05
            
            bond =(times=[1,2,3,4,5],cfs=[0,0,0,0,100])
            @test duration(KeyRateZero(1),c,bond.cfs,bond.times) ≈ 0.0 
            @test duration(KeyRateZero(2),c,bond.cfs,bond.times) ≈ 0.0 
            @test duration(KeyRateZero(3),c,bond.cfs,bond.times) ≈ 0.0 
            @test duration(KeyRateZero(4),c,bond.cfs,bond.times) ≈ 0.0 
            @test duration(KeyRateZero(5),c,bond.cfs,bond.times) ≈ duration(c,bond.cfs,bond.times) atol = 0.1
            



        end
    end

end

@testset "spread" begin
    cfs = fill(10,10)
    @test spread(0.04,0.05,cfs) ≈ Yields.Periodic(0.01,1)
    @test spread(Yields.Continuous(0.04),Yields.Continuous(0.05),cfs) ≈ Yields.Periodic(1)(Yields.Continuous(0.05)) - Yields.Periodic(1)(Yields.Continuous(0.04))

      # 2021-03-31 rates from Treasury.gov
    rates =[0.01, 0.01, 0.03, 0.05, 0.07, 0.16, 0.35, 0.92, 1.40, 1.74, 2.31, 2.41] ./ 100
    mats = [1/12, 2/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30]
  
    y = Yields.CMT(rates,mats)

    y2 = y + Yields.Periodic(0.01,1)

    s = spread(y,y2,cfs)

    @test s ≈ Yields.Periodic(0.01,1) atol=0.002
end


include("run_doctests.jl")