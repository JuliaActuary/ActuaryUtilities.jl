using ActuaryUtilities

using Dates
using Test

const Yields = ActuaryUtilities.Yields

import DayCounts

include("excel.jl")

@testset "Temporal functions" begin
    @testset "years_between" begin
        @test years_between(Date(2018, 9, 30), Date(2018, 9, 30)) == 0
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

    end

    @testset "pv with timepoints" begin
        cf = [100, 100]

        @test pv(0.05, cf, [1,2]) ≈ cf[1] / 1.05 + cf[2] / 1.05^2
    end

    @testset "pv with vector discount rates" begin
        cf = [100, 100]
        @test pv([0.0,0.05], cf) ≈ 100 / 1.0 + 100 / 1.05
        @test pv(ActuaryUtilities.Yields.Forward([0.0,0.05]), cf) ≈ 100 / 1.0 + 100 / 1.05
        @test pv([0.05,0.0], cf) ≈ 100 / 1.05 + 100 / 1.05
        @test pv([0.05,0.1], cf) ≈ 100 / 1.05 + 100 / 1.05 / 1.1

        ts = [0.5,1]
        @test pv(ActuaryUtilities.Yields.Forward([0.0,0.05], ts), cf, ts) ≈ 100 / 1.0 + 100 / 1.05^0.5 
        @test pv(ActuaryUtilities.Yields.Forward([0.05,0.0], ts), cf, ts) ≈ 100 / 1.05^0.5 + 100 / 1.05^0.5 
        @test pv(ActuaryUtilities.Yields.Forward([0.05,0.1], ts), cf, ts) ≈ 100 / 1.05^0.5 + 100 / (1.05^0.5) / (1.1^0.5)

        #without explicit Yields constructor
        @test pv([0.0,0.05], cf, ts) ≈ 100 / 1.0 + 100 / 1.05^0.5 

        @test price([0.0,0.05], cf, ts) ≈ pv([0.0,0.05], cf, ts)
        @test price([0.0,0.05], -1 .* cf, ts) ≈ abs(pv([0.0,0.05], cf, ts))

        
    end


    @testset "irr" begin

        v = [-70000,12000,15000,18000,21000,26000]
        
        # per Excel (example comes from Excel help text)
        @test isapprox(irr(v[1:2]), -0.8285714285714, atol = 0.001)
        @test isapprox(irr(v[1:3]), -0.4435069413346, atol = 0.001)
        @test isapprox(irr(v[1:4]), -0.1821374641455, atol = 0.001)
        @test isapprox(irr(v[1:5]), -0.0212448482734, atol = 0.001)
        @test isapprox(irr(v[1:6]),  0.0866309480365, atol = 0.001)

        # much more challenging to solve b/c of the overflow below zero
        cfs = [t % 10 == 0 ? -10 : 1.5 for t in 0:99]

        @test isapprox(irr(cfs), 0.06463163963925866, atol = 0.001)

        # issue #28
        cfs = [-8.728037307132952e7, 3.043754023830998e7, 2.963004184784189e7, 2.8803030748755097e7, 2.7956912111811966e7, 2.7092182051244527e7, 2.6209069543806538e7, 2.5307964329840004e7, 2.438961041057478e7, 2.3455084653011695e7, 2.2505925520018265e7, 2.154395414765592e7, 2.0571076113065004e7, 1.958930608135183e7, 1.8600627464895025e7, 1.7606980923262402e7, 1.661046149512893e7, 1.561312825963898e7, 1.461760481586352e7, 1.3626801207410209e7, 1.2644733969499402e7, 1.1675393687299855e7, 1.0722720151658386e7, 9.79075673433771e6, 8.883278741880089e6, 8.004445298876338e6, 7.1588010859461725e6, 6.351121678665243e6, 5.585860320479795e6, 4.8673895159943625e6, 4.19908059495347e6, 3.583538247530099e6, 3.022766488834396e6, 2.5181072324190177e6, 2.0701053881076649e6, 1.6782921224664208e6, 1.3410605489291362e6, 1.0556643097527474e6, 818348.5357315112, 624147.9373214925, 467849.788997191, 344241.752520618, 248285.65630649775, 175235.5475426321, 120677.87174498942, 80759.09804678289, 52186.83400936739, 32211.057718402008, 18589.51907385164, 9540.782278174447, 3688.4015341755294]
        @test irr(cfs,0:50) ≈ 0.3176680627111823

        # test the unsolvable

        @test isnothing(irr([100,100]))
        @test isnothing(irr([-1e8,0.,0.,0.],0:3))

    end

    @testset "numpy examples" begin

        @test isapprox(irr([-150000, 15000, 25000, 35000, 45000, 60000]),  0.0524,     atol = 1e-4)
        @test isapprox(irr([-100, 0, 0, 74]), -0.0955,     atol = 1e-4)
        @test isapprox(irr([-100, 39, 59, 55, 20]),  0.28095,    atol = 1e-4)
        @test isapprox(irr([-100, 100, 0, -7]), -0.0833,     atol = 1e-4)
        @test isapprox(irr([-100, 100, 0, 7]),  0.06206,    atol = 1e-4)

        # this has multiple roots, of which 0.709559 and 0.0886. Want to find the one closer to zero
        @test isapprox(irr([-5, 10.5, 1, -8, 1]),  0.0886,     atol = 1e-4)
    end

    @testset "xirr with float times" begin

    
        @test isapprox(irr([-100,100], [0,1]), 0.0, atol = 0.001)
        @test isapprox(irr([-100,110], [0,1]), 0.1, atol = 0.001)

    end

    @testset "xirr with real dates" begin

        v = [-70000,12000,15000,18000,21000,26000]
        dates = Date(2019, 12, 31):Year(1):Date(2024, 12, 31)
        times = map(d->DayCounts.yearfrac(dates[1], d, DayCounts.Thirty360()), dates)
    # per Excel (example comes from Excel help text)
        @test isapprox(irr(v[1:2], times[1:2]), -0.8285714285714, atol = 0.001)
        @test isapprox(irr(v[1:3], times[1:3]), -0.4435069413346, atol = 0.001)
        @test isapprox(irr(v[1:4], times[1:4]), -0.1821374641455, atol = 0.001)
        @test isapprox(irr(v[1:5], times[1:5]), -0.0212448482734, atol = 0.001)
        @test isapprox(irr(v[1:6], times[1:6]),  0.0866309480365, atol = 0.001)

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

@testset "duration" begin
    
    @testset "wikipedia example" begin
        times = [0.5,1,1.5,2]
        cfs = [10,10,10,110]
        V = present_value(0.04, cfs, times)

        @test duration(Macaulay(), 0.04, cfs, times) ≈ 1.777570320376649
        @test duration(Modified(), 0.04, cfs, times) ≈ 1.777570320376649 / (1 + 0.04)
        @test duration(0.04, cfs, times) ≈ 1.777570320376649 / (1 + 0.04)
        @test duration(DV01(), 0.04, cfs, times) ≈ 1.777570320376649 / (1 + 0.04) * V / 100
        
        @test duration(Yields.Constant(0.04), cfs, times) ≈ 1.777570320376649 / (1 + 0.04)
        @test duration(Yields.Constant(0.04), -1 .* cfs, times) ≈ 1.777570320376649 / (1 + 0.04)
        @test duration(Yields.Forward([0.04,0.04]), cfs, times) ≈ 1.777570320376649 / (1 + 0.04)
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

        # the same, but with a functional argument
        value(i) = present_value(i, cfs, times)
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
        @test duration(0.03, cfs, times) ≈ 2.780101622010806
        @test convexity(0.03, cfs, times) ≈ 10.62580548268594
    end

end


include("run_doctests.jl")