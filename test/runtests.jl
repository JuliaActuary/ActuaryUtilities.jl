using ActuaryUtilities

using Dates
using Test
using DayCounts

@testset "Temporal functions" begin
    @testset "years_between" begin
        @test years_between(Date(2018,9,30),Date(2018,9,30)) == 0
        @test years_between(Date(2018,9,30),Date(2018,9,30),true) == 0
        @test years_between(Date(2018,9,30),Date(2019,9,30),false) == 0
        @test years_between(Date(2018,9,30),Date(2019,9,30),true) == 1
        @test years_between(Date(2018,9,30),Date(2019,10,1),true) == 1
        @test years_between(Date(2018,9,30),Date(2019,10,1),false) == 1
    end

    @testset "duration tests" begin
        @test duration(Date(2018,9,30),Date(2019,9,30)) == 2
        @test duration(Date(2018,9,30),Date(2018,9,30)) == 1
        @test duration(Date(2018,9,30),Date(2018,10,1)) == 1
        @test duration(Date(2018,9,30),Date(2019,10,1)) == 2
        @test duration(Date(2018,9,30),Date(2018,6,30)) == 0
        @test duration(Date(2018,9,30),Date(2017,6,30)) == -1
        @test duration(Date(2018,10,15),Date(2019,9,30)) == 1
        @test duration(Date(2018,10,15),Date(2019,10,30)) == 2
        @test duration(Date(2018,10,15),Date(2019,10,15)) == 2
        @test duration(Date(2018,10,15),Date(2019,10,14)) == 1
    end
end

@testset "financial calcs" begin

    @testset "pv" begin

    v = [100, 100]
        @test pv(0.05,v) ≈ v[1] / 1.05 + v[2] / 1.05^2
    end

    @testset "pv with timepoints" begin

    v = [100, 100]
        @test pv(0.05,v,[1,2]) ≈ v[1] / 1.05 + v[2] / 1.05^2
    end

    @testset "irr" begin

        v = [-70000,12000,15000,18000,21000,26000]
        
        # per Excel (example comes from Excel help text)
        @test isapprox(irr(v[1:2]), -0.8285714285714,atol = 0.001)
        @test isapprox(irr(v[1:3]), -0.4435069413346,atol = 0.001)
        @test isapprox(irr(v[1:4]), -0.1821374641455,atol = 0.001)
        @test isapprox(irr(v[1:5]), -0.0212448482734,atol = 0.001)
        @test isapprox(irr(v[1:6]),  0.0866309480365,atol = 0.001)

        # much more challenging to solve b/c of the overflow below zero
        cfs = [t % 10 == 0 ? -10 : 1.5 for t in 0:99]

        @test isapprox(irr(cfs), 0.06463163963925866,atol=0.001)

        # test the unsolvable

        @test isnothing(irr([100,100]))

    end

    @testset "xirr with float times" begin

    
        @test isapprox(irr([-100,100],[0,1]), 0.0, atol =0.001)
        @test isapprox(irr([-100,110],[0,1]), 0.1, atol =0.001)

    end

    @testset "xirr with real dates" begin

    v = [-70000,12000,15000,18000,21000,26000]
    dates = Date(2019,12,31):Year(1):Date(2024,12,31)
    times = yearfrac.(dates[1],dates,Thirty360)
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
        @test breakeven([-10,1,2,3,4,8],0.10) == 5
        @test breakeven([-10,15,2,3,4,8],0.10) == 1
        @test breakeven([-10,15,2,3,4,8],0.10) == 1
        @test isnothing(breakeven([-10,-15,2,3,4,8],0.10))
    end

    @testset "timepoints" begin
        times = [t for t in 0:5]
        @test breakeven([-10,1,2,3,4,8],times,0.10) == 5
        @test breakeven([-10,15,2,3,4,8],times,0.10) == 1
        @test breakeven([-10,15,2,3,4,8],times,0.10) == 1
        @test isnothing(breakeven([-10,-15,2,3,4,8],times,0.10))
    end
end


include("run_doctests.jl")