using ActuaryUtilities

using Dates
using Test

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
        @test duration(Date(2018,9,30),Date(2019,9,30)) == 1
        @test duration(Date(2018,9,30),Date(2018,9,30)) == 0
        @test duration(Date(2018,9,30),Date(2018,10,1)) == 1
        @test duration(Date(2018,9,30),Date(2019,10,1)) == 2
        @test duration(Date(2018,9,30),Date(2018,6,30)) == 0
        @test duration(Date(2018,10,15),Date(2019,9,30)) == 1
        @test duration(Date(2018,10,15),Date(2019,10,30)) == 2
        @test duration(Date(2018,10,15),Date(2019,10,15)) == 1
        @test duration(Date(2018,10,15),Date(2019,10,14)) == 1
    end
end