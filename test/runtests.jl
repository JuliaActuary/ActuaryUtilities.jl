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

end