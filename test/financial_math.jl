@testset "Interest Curves" begin
    @testset "Stepwise" begin
        rates = [0.01,0.05,0.05,0.1]
        times = [1,2,3,4]
        ic = InterestCurve(rates, times, StepwiseInterp())

        @test interest_rate(ic, 0) == 0.01
        @test interest_rate(ic, 1) == 0.01
        @test interest_rate(ic, 3.001) == 0.1
        @test interest_rate(ic, 4) == 0.1

        @test interest_rate.(ic, [1,2,3,4]) == rates

        @test discount_rate(ic, 0) ≈ 1.0
        @test discount_rate(ic, 1) ≈ 1 / 1.01
        @test discount_rate(ic, 3) ≈ 1 / 1.01 * 1 / 1.05 * 1 / 1.05
        @test discount_rate(ic, 4) ≈ 1 / 1.01 * 1 / 1.05 * 1 / 1.05 * 1 / 1.1

        # test default constructor
        ic = InterestCurve(rates, times)

        @test interest_rate(ic, 0) == 0.01
    end

    @testset "Interpolations.jl methods" begin
        rates = [0.01,0.05,0.05,0.1]
        times = [1,2,3,4]
        ic = InterestCurve(rates, times, LinearInterp())
        
        @test interest_rate(ic, 0) == 0.01
        @test interest_rate(ic, 1) == 0.01
        @test interest_rate(ic, 1.5) ≈ 0.03
        @test interest_rate(ic, 3.5) ≈ 0.075
        @test interest_rate(ic, 4.5) == 0.1

    end

    @testset "scalar" begin

        @test interest_rate(0.05, 1) == 0.05
        @test interest_rate.(0.05, 1:2) == [0.05,0.05]

        @test discount_rate(0.05, 1) == 1 / 1.05
        @test discount_rate.(0.05, 1:2) == [1 / 1.05,1 / 1.05^2]
    
    end

    @testset "vector" begin

        @test interest_rate(InterestCurve([0.05,0.05]), 1) == 0.05
        @test interest_rate(InterestCurve([0.05,0.1]), 1) == 0.05
        @test interest_rate(InterestCurve([0.05,0.1]), 2) == 0.1
        @test interest_rate(InterestCurve([0.05,0.1]), 3) == 0.1
        @test interest_rate.(InterestCurve([0.05,0.1]), 1:2) == [0.05,0.1]

        @test discount_rate(InterestCurve([0.05,0.1]), 1) ≈ 1 / 1.05
        @test discount_rate(InterestCurve([0.05,0.1]), 2) ≈ 1 / 1.05 * 1 / 1.1


    end

end