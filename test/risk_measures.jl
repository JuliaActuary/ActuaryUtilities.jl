@testset "Risk Measures" begin
    sample = [0:100...]

    @testset "VaR" begin
        @test VaR(sample, 0.9) ≈ 90
        @test VaR(sample, 0.9, rev=true) ≈ 10
        @test VaR(sample, 0.1) ≈ VaR(sample, 1 - 0.9)
        @test VaR(sample, 0.1) == ValueAtRisk(sample, 0.1)
    end

    @testset "CTE" begin
        @test CTE(sample, 0.9) ≈ sum(90:100) / length(90:100)
        @test CTE(sample, 0.1) ≈ sum(10:100) / length(10:100)
        @test CTE(sample, 0.15) >= CTE(sample, 0.1) # monotonicity  
        @test CTE(sample, 0.9, rev=true) ≈ sum(0:10) / length(0:10)
        @test CTE(sample, 0.9) == ConditionalTailExpectation(sample, 0.9)
    end


    @testset "duplicated values" begin
        sample = zeros(100)
        sample[end] = 100

        @test CTE(sample, 0) ≈ 100 / 100
        @test CTE(sample, 0.5) ≈ 100 / 50

        @test VaR(sample, 0) ≈ 0
        @test VaR(sample, 0.5) ≈ 0
        @test VaR(sample, 0.99) ≈ 0

    end

end

