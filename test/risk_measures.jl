@testset "Risk Measures" begin
    sample = [0:100...]

    @testset "VaR" begin
        @test VaR(sample,.9) ≈ 90
        @test VaR(sample,.9,rev=true) ≈ 10
        @test VaR(sample,.1) ≈ VaR(sample,1-.9)
        @test VaR(sample,.1) == ValueAtRisk(sample,.1) 
    end
    
    @testset "CTE" begin
        @test CTE(sample,.9) ≈ sum(90:100) / length(90:100)
        @test CTE(sample,.1) ≈ sum(10:100) / length(10:100)
        @test CTE(sample,.15) >= CTE(sample,.1) # monotonicity  
        @test CTE(sample,.9,rev=true) ≈ sum(0:10) / length(0:10)
        @test CTE(sample,.9) == ConditionalTailExpectation(sample,.9)
    end

end

