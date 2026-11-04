# M. Lorig, An Introduction to Financial Derivatives: Open Lecture Notes (2026),
# arXiv:2611.01355v1, Exercise 2.1: a forward on a stock paying known dividends.
# Replication formulas (A.30)-(A.34) and their simplified value in (A.37).
# https://arxiv.org/html/2611.01355v1#Ch2.Thmexercise1
# https://arxiv.org/html/2611.01355v1#A1.S2
@testset "Lorig: remaining dividends in a forward replication" begin
    # Chosen parameters: unequal dividends and irregular payment times make
    # inclusion of an already-paid dividend or a wrong valuation origin visible.
    r, s0, maturity = 0.05, 100.0, 2.0
    dividends = [1.5, 2.0, 3.0]
    times = [0.25, 0.9, 1.6]
    origins = [0.0; times[1:(end - 1)]]
    delivery = s0 * exp(r * maturity) -
        sum(dividends .* exp.(r .* (maturity .- times)))

    for interest in (FC.Continuous(r), FM.Yield.Constant(FC.Continuous(r)))
        remaining = present_values(interest, dividends, times)
        wrapped = present_values(interest, FC.Cashflow.(dividends, times))
        for k in eachindex(dividends)
            t = origins[k]
            expected = sum(dividends[i] * exp(-r * (times[i] - t)) for i in k:lastindex(dividends))
            @test remaining[k] ≈ expected rtol = 1.0e-13
            @test wrapped[k] ≈ expected rtol = 1.0e-13

            # Choose the later spot independently; (A.31) gives the new fair
            # forward price and (A.34) discounts its difference from delivery.
            spot = k == 1 ? s0 : 100.0 + 2k
            forward = spot * exp(r * (maturity - t)) -
                sum(dividends[i] * exp(r * (maturity - times[i])) for i in k:lastindex(dividends))
            value = spot - remaining[k] - delivery * exp(-r * (maturity - t))
            @test value ≈ (forward - delivery) * exp(-r * (maturity - t)) atol = 1.0e-12
            # At each origin the preceding payments have already been made.
            # Remaining dividends cancel against those in the original delivery
            # price, leaving only the accumulated dividends already received.
            paid = sum((dividends[i] * exp(r * (t - times[i])) for i in 1:(k - 1)); init = 0.0)
            @test value ≈ spot - s0 * exp(r * t) + paid atol = 1.0e-12
            if k == 1
                @test abs(value) <= 1.0e-12
            end
        end
    end
end
