# Worked examples also appear in docs/src/convexity.md.
@testset "Convexity shock coordinates: published formulas and AD" begin
    cfs = [5.0, 5.0, 105.0]
    times = [1.0, 2.0, 3.0]
    y = 0.04
    curve = FM.Yield.Constant(FC.Periodic(y, 1))
    discounted = cfs ./ (1 + y) .^ times
    value = sum(discounted)
    weights = discounted ./ value
    valuation(c) = FC.pv(c, cfs, times)

    # Nawalkha, Soto & Beliaeva (2005), Interest Rate Risk Modeling,
    # ch. 1, pp. 5–6, eq. (1.2): C = sum(w_t * t^2).
    # https://catalogimages.wiley.com/images/db/pdf/0471427241.excerpt.pdf
    continuous = sum(weights .* times .^ 2)
    @test value ≈ 102.77509103322713
    @test continuous ≈ 8.40087191197841
    # This independent price function uses no library bump or sensitivity helper.
    # ForwardDiff differentiates the exponential; no t^2 is supplied to AD.
    shifted_price(s) = sum(cfs .* exp.(-(log1p(y) + s) .* times))
    ad_reference = ForwardDiff.derivative(s -> ForwardDiff.derivative(shifted_price, s), 0.0) / value
    @test ad_reference ≈ continuous
    @test convexity(curve, cfs, times) ≈ ad_reference # constant-curve analytic path
    @test convexity(curve, valuation) ≈ ad_reference # scalar nested AD
    @test convexity(curve, times, cfs, times) ≈ ad_reference
    @test sum(convexity(KeyRates(times), curve, cfs, times)) ≈ ad_reference
    @test sum(convexity(KeyRates(times), valuation, curve)) ≈ ad_reference

    # Coleman (2011), A Guide to Duration, DV01, and Yield Curve Risk
    # Transformations, pp. 3–4: the rate coordinate/compounding matters.
    # https://closemountain.com/papers/risktransform1_brief.pdf
    equivalent = FM.Yield.Constant(FC.Continuous(log1p(y)))
    @test FC.pv(equivalent, cfs, times) ≈ value
    @test convexity(equivalent, cfs, times) ≈ continuous
    @test convexity(FC.Continuous(log1p(y)), cfs, times) ≈ continuous

    # Clarke, de Silva & Thorley (2013), Fundamentals of Futures and Options,
    # appendix p. 127, eq. (A.15) and the modification immediately below it.
    # https://www.cfainstitute.org/sites/default/files/-/media/documents/book/rf-publication/2013/rf-v2013-n3-1-pdf.pdf
    # The old constant-curve statistic omitted the annual-yield divisor.
    old_statistic = sum(weights .* times .* (times .+ 1))
    annual = old_statistic / (1 + y)^2
    @test old_statistic ≈ 11.262334786519965
    @test annual ≈ 10.412661599962984
    annual_price(r) = sum(cfs ./ (1 + r) .^ times)
    @test ForwardDiff.derivative(r -> ForwardDiff.derivative(annual_price, r), y) / value ≈ annual
    for yield in (y, FC.Periodic(y, 1))
        @test convexity(yield, cfs, times) ≈ annual
        @test convexity(yield, valuation) ≈ annual
    end

    # t is time in years, including fractional years, not the cashflow index.
    @test convexity(curve, [100.0], [2.5]) ≈ 2.5^2
    @test convexity(curve, c -> 100FC.discount(c, 2.5)) ≈ 2.5^2
end

@testset "Parallel convexity includes mixed key-rate derivatives" begin
    cfs = [5.0, 5.0, 105.0]
    times = [1.0, 2.0, 3.0]
    curve = FM.Yield.Constant(FC.Periodic(0.04, 1))
    kr = KeyRates([1.0, 3.0]) # the middle cashflow lies between the two knots
    valuation(c) = FC.pv(c, cfs, times)
    value = valuation(curve)

    # Ho (1992), Key Rate Durations: Measures of Interest Rate Risks:
    # https://doi.org/10.3905/jfi.1992.408049
    # Linear interpolation gives the 2-year zero shock (s[1] + s[2])/2.
    # Reitano (1991), Multivariate Duration Analysis, section 3(c),
    # definitions 3.3–3.5 and eq. (3.28): K_jk = P_jk/P.
    # https://www.soa.org/globalassets/assets/library/monographs/50th-anniversary/investment-section/1999/january/m-as99-2-05.pdf
    r = log1p(0.04)
    price(s) = 5exp(-r - s[1]) + 5exp(-2r - s[1] - s[2]) + 105exp(-3r - 3s[2])
    expected = ForwardDiff.hessian(price, zeros(2)) / value
    analytic = convexity(kr, curve, cfs, times)
    automatic = convexity(kr, valuation, curve)
    @test analytic ≈ expected
    @test automatic ≈ expected
    @test expected[1, 2] ≈ 5 / 1.04^2 / value
    @test expected[2, 1] ≈ expected[1, 2]
    @test sum(analytic) ≈ 8.40087191197841
    @test sum(automatic) ≈ convexity(curve, valuation)
    # For a parallel shock, s = x*ones(2): C_parallel = ones' * K * ones.
    # Summing only diagonal entries loses the two mixed-derivative terms.
    diagonal_only = analytic[1, 1] + analytic[2, 2]
    @test diagonal_only < sum(analytic)
    @test sum(analytic) - diagonal_only ≈ 2expected[1, 2]
end
