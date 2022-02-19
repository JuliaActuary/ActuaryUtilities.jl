var documenterSearchIndex = {"docs":
[{"location":"#ActuaryUtilities.jl","page":"Home","title":"ActuaryUtilities.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"DocTestSetup = quote\n    using ActuaryUtilities\n    using Dates\nend","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [ActuaryUtilities]","category":"page"},{"location":"#ActuaryUtilities.CTE-Tuple{Any,Any}","page":"Home","title":"ActuaryUtilities.CTE","text":"CTE(v::AbstractArray,p::Real;rev::Bool=false)\n\nThe average of the values ≥ the pth percentile of the vector v is the Conditiona Tail Expectation. Assumes more positive values are higher risk measures, so a higher p will return a more positive number, but this can be reversed if rev is true.\n\nMay also be called with ConditionalTailExpectation(...).\n\nAlso known as Tail Value at Risk (TVaR), or Tail Conditional Expectation (TCE)\n\n\n\n\n\n","category":"method"},{"location":"#ActuaryUtilities.ConditionalTailExpectation","page":"Home","title":"ActuaryUtilities.ConditionalTailExpectation","text":"CTE\n\n\n\n\n\n","category":"function"},{"location":"#ActuaryUtilities.VaR-Tuple{Any,Any}","page":"Home","title":"ActuaryUtilities.VaR","text":"VaR(v::AbstractArray,p::Real;rev::Bool=false)\n\nThe pth quantile of the vector v is the Value at Risk. Assumes more positive values are higher risk measures, so a higher p will return a more positive number, but this can be reversed if rev is true.\n\nAlso can be called with ValueAtRisk(...).\n\n\n\n\n\n","category":"method"},{"location":"#ActuaryUtilities.ValueAtRisk","page":"Home","title":"ActuaryUtilities.ValueAtRisk","text":"VaR\n\n\n\n\n\n","category":"function"},{"location":"#ActuaryUtilities.accum_offset-Tuple{Any}","page":"Home","title":"ActuaryUtilities.accum_offset","text":"accum_offset(x; op=*, init=1.0)\n\nA shortcut for the common operation wherein a vector is scanned with an operation, but has an initial value and the resulting array is offset from the traditional accumulate. \n\nThis is a common pattern when calculating things like survivorship given a mortality vector and you want the first value of the resulting vector to be 1.0, and the second value to be 1.0 * x[1], etc.\n\nTwo keyword arguments:\n\nop is the binary (two argument) operator you want to use, such as * or +\ninit is the initial value in the returned array\n\nExamples\n\njulia> accum_offset([0.9, 0.8, 0.7])\n3-element Array{Float64,1}:\n 1.0\n 0.9\n 0.7200000000000001\n\njulia> accum_offset(1:5) # the product of elements 1:n, with the default `1` as the first value\n5-element Array{Int64,1}:\n  1\n  1\n  2\n  6\n 24\n\njulia> accum_offset(1:5,op=+)\n5-element Array{Int64,1}:\n  1\n  2\n  4\n  7\n 11\n\n\n\n\n\n\n","category":"method"},{"location":"#ActuaryUtilities.breakeven-Union{Tuple{T}, Tuple{T,Array{T,1} where T,Array{T,1} where T}} where T<:Yields.AbstractYield","page":"Home","title":"ActuaryUtilities.breakeven","text":"breakeven(yield, cashflows::Vector)\nbreakeven(yield, cashflows::Vector,times::Vector)\n\nCalculate the time when the accumulated cashflows breakeven given the yield.\n\nAssumptions:\n\ncashflows occur at the end of the period\ncashflows evenly spaced with the first one occuring at time zero if times not given\n\nReturns nothing if cashflow stream never breaks even.\n\njulia> breakeven(0.10, [-10,1,2,3,4,8])\n5\n\njulia> breakeven(0.10, [-10,15,2,3,4,8])\n1\n\njulia> breakeven(0.10, [-10,-15,2,3,4,8]) # returns the `nothing` value\n\n\n\n\n\n\n\n","category":"method"},{"location":"#ActuaryUtilities.convexity-Tuple{Any,Any,Any}","page":"Home","title":"ActuaryUtilities.convexity","text":"convexity(yield,cfs,times)\nconvexity(yield,valuation_function)\n\nCalculates the convexity.     - yield should be a fixed effective yield (e.g. 0.05).     - times may be omitted and it will assume cfs are evenly spaced beginning at the end of the first period.\n\nExamples\n\nUsing vectors of cashflows and times\n\njulia> times = 1:5\njulia> cfs = [0,0,0,0,100]\njulia> duration(0.03,cfs,times)\n4.854368932038834\njulia> duration(Macaulay(),0.03,cfs,times)\n5.0\njulia> duration(Modified(),0.03,cfs,times)\n4.854368932038835\njulia> convexity(0.03,cfs,times)\n28.277877274012614\n\n\nUsing any given value function: \n\njulia> lump_sum_value(amount,years,i) = amount / (1 + i ) ^ years\njulia> my_lump_sum_value(i) = lump_sum_value(100,5,i)\njulia> duration(0.03,my_lump_sum_value)\n4.854368932038835\njulia> convexity(0.03,my_lump_sum_value)\n28.277877274012617\n\n\n\n\n\n\n","category":"method"},{"location":"#ActuaryUtilities.duration-Tuple{Dates.Date,Dates.Date}","page":"Home","title":"ActuaryUtilities.duration","text":"duration(d1::Date, d2::Date)\n\nCompute the duration given two dates, which is the number of years since the first date. The interval [0,1) is defined as having  duration 1. Can return negative durations if second argument is before the first.\n\njulia> issue_date  = Date(2018,9,30);\n\njulia> duration(issue_date , Date(2019,9,30) ) \n2\njulia> duration(issue_date , issue_date) \n1\njulia> duration(issue_date , Date(2018,10,1) ) \n1\njulia> duration(issue_date , Date(2019,10,1) ) \n2\njulia> duration(issue_date , Date(2018,6,30) ) \n0\njulia> duration(Date(2018,9,30),Date(2017,6,30)) \n-1\n\n\n\n\n\n","category":"method"},{"location":"#ActuaryUtilities.duration-Tuple{Macaulay,Any,Any,Any}","page":"Home","title":"ActuaryUtilities.duration","text":"duration(Macaulay(),interest_rate,cfs,times)\nduration(Modified(),interest_rate,cfs,times)\nduration(DV01(),interest_rate,cfs,times)\nduration(interest_rate,cfs,times)             # Modified Duration\nduration(interest_rate,valuation_function)    # Modified Duration\n\nCalculates the Macaulay, Modified, or DV01 duration. times may be ommitted and the valuation will assume evenly spaced cashflows starting at the end of the first period.\n\ninterest_rate should be a fixed effective yield (e.g. 0.05).\n\nWhen not given Modified() or Macaulay() as an argument, will default to Modified().\n\nExamples\n\nUsing vectors of cashflows and times\n\njulia> times = 1:5\njulia> cfs = [0,0,0,0,100]\njulia> duration(0.03,cfs,times)\n4.854368932038834\njulia> duration(Macaulay(),0.03,cfs,times)\n5.0\njulia> duration(Modified(),0.03,cfs,times)\n4.854368932038835\njulia> convexity(0.03,cfs,times)\n28.277877274012614\n\n\nUsing any given value function: \n\njulia> lump_sum_value(amount,years,i) = amount / (1 + i ) ^ years\njulia> my_lump_sum_value(i) = lump_sum_value(100,5,i)\njulia> duration(0.03,my_lump_sum_value)\n4.854368932038835\njulia> convexity(0.03,my_lump_sum_value)\n28.277877274012617\n\n\n\n\n\n\n","category":"method"},{"location":"#ActuaryUtilities.eurocall-Tuple{}","page":"Home","title":"ActuaryUtilities.eurocall","text":"eurocall(;S=1.,K=1.,τ=1,r,σ,q=0.)\n\nCalculate the Black-Scholes implied option price for a european call, where:\n\nS is the current asset price\nK is the strike or exercise price\nτ is the time remaining to maturity (can be typed with \\tau[tab])\nr is the continuously compounded risk free rate\nσ is the (implied) volatility (can be typed with \\sigma[tab])\nq is the continuously paid dividend rate\n\nRates should be input as rates (not percentages), e.g.: 0.05 instead of 5 for a rate of five percent.\n\n!!! Experimental: this function is well-tested, but the derivatives functionality (API) may change in a future version of ActuaryUtilities.\n\nExtended Help\n\nThis is the same as the formulation presented in the dividend extension of the BS model in Wikipedia.\n\nOther general comments:\n\nSwap/OIS curves are generally better sources for r than government debt (e.g. US Treasury) due to the collateralized nature of swap instruments.\n(Implied) volatility is characterized by a curve that is a function of the strike price (among other things), so take care when using \nYields.jl can assist with converting rates to continuously compounded if you need to perform conversions.\n\n\n\n\n\n","category":"method"},{"location":"#ActuaryUtilities.europut-Tuple{}","page":"Home","title":"ActuaryUtilities.europut","text":"europut(;S=1.,K=1.,τ=1,r,σ,q=0.)\n\nCalculate the Black-Scholes implied option price for a european call, where:\n\nS is the current asset price\nK is the strike or exercise price\nτ is the time remaining to maturity (can be typed with \\tau[tab])\nr is the continuously compounded risk free rate\nσ is the (implied) volatility (can be typed with \\sigma[tab])\nq is the continuously paid dividend rate\n\nRates should be input as rates (not percentages), e.g.: 0.05 instead of 5 for a rate of five percent.\n\n!!! Experimental: this function is well-tested, but the derivatives functionality (API) may change in a future version of ActuaryUtilities.\n\nExtended Help\n\nThis is the same as the formulation presented in the dividend extension of the BS model in Wikipedia.\n\nOther general comments:\n\nSwap/OIS curves are generally better sources for r than government debt (e.g. US Treasury) due to the collateralized nature of swap instruments.\n(Implied) volatility is characterized by a curve that is a function of the strike price (among other things), so take care when using \nYields.jl can assist with converting rates to continuously compounded if you need to perform conversions.\n\n\n\n\n\n","category":"method"},{"location":"#ActuaryUtilities.internal_rate_of_return-Tuple{Any}","page":"Home","title":"ActuaryUtilities.internal_rate_of_return","text":"internal_rate_of_return(cashflows::vector)::Yields.Rate\ninternal_rate_of_return(cashflows::Vector, timepoints::Vector)::Yields.Rate\n\nCalculate the internalrateof_return with given timepoints. If no timepoints given, will assume that a series of equally spaced cashflows, assuming the first cashflow occurring at time zero and subsequent elements at time 1, 2, 3, ..., n. \n\nReturns a Yields.Rate type with periodic compounding once per period (e.g. annual effective if the timepoints given represent years). Get the scalar rate by calling Yields.rate() on the result.\n\nExample\n\njulia> internal_rate_of_return([-100,110],[0,1]) # e.g. cashflows at time 0 and 1\n0.10000000001652906\njulia> internal_rate_of_return([-100,110]) # implied the same as above\n0.10000000001652906\n\nSolver notes\n\nWill try to return a root within the range [-2,2]. If the fast solver does not find one matching this condition, then a more robust search will be performed over the [.99,2] range.\n\nThe solution returned will be in the range [-2,2], but may not be the one nearest zero. For a slightly slower, but more robust version, call ActuaryUtilities.irr_robust(cashflows,timepoints) directly.\n\n\n\n\n\n","category":"method"},{"location":"#ActuaryUtilities.irr","page":"Home","title":"ActuaryUtilities.irr","text":"irr(cashflows::vector)\nirr(cashflows::Vector, timepoints::Vector)\n\nAn alias for `internal_rate_of_return`.\n\n\n\n\n\n","category":"function"},{"location":"#ActuaryUtilities.moic-Tuple{T} where T<:AbstractArray","page":"Home","title":"ActuaryUtilities.moic","text":"moic(cashflows<:AbstractArray)\n\nThe multiple on invested capital (\"moic\") is the un-discounted sum of distributions divided by the sum of the contributions. The function assumes that negative numbers in the array represent contributions and positive numbers represent distributions.\n\nExamples\n\njulia> moic([-10,20,30])\n5.0\n\n\n\n\n\n","category":"method"},{"location":"#ActuaryUtilities.present_value-Union{Tuple{T}, Tuple{T,Any,Any}} where T<:Yields.AbstractYield","page":"Home","title":"ActuaryUtilities.present_value","text":"present_value(interest, cashflows::Vector, timepoints)\npresent_value(interest, cashflows::Vector)\n\nDiscount the cashflows vector at the given interest_interestrate,  with the cashflows occurring at the times specified in timepoints. If no timepoints given, assumes that cashflows happen at times 1,2,...,n.\n\nThe interest can be an InterestCurve, a single scalar, or a vector wrapped in an InterestCurve. \n\nExamples\n\njulia> present_value(0.1, [10,20],[0,1])\n28.18181818181818\njulia> present_value(Yields.Forward([0.1,0.2]), [10,20],[0,1])\n28.18181818181818 # same as above, because first cashflow is at time zero\n\nExample on how to use real dates using the DayCounts.jl package\n\n\nusing DayCounts \ndates = Date(2012,12,31):Year(1):Date(2013,12,31)\ntimes = map(d -> yearfrac(dates[1], d, DayCounts.Actual365Fixed()),dates) # [0.0,1.0]\npresent_value(0.1, [10,20],times)\n\n# output\n28.18181818181818\n\n\n\n\n\n\n","category":"method"},{"location":"#ActuaryUtilities.present_values-Tuple{Any,Any}","page":"Home","title":"ActuaryUtilities.present_values","text":"present_value(interest, cashflows::Vector, timepoints)\npresent_value(interest, cashflows::Vector)\n\nEfficiently calculate a vector representing the present value of the given cashflows at each period prior to the given timepoint.\n\nExamples\n\njulia> present_values(0.00, [1,1,1])\n[3,2,1]\n\njulia> present_values(Yields.Forward([0.1,0.2]), [10,20],[0,1])\n2-element Vector{Float64}:\n 28.18181818181818\n 18.18181818181818\n\n\n\n\n\n","category":"method"},{"location":"#ActuaryUtilities.price-Tuple{Any,Any}","page":"Home","title":"ActuaryUtilities.price","text":"price(...)\n\nThe absolute value of the present_value(...). \n\nExtended help\n\nUsing price can be helpful if the directionality of the value doesn't matter. For example, in the common usage, duration is more interested in the change in price than present value, so price is used there.\n\n\n\n\n\n","category":"method"},{"location":"#ActuaryUtilities.pv","page":"Home","title":"ActuaryUtilities.pv","text":"pv()\n\nAn alias for `present_value`.\n\n\n\n\n\n","category":"function"},{"location":"#ActuaryUtilities.years_between","page":"Home","title":"ActuaryUtilities.years_between","text":"Years_Between(d1::Date, d2::Date)\n\nCompute the number of integer years between two dates, with the  first date typically before the second. Will return negative number if first date is after the second. Use third argument to indicate if calendar  anniversary should count as a full year.\n\nExamples\n\njulia> d1 = Date(2018,09,30);\n\njulia> d2 = Date(2019,09,30);\n\njulia> d3 = Date(2019,10,01);\n\njulia> years_between(d1,d3) \n1\njulia> years_between(d1,d2,false) # same month/day but `false` overlap\n0 \njulia> years_between(d1,d2) # same month/day but `true` overlap\n1 \njulia> years_between(d1,d2) # using default `true` overlap\n1 \n\n\n\n\n\n","category":"function"}]
}
