module FinancialMath

import ..FinanceCore
import ..FinanceCore: irr, internal_rate_of_return, pv, present_value
import ..FinanceModels
import ..ForwardDiff
import ..ActuaryUtilities: duration
import Random

export irr, internal_rate_of_return, spread,
    pv, present_value, price, present_values,
    breakeven, moic,
    Macaulay, Modified, DV01, IR01, CS01, Effective, Spread, KeyRates, KeyRatePar, KeyRateZero, KeyRate, duration, convexity,
    sensitivities, dv01, zspread, locked_floater, reproject

include("financial_math/scalar_measures.jl")
include("financial_math/key_rate_sensitivities.jl")
include("financial_math/contract_sensitivities.jl")
include("financial_math/scenario_sensitivities.jl")

end
