module FinancialMath

import ..FinanceCore
import ..FinanceCore: irr, internal_rate_of_return, pv, present_value
import ..FinanceModels
import ..ForwardDiff
import ConstructionBase
import DiffResults
import Random

export irr, internal_rate_of_return, spread,
    pv, present_value, present_values,
    breakeven, moic,
    Macaulay, Modified, DV01, IR01, CS01, Effective, Spread, KeyRates, duration, convexity,
    sensitivities, dv01, zspread, locked_floater, Scenarios

include("financial_math/cashflow_risk.jl")
include("financial_math/measures.jl")
include("financial_math/curve_shifts.jl")
include("financial_math/derivatives.jl")
include("financial_math/valuation.jl")
include("financial_math/scalar_measures.jl")
include("financial_math/spread.jl")
include("financial_math/key_rate_sensitivities.jl")
include("financial_math/contract_sensitivities.jl")
include("financial_math/quote_sensitivities.jl")
include("financial_math/scenario_sensitivities.jl")

end
