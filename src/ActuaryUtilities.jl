module ActuaryUtilities

using Reexport
import Dates
import FinanceCore
@reexport using FinanceCore: internal_rate_of_return, irr, present_value, pv
import ForwardDiff
import QuadGK
import FinanceModels
import StatsBase
using PrecompileTools
import Distributions

# need to define this here to extend it without conflict inside FinancialMath
function duration() end

include("utilities.jl")
include("financial_math.jl")
include("risk_measures.jl")



# include("precompile.jl")


@reexport using .FinancialMath
@reexport using .RiskMeasures
@reexport using .Utilities

end # module
