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

# Declare the generic function here (with no methods) so both the FinancialMath
# and Utilities submodules can add methods to the same `duration` without one
# shadowing the other. `function duration end` adds no method; `duration()` with
# parens would define a callable zero-arg method that silently returns `nothing`.
function duration end

include("financial_math.jl")
include("risk_measures.jl")
include("utilities.jl")

@reexport using .FinancialMath
@reexport using .RiskMeasures
@reexport using .Utilities

include("precompile.jl")

end # module
