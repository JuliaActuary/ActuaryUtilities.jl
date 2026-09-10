# A valid user curve need not return continuously compounded zero rates.
struct PeriodicZeroSensitivityCurve{T} <: FM.Yield.AbstractYieldModel
    rate::T
end
Base.zero(c::PeriodicZeroSensitivityCurve, t) = FC.Periodic(c.rate, 1)
FC.discount(c::PeriodicZeroSensitivityCurve, t) = FC.discount(zero(c, t), t)
