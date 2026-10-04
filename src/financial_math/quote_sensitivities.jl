## Market-input sensitivities
# Differentiate a valuation with respect to named vectors of market inputs, such
# as quoted rates or spreads, that the valuation turns into curves itself.

const _NamedInputs = NamedTuple{<:Any, <:Tuple{AbstractVector{<:Real}, Vararg{AbstractVector{<:Real}}}}

"""
    sensitivities(valuation, [order,] inputs::NamedTuple) -> NamedTuple

Differentiate `valuation(inputs)` with respect to every element of each named input
vector. Use this form when the valuation builds its own curves from market data,
for example by fitting quotes or layering spreads:

```julia
sensitivities((; sofr = rates, credit = spreads)) do m
    curve = fit(Spline.Linear(), OISYield.(m.sofr, tenors), Fit.Bootstrap())
    present_value(curve + credit_curve(m.credit), liability)
end
```

Each input must be an `AbstractVector{<:Real}`; wrap a scalar input in a one-element
vector. The valuation receives vectors of the same shapes. As for curves, `order` is
[`FirstOrder()`](@ref) (the default) or [`SecondOrder()`](@ref), and the result is
`(; value, duration, dv01)`, with `convexity` at second order. Each derivative field is keyed
by input name, with one entry per element:

- `duration.name`: the vector `-∂V/∂xᵢ / V`
- `dv01.name`: the vector `-∂V/∂xᵢ / 10000`, the signed value lost for a 0.0001 increase in
  each element (negative when the value rises)
- `convexity.name.other`: the matrix `∂²V/∂xᵢ∂yⱼ / V`

Dollar measures assume the inputs are rates in decimal units, so 0.0001 is one basis
point, and they keep the position's sign. They remain defined at zero value, where
the normalized measures are not. The bump coordinate is the one in which the inputs
are expressed: an annual quoted rate is bumped as an annual rate.

The sum of `duration.name` or `dv01.name` is the measure for a shift added to every element of
that input. It equals the derivative of a parallel shift only where the valuation is
differentiable, which excludes interpolation kinks such as a flat `Spline.MonotoneConvex`
interval (see FinanceModels'
[Sensitivities Through Calibration](https://docs.juliaactuary.org/FinanceModels/stable/calibration_sensitivities/)).

The valuation must be differentiable with ForwardDiff. At first order it runs once for the value
and once more for the derivatives, which takes one forward pass per 64 input elements. A
valuation that fits a curve therefore fits it twice (once more per additional 64 elements), not
once per input.

See also [`KeyRates`](@ref) for sensitivities to bumps of a given curve.
"""
sensitivities(valuation::F, inputs::_NamedInputs) where {F} = sensitivities(valuation, FirstOrder(), inputs)
sensitivities(valuation::F, order::_Order, inputs::_NamedInputs) where {F} =
    _sensitivities(_named_ad(valuation, inputs, order), order)
