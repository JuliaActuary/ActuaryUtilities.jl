## Market-input sensitivities
# Differentiate a valuation with respect to named vectors of market inputs, such
# as quoted rates or spreads, that the valuation turns into curves itself.

"""
    sensitivities(valuation, inputs::NamedTuple) -> (; value, duration, dv01, key_rate, key_rate_dv01)

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
vector. The valuation receives views of the same shapes. For each named input the
result contains:

- `duration`: `-∂V/∂s / V` for a parallel shift `s` added to every element
- `dv01`: `-∂V/∂s / 10000`, the signed value loss for a 0.0001 increase in every element
  (negative when the value rises)
- `key_rate`: the per-element vector `-∂V/∂xᵢ / V`
- `key_rate_dv01`: the per-element vector `-∂V/∂xᵢ / 10000`

Dollar measures assume the inputs are rates in decimal units, so 0.0001 is one basis
point, and they keep the position's sign. They remain defined at zero value, where
the normalized measures are not. The bump coordinate is the one in which the inputs
are expressed: an annual quoted rate is bumped as an annual rate.

`duration` and `dv01` sum the per-element derivatives. They equal the derivative of a
parallel shift only where the valuation is differentiable, which excludes interpolation
kinks such as a flat `Spline.MonotoneConvex` interval (see FinanceModels'
[Sensitivities Through Calibration](https://docs.juliaactuary.org/FinanceModels/stable/calibration_sensitivities/)).

The valuation must be differentiable with ForwardDiff. It runs once for the value and
once more for the derivatives, which takes one forward pass per 64 input elements. A
valuation that fits a curve therefore fits it twice (once more per additional 64
elements), not once per input.

See also [`KeyRates`](@ref) for sensitivities to bumps of a given curve.
"""
function sensitivities(valuation::F, inputs::NamedTuple{roles, <:Tuple{AbstractVector{<:Real}, Vararg{AbstractVector{<:Real}}}}) where {F, roles}
    lengths = map(length, values(inputs))
    stops = cumsum(lengths)
    ranges = ntuple(i -> (stops[i] - lengths[i] + 1):stops[i], length(roles))
    x0 = reduce(vcat, map(v -> float.(v), values(inputs)))
    rebuild(x) = NamedTuple{roles}(map(r -> view(x, r), ranges))
    f(x) = valuation(rebuild(x))
    value = f(x0)
    # The valuation can return BigFloat or an outer AD Dual; size the buffer from it.
    gradient = zeros(typeof(value), length(x0))
    config = ForwardDiff.GradientConfig(f, x0, ForwardDiff.Chunk(x0, 64))
    ForwardDiff.gradient!(gradient, f, x0, config)
    grads = NamedTuple{roles}(map(r -> gradient[r], ranges))
    return (;
        value,
        duration = map(g -> -sum(g) / value, grads),
        dv01 = map(g -> -sum(g) / 10_000, grads),
        key_rate = map(g -> -g ./ value, grads),
        key_rate_dv01 = map(g -> -g ./ 10_000, grads),
    )
end
