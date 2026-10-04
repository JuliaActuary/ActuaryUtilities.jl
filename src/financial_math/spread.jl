## Constant spread between two curves

"""
    spread(curve1, curve2, cashflows, times = eachindex(cashflows); tol = 1e-12, maxiter = 100)

Find the constant spread to add to `curve1` so the cashflows have the same present
value as under `curve2`.

The spread is found via a damped Newton iteration on the pricing residual. A candidate is accepted
once the undamped Newton step is smaller than `tol` in rate units (not currency) **and** the
candidate reprices the cashflows: its pricing residual must be at most `sqrt(eps)` times the sum
of the gross discounted values `Σ|cfᵢ|⋅dfᵢ` under the two curves. Both tests are unchanged by
scaling the cashflows, and the gross scale is nonzero for zero-price and mixed-sign streams. A
small step alone is not enough: near a combined annual rate of -100% the price derivative is so
large that the step is tiny far from the root. Steps never go more than halfway to the edge of
the spread's domain (the spread must exceed -1, and so must the combined rate per period when it
is added to a periodic base). An `ErrorException` is thrown if the valuation or its derivative is
`NaN`, or if no candidate is accepted within `maxiter` iterations.

!!! note
    For mixed-sign cashflows the pricing residual can have more than one exact root (e.g. a duration-neutral asset/liability pair); the root reached from a starting spread of zero is returned.

# Examples

```julia-repl
julia> spread(0.04, 0.05, fill(10.0, 10))
Periodic(0.009999999999999964, 1)
```
"""
spread(curve1, curve2, cashflows, times...; tol = 1.0e-12, maxiter = 100) =
    _spread(curve1, curve2, _cashflow_inputs(cashflows, times...)...; tol, maxiter)
function _spread(curve1, curve2, cashflows, times; tol, maxiter)
    times = FinanceCore.timepoint.(cashflows, times)
    cashflows = FinanceCore.amount.(cashflows)
    pv2 = FinanceCore.pv(curve2, cashflows, times)
    gross2 = sum(abs(cf) * FinanceCore.discount(curve2, t) for (cf, t) in zip(cashflows, times))

    combined(s) = curve1 + FinanceCore.Periodic(s, 1)
    f(s) = FinanceCore.pv(combined(s), cashflows, times) - pv2
    # Dampen Newton steps: mixed-sign cashflows can have nearly zero price derivatives, and
    # near the domain edge a full step would leave it.
    max_step = 0.25
    floor = _spread_floor(curve1)
    converged, s, newton = _newton(f, 0.0, maxiter) do s, newton, dfs
        isnan(_primal(newton)) && throw(
            ErrorException("spread: the valuation or its derivative is NaN at spread $(_primal(s))")
        )
        # Convergence is decided on primal values; the returned candidate keeps any partials.
        if isfinite(_primal(newton)) && abs(_primal(newton)) < tol
            # The dual evaluation sums in another order, so the candidate's step takes the
            # residual from `f` itself, which keeps the root's rounding error at its previous size.
            c = s - f(s) / dfs
            residual, gross1 = _primal_residual_and_gross(combined(c), cashflows, times, pv2)
            scale = gross1 + _primal(gross2)
            isfinite(residual) && isfinite(scale) &&
                abs(residual) <= sqrt(eps(float(typeof(residual)))) * scale &&
                return true, c
        end
        # the damping choice is discrete, so it uses the primal step; an undamped step keeps its partials
        p = _primal(newton)
        step = !isfinite(p) || abs(p) > max_step ? copysign(max_step, p) : newton
        return false, max(s - step, (s + floor) / 2)
    end
    converged || throw(ErrorException("spread did not converge in $maxiter iterations (last Newton step = $newton)"))
    return FinanceCore.Periodic(s, 1)
end

# The Newton iteration `spread` and `zspread` share; each keeps its own coordinate, safeguards and
# acceptance rule in `advance`. From `s`, every iteration evaluates `f` and `f′` together, and an
# exact root ends the solve. Otherwise `advance(s, newton, f′)`, given the Newton step `f / f′`,
# returns `(true, root)` to accept a root or `(false, next)` to continue from `next`. Returns
# `(converged, s, newton)`: the root, or the last iterate and Newton step after `maxiter` iterations.
function _newton(advance::A, f::F, s, maxiter) where {A, F}
    newton = oftype(s, NaN)
    for _ in 1:maxiter
        fs, dfs = _value_and_derivative(f, s)
        iszero(fs) && return true, s, newton
        newton = fs / dfs
        done, s = advance(s, newton, dfs)
        done && return true, s, newton
    end
    return false, s, newton
end

# The lowest spread `s` for which `curve1 + Periodic(s, 1)` is a valid rate. `Periodic(s, 1)` needs
# s > -1. A periodic base `r` compounded `n` times adds the spread nominally in its own convention,
# so the combined rate `r + n((1 + s)^(1/n) - 1)` must also exceed -n. A bare number is an annual rate.
_spread_floor(base::Real) = _periodic_spread_floor(_primal(base), 1)
_spread_floor(base::FinanceCore.Rate{<:Any, FinanceCore.Periodic}) =
    _periodic_spread_floor(_primal(FinanceCore.rate(base)), base.compounding.frequency)
_spread_floor(base) = -1.0
_periodic_spread_floor(r, n) = max(-1.0, float(max(-r / n, zero(r)))^n - 1)

# The pricing residual against `target` and the gross discounted value Σ|cfᵢ|⋅dfᵢ of the cashflows
# under `curve`, as primal values, from one discount per cashflow.
function _primal_residual_and_gross(curve, cashflows, times, target)
    value = gross = zero(_primal(target))
    for (cf, t) in zip(cashflows, times)
        a, d = _primal(cf), _primal(FinanceCore.discount(curve, t))
        value += a * d
        gross += abs(a) * d
    end
    return value - _primal(target), gross
end
