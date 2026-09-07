## Key rate sensitivities via AD on zero-rate bumps (Yield.TenorShift)
#
# KRDs are computed by layering a triangular-hat zero-rate bump on top of the
# user's curve via `FinanceModels.Yield.TenorShift`, then taking ForwardDiff
# gradients/hessians w.r.t. the bump magnitudes. This works on any
# `AbstractYieldModel` — no curve-internal field is required, and there is no
# special dispatch for `ZeroRateCurve`. Callers always pass `tenors` (the KRD
# knot grid) explicitly; for a ZRC, the natural choice is `zrc.tenors`.

const AYM = FinanceModels.Yield.AbstractYieldModel

# Triangular hats with flat extrapolation outside the knot range.
# At a knot τᵢ the bump equals bᵢ; between τᵢ and τᵢ₊₁ it is linear.
function _hat_bump(tenors, bumps, t)
    t <= first(tenors) && return first(bumps)
    t >= last(tenors)  && return last(bumps)
    i = searchsortedlast(tenors, t)
    w = (t - tenors[i]) / (tenors[i + 1] - tenors[i])
    return (one(w) - w) * bumps[i] + w * bumps[i + 1]
end

# Layer a hat-function zero-rate bump over `curve` lazily.
_bumped(curve, tenors, bumps) = FinanceModels.Yield.TenorShift(
    curve,
    (z, t) -> z + FinanceCore.Continuous(_hat_bump(tenors, bumps, t)),
)

function _ad_derivatives(f::F, z, order) where {F}
    # The valuation can return BigFloat or an outer AD Dual even when bumps are
    # Float64. Establish its type before allocating the Hessian result buffers.
    value = f(z)
    g = zeros(typeof(value), length(z))
    if order == 1
        ForwardDiff.gradient!(g, f, z)
        return (; value, gradient = g)
    end
    result = DiffResults.DiffResult(value, g, similar(g, length(z), length(z)))
    result = ForwardDiff.hessian!(result, f, z)
    return (; value = DiffResults.value(result), gradient = g, hessian = DiffResults.hessian(result))
end

# One derivative engine over named curve roles. All adapters share the same
# validated grid, bump layout, derivative order, and named result structure.
function _ncurve_ad(valuation::F, curves::NamedTuple{roles}, tenors; order = 1) where {F, roles}
    order in (1, 2) || throw(ArgumentError("derivative order must be 1 or 2"))
    grid = KeyRates(tenors).tenors
    isempty(curves) && throw(ArgumentError("at least one curve role is required"))
    all(c -> c isa AYM, curves) || throw(ArgumentError("every curve role must be an AbstractYieldModel"))
    n, k = length(grid), length(curves)
    indices(i) = ((i - 1) * n + 1):(i * n)
    slice(b, i) = length(curves) == 1 ? b : view(b, indices(i))
    # Derive the count from the typed tuple inside the AD callback so its
    # return type remains inferable across the derivative function barrier.
    f(b) = valuation(NamedTuple{roles}(ntuple(i -> _bumped(curves[i], grid, slice(b, i)), length(curves))))
    z = zeros(k * n)
    result = _ad_derivatives(f, z, order)
    value, g = result.value, result.gradient
    gradient = NamedTuple{roles}(ntuple(i -> slice(g, i), k))
    order == 1 && return (; value, gradient)
    h = result.hessian
    block(i, j) = k == 1 ? h : view(h, indices(i), indices(j))
    hessian = NamedTuple{roles}(ntuple(i -> NamedTuple{roles}(ntuple(j -> block(i, j), k)), k))
    return (; value, gradient, hessian)
end

# Compatibility adapters for the public single- and two-curve return shapes.
function _keyrate_ad(curve::AYM, tenors::AbstractVector, valuation_fn::F; order = 1) where {F}
    r = _ncurve_ad(c -> valuation_fn(c.curve), (; curve), tenors; order)
    result = (; value = r.value, gradient = r.gradient.curve)
    return order == 1 ? result : merge(result, (; hessian = r.hessian.curve.curve))
end
function _keyrate_ad(base::AYM, credit::AYM, tenors::AbstractVector, valuation_fn::F; order = 1) where {F}
    r = _ncurve_ad(c -> valuation_fn(c.base, c.credit), (; base, credit), tenors; order)
    result = (; value = r.value, base_gradient = r.gradient.base, credit_gradient = r.gradient.credit)
    return order == 1 ? result : merge(
            result, (;
                base_hessian = r.hessian.base.base,
                credit_hessian = r.hessian.credit.credit,
                cross_hessian = r.hessian.base.credit,
            )
        )
end

# ─── Closed-form KRD for the vanilla cashflow case ──────────────────────
#
# When the valuation function is just `Σ cf_k · disc(curve, t_k)`, the
# gradient and Hessian of V(b) w.r.t. the hat-bump vector b are linear /
# quadratic in the hat weights at each cashflow time. The triangular hats
# only support 1–2 pillars per time, so each t_k writes at most 2 entries
# of the gradient and a 2×2 Hessian block. Total work is O(N_cf),
# independent of the number of KRD pillars — no ForwardDiff Dual
# arithmetic over an N-wide partials vector. Numerically equivalent to
# `_keyrate_ad` for these inputs; just much cheaper for typical bond /
# liability cashflow vectors.

# Active hat pair at `t`. Returns (i, w_i, j, w_j) such that the hat sum
# at t equals `w_i * b[i] + w_j * b[j]`. At/beyond the endpoints only one
# hat is active (the other weight is 0 and j == i).
@inline function _active_hats(tenors, t)
    n = length(tenors)
    if t <= first(tenors)
        return 1, one(float(t)), 1, zero(float(t))
    elseif t >= last(tenors)
        return n, one(float(t)), n, zero(float(t))
    else
        i = searchsortedlast(tenors, t)
        w_right = (t - tenors[i]) / (tenors[i + 1] - tenors[i])
        w_left = one(w_right) - w_right
        return i, w_left, i + 1, w_right
    end
end

# Single- and two-curve analytic KRD are the L = 1 and L = 2 cases of the
# multi-curve kernel `_ncurve_analytic` (just below). For the vanilla
# `Σ cf · ∏ disc` valuation the per-role gradients and all three Hessian blocks
# (base, credit, cross) coincide — see the kernel's note — so the two-curve
# adapter aliases the single shared gradient / Hessian into the base / credit /
# cross names. Downstream callers only broadcast (`./`) these, never mutate them
# in place, so the aliasing is safe and the public two-curve forms still hand
# back distinct output buffers.
_keyrate_analytic(curve::AYM, tenors::AbstractVector, cfs::AbstractVector, times; order = 1) =
    _ncurve_analytic((; curve), tenors, cfs, times; order)

function _keyrate_analytic(
        base::AYM, credit::AYM, tenors::AbstractVector,
        cfs::AbstractVector, times; order = 1
    )
    an = _ncurve_analytic((; base, credit), tenors, cfs, times; order)
    order >= 2 || return (;
        value = an.value,
        base_gradient = an.gradient, credit_gradient = an.gradient,
    )
    return (;
        value = an.value,
        base_gradient = an.gradient, credit_gradient = an.gradient,
        base_hessian = an.hessian, credit_hessian = an.hessian, cross_hessian = an.hessian,
    )
end

# N-curve analytic. `curves::NamedTuple{roles}` of L curves with a shared tenor
# grid; the discount is the product ∏_layers disc_layer(t). All roles must be
# discount-role layers (multiplicatively composed); do not pass `:index`.
# Under multiplicative composition every per-role gradient and every (role,
# role) Hessian block carry identical values, so the helper returns a single
# shared gradient vector and a single shared Hessian matrix. The single-, two-,
# and N-curve public wrappers all delegate here and alias these across their
# role positions.
function _ncurve_analytic(
        curves::NamedTuple, tenors::AbstractVector,
        cfs::AbstractVector, times; order = 1
    )
    KeyRates(tenors)
    checkbounds(Bool, times, eachindex(cfs)) || throw(
        DimensionMismatch("times must contain at least one entry for each cashflow")
    )
    L = length(curves)
    n = length(tenors)
    T = float(promote_type(eltype(cfs), eltype(times)))
    grad_shared = zeros(T, n)
    hess_shared = order >= 2 ? zeros(T, n, n) : nothing
    V = zero(T)
    @inbounds for k in eachindex(cfs)
        t = times[k]
        # `prod` over the curve tuple is unrolled and type-stable even when the
        # roles have different concrete types (e.g. a ZeroRateCurve base with a
        # flat Constant credit). A `for c in values(curves)` loop would make `c`
        # non-concrete for a heterogeneous tuple and box `discount(c, t)` once
        # per cashflow — an O(N_cf) allocation hit on the two-curve IR01/CS01 path.
        d = prod(c -> FinanceCore.discount(c, t), values(curves))
        cfd = cfs[k] * d
        V += cfd
        i, wi, j, wj = _active_hats(tenors, t)
        grad_shared[i] -= t * cfd * wi
        if i != j
            grad_shared[j] -= t * cfd * wj
        end
        if order >= 2
            tt = t * t * cfd
            hess_shared[i, i] += tt * wi * wi
            if i != j
                ij = tt * wi * wj
                hess_shared[i, j] += ij
                hess_shared[j, i] += ij
                hess_shared[j, j] += tt * wj * wj
            end
        end
    end
    if order >= 2
        return (; value = V, gradient = grad_shared, hessian = hess_shared)
    else
        return (; value = V, gradient = grad_shared)
    end
end

# Normalize the three Hessian blocks of a two-curve AD/analytic result into the
# `(; base, credit, cross)` convexity NamedTuple. Each `./` allocates a fresh
# array, so callers always receive distinct output buffers even when the
# analytic inputs alias a single shared matrix.
_conv_blocks(r) = (;
    base = r.base_hessian ./ r.value,
    credit = r.credit_hessian ./ r.value,
    cross = r.cross_hessian ./ r.value,
)

## AbstractYieldModel + KeyRates(tenors): KRD / IR01 / CS01 / convexity / sensitivities
#
# These dispatches accept any `FinanceModels.Yield.AbstractYieldModel`. The
# KRD knot grid is carried by `KeyRates(tenors)`. Internally the AD path layers
# a hat-function zero-rate bump over the user's curve via `Yield.TenorShift`;
# the user's curve is never resampled or rebuilt.
#
# `ZeroRateCurve` inputs go through the same path — it has no special dispatch.
#
# Tenor grid is required (no default) because KRD bucket conventions vary
# (Bloomberg, FRTB, BMA SBA, etc.); downstream should choose explicitly.

"""
    duration(valuation_fn, curve::AbstractYieldModel, tenors) -> scalar
    duration(curve::AbstractYieldModel, tenors, cfs, times) -> scalar
    duration(curve::AbstractYieldModel, tenors, cfs::AbstractVector{<:Cashflow}) -> scalar

Scalar modified duration for any `AbstractYieldModel` evaluated against a KRD
knot grid. Equivalent to `sum(duration(KeyRates(tenors), ...))`.

Use [`KeyRates`](@ref) to obtain the per-knot vector decomposition.

# Example
```julia
duration(pv, my_composite_curve, [0.25, 1, 5, 10, 30])
```
"""
function duration(valuation_fn::Function, curve::AYM, tenors)
    return sum(duration(KeyRates(tenors), valuation_fn, curve))
end
function duration(curve::AYM, tenors, cfs, times)
    return sum(duration(KeyRates(tenors), curve, cfs, times))
end
duration(curve::AYM, tenors, cfs::AbstractVector{<:FinanceCore.Cashflow}) = duration(curve, tenors, _extract_cfs_times(cfs)...)

"""
    duration(kr::KeyRates, valuation_fn, curve::AbstractYieldModel) -> Vector
    duration(kr::KeyRates, curve::AbstractYieldModel, cfs, times) -> Vector
    duration(kr::KeyRates, curve::AbstractYieldModel, cfs::AbstractVector{<:Cashflow}) -> Vector

Key-rate durations (modified) for any `AbstractYieldModel`, computed by
layering a triangular-hat zero-rate bump at each tenor in `kr.tenors` over
the user's curve via `Yield.TenorShift`, then taking the AD gradient w.r.t.
the bump magnitudes. The user's curve is preserved at all non-knot points.

# Tenor grid

`kr.tenors` is the KRD knot grid — a separate modeling choice from any
tenor structure baked into the curve itself. You can evaluate key-rate
durations on any grid (e.g. Bloomberg `{0.25, 1, 2, 5, 10, 30}`, FRTB
`{0.25, 0.5, 1, 2, 3, 5, 10, 15, 20, 30}`, etc.) without re-fitting the
underlying curve.

The grid must be sorted ascending, distinct, and strictly positive. These
preconditions are not checked at runtime — a malformed grid produces wrong
gradients silently.

# Bump shape and endpoint extrapolation

The bump at the i-th knot is a triangular hat centered at `tenors[i]` with
support `[tenors[i-1], tenors[i+1]]`. Outside the knot range it is flat:
bumping `tenors[1]` perturbs all cashflows at `t ≤ tenors[1]` equally, and
bumping `tenors[end]` perturbs all cashflows at `t ≥ tenors[end]` equally.
For long-duration insurance liabilities (LTC, deferred / payout annuities),
the last-knot KRD absorbs all super-tenor sensitivity — extend the grid
past your longest cashflow if you want that decomposed.

For a linearly-interpolated zero-rate curve the result matches AD over
the curve's own rates exactly. For other splines the bump kernel is
hat-shaped rather than spline-shaped, so per-knot KRDs shift slightly;
the sum of KRDs (= scalar modified duration) is invariant either way.

# Example
```julia
duration(KeyRates([0.25, 1, 5, 10, 30]), pv, curve)

duration(KeyRates([0.25, 1, 5, 10, 30]), curve) do c
    pv(c)
end
```
"""
function duration(kr::KeyRates, valuation_fn::Function, curve::AYM)
    ad = _keyrate_ad(curve, kr.tenors, valuation_fn)
    return -ad.gradient ./ ad.value
end
function duration(kr::KeyRates, curve::AYM, cfs, times)
    an = _keyrate_analytic(curve, kr.tenors, cfs, times)
    return -an.gradient ./ an.value
end
duration(kr::KeyRates, curve::AYM, cfs::AbstractVector{<:FinanceCore.Cashflow}) = duration(kr, curve, _extract_cfs_times(cfs)...)

"""
    duration(::DV01, valuation_fn, curve::AbstractYieldModel, tenors) -> scalar
    duration(::DV01, curve::AbstractYieldModel, tenors, cfs, times) -> scalar
    duration(::DV01, kr::KeyRates, valuation_fn, curve::AbstractYieldModel) -> Vector
    duration(::DV01, kr::KeyRates, curve::AbstractYieldModel, cfs, times) -> Vector

DV01 (scalar or per-knot vector) for any `AbstractYieldModel`. Equivalent to
the `KeyRates` variants of `duration` but in dollars per basis point.
"""
function duration(::DV01, valuation_fn::Function, curve::AYM, tenors)
    return sum(duration(DV01(), KeyRates(tenors), valuation_fn, curve))
end
function duration(::DV01, curve::AYM, tenors, cfs, times)
    return sum(duration(DV01(), KeyRates(tenors), curve, cfs, times))
end
duration(::DV01, curve::AYM, tenors, cfs::AbstractVector{<:FinanceCore.Cashflow}) = duration(DV01(), curve, tenors, _extract_cfs_times(cfs)...)

function duration(::DV01, kr::KeyRates, valuation_fn::Function, curve::AYM)
    ad = _keyrate_ad(curve, kr.tenors, valuation_fn)
    return -ad.gradient ./ 10_000
end
function duration(::DV01, kr::KeyRates, curve::AYM, cfs, times)
    an = _keyrate_analytic(curve, kr.tenors, cfs, times)
    return -an.gradient ./ 10_000
end
duration(::DV01, kr::KeyRates, curve::AYM, cfs::AbstractVector{<:FinanceCore.Cashflow}) = duration(DV01(), kr, curve, _extract_cfs_times(cfs)...)

"""
    duration(::IR01, valuation_fn, base::AbstractYieldModel, credit::AbstractYieldModel, tenors) -> scalar
    duration(::IR01, base::AbstractYieldModel, credit::AbstractYieldModel, tenors, cfs, times) -> scalar
    duration(::IR01, kr::KeyRates, valuation_fn, base, credit) -> Vector
    duration(::IR01, kr::KeyRates, base, credit, cfs, times) -> Vector
    duration(::CS01, ...) -> ...

Two-curve IR01/CS01 for any `AbstractYieldModel` pair sharing a tenor
grid. IR01 bumps the base (risk-free) curve only; CS01 bumps the credit
(spread) curve only.
"""
function duration(::IR01, valuation_fn::Function, base::AYM, credit::AYM, tenors)
    return sum(duration(IR01(), KeyRates(tenors), valuation_fn, base, credit))
end
function duration(::IR01, base::AYM, credit::AYM, tenors, cfs, times)
    return sum(duration(IR01(), KeyRates(tenors), base, credit, cfs, times))
end
duration(::IR01, base::AYM, credit::AYM, tenors, cfs::AbstractVector{<:FinanceCore.Cashflow}) = duration(IR01(), base, credit, tenors, _extract_cfs_times(cfs)...)

function duration(::IR01, kr::KeyRates, valuation_fn::Function, base::AYM, credit::AYM)
    ad = _keyrate_ad(base, credit, kr.tenors, valuation_fn)
    return -ad.base_gradient ./ 10_000
end
function duration(::IR01, kr::KeyRates, base::AYM, credit::AYM, cfs, times)
    an = _keyrate_analytic(base, credit, kr.tenors, cfs, times)
    return -an.base_gradient ./ 10_000
end
duration(::IR01, kr::KeyRates, base::AYM, credit::AYM, cfs::AbstractVector{<:FinanceCore.Cashflow}) = duration(IR01(), kr, base, credit, _extract_cfs_times(cfs)...)

function duration(::CS01, valuation_fn::Function, base::AYM, credit::AYM, tenors)
    return sum(duration(CS01(), KeyRates(tenors), valuation_fn, base, credit))
end
function duration(::CS01, base::AYM, credit::AYM, tenors, cfs, times)
    return sum(duration(CS01(), KeyRates(tenors), base, credit, cfs, times))
end
duration(::CS01, base::AYM, credit::AYM, tenors, cfs::AbstractVector{<:FinanceCore.Cashflow}) = duration(CS01(), base, credit, tenors, _extract_cfs_times(cfs)...)

function duration(::CS01, kr::KeyRates, valuation_fn::Function, base::AYM, credit::AYM)
    ad = _keyrate_ad(base, credit, kr.tenors, valuation_fn)
    return -ad.credit_gradient ./ 10_000
end
function duration(::CS01, kr::KeyRates, base::AYM, credit::AYM, cfs, times)
    an = _keyrate_analytic(base, credit, kr.tenors, cfs, times)
    return -an.credit_gradient ./ 10_000
end
duration(::CS01, kr::KeyRates, base::AYM, credit::AYM, cfs::AbstractVector{<:FinanceCore.Cashflow}) = duration(CS01(), kr, base, credit, _extract_cfs_times(cfs)...)

# Do-block-first forwarders (support `f(args...) do x; ...; end` syntax)
duration(vf::Function, kr::KeyRates, curve::AYM) = duration(kr, vf, curve)
duration(vf::Function, ::DV01, curve::AYM, tenors) = duration(DV01(), vf, curve, tenors)
duration(vf::Function, ::DV01, kr::KeyRates, curve::AYM) = duration(DV01(), kr, vf, curve)
duration(vf::Function, ::IR01, base::AYM, credit::AYM, tenors) = duration(IR01(), vf, base, credit, tenors)
duration(vf::Function, ::IR01, kr::KeyRates, base::AYM, credit::AYM) = duration(IR01(), kr, vf, base, credit)
duration(vf::Function, ::CS01, base::AYM, credit::AYM, tenors) = duration(CS01(), vf, base, credit, tenors)
duration(vf::Function, ::CS01, kr::KeyRates, base::AYM, credit::AYM) = duration(CS01(), kr, vf, base, credit)

"""
    convexity(valuation_fn, curve::AbstractYieldModel, tenors) -> scalar
    convexity(curve::AbstractYieldModel, tenors, cfs, times) -> scalar
    convexity(kr::KeyRates, valuation_fn, curve::AbstractYieldModel) -> Matrix
    convexity(kr::KeyRates, curve::AbstractYieldModel, cfs, times) -> Matrix
    convexity(base::AbstractYieldModel, credit::AbstractYieldModel, tenors, cfs, times) -> NamedTuple
    convexity(kr::KeyRates, base, credit, cfs, times) -> NamedTuple
    convexity(kr::KeyRates, curves::NamedTuple, cfs, times) -> NamedTuple{roles}{roles}

Key-rate convexity (matrix) and scalar convexity for any `AbstractYieldModel`,
pair, or named tuple of discount-role curves. Mirrors `duration` but returns
∂²V/∂rᵢ∂rⱼ rather than ∂V/∂rᵢ.

For the `NamedTuple` form, every named curve must be a discount-role layer
(multiplicatively composed); do not pass `:index`. Per-role and per-pair
outputs alias a single shared matrix — values coincide by construction
under multiplicative composition. `copy` if you need independent buffers.

The scalar forms (first two signatures) return the parallel-shift second
derivative ∂²V/∂s² under a *continuous-rate* shock — matching the matrix
forms exactly under partition of unity of the KRD hats. `tenors` is accepted
for API symmetry but is not used by the scalar derivative computation.

If you also want the durations / DV01s, prefer [`sensitivities`](@ref) — it returns
the value, gradient, and Hessian from one AD pass at the same cost.
"""
# Continuous-shock parallel-shift convexity via a single scalar second
# derivative. Under partition of unity of the KRD hat functions (`_hat_bump`
# above), `sum(convexity(KeyRates(tenors), …))` equals ∂²V/∂s² for parallel
# shift `s` by the chain rule — the matrix path returns the right number but
# pays O(N² AD work + dense Hessian allocation) for what is an O(1) scalar
# second derivative. This helper performs the scalar derivative directly on a
# `TenorShift`-bumped curve, matching the matrix-sum form exactly while
# avoiding the per-pillar Hessian.
_parallel_bumped(curve, s) = FinanceModels.Yield.TenorShift(curve, (z, t) -> z + FinanceCore.Continuous(s))

function _parallel_continuous_convexity(curve::AYM, valuation_fn)
    v(s) = abs(valuation_fn(_parallel_bumped(curve, s)))
    ∂²V = ForwardDiff.derivative(s2 -> ForwardDiff.derivative(v, s2), 0.0)
    return ∂²V / v(0.0)
end

convexity(valuation_fn::Function, curve::AYM, _tenors) =
    _parallel_continuous_convexity(curve, valuation_fn)
convexity(curve::AYM, _tenors, cfs, times) =
    _parallel_continuous_convexity(curve, c -> sum(_cf_value(cfs[k]) * FinanceCore.discount(c, times[k]) for k in eachindex(cfs)))
convexity(curve::AYM, _tenors, cfs::AbstractVector{<:FinanceCore.Cashflow}) =
    convexity(curve, _tenors, _extract_cfs_times(cfs)...)

function convexity(kr::KeyRates, valuation_fn::Function, curve::AYM)
    ad = _keyrate_ad(curve, kr.tenors, valuation_fn; order = 2)
    return ad.hessian ./ ad.value
end
function convexity(kr::KeyRates, curve::AYM, cfs, times)
    an = _keyrate_analytic(curve, kr.tenors, cfs, times; order = 2)
    return an.hessian ./ an.value
end
convexity(kr::KeyRates, curve::AYM, cfs::AbstractVector{<:FinanceCore.Cashflow}) = convexity(kr, curve, _extract_cfs_times(cfs)...)

function convexity(valuation_fn::Function, base::AYM, credit::AYM, tenors)
    cv = convexity(KeyRates(tenors), valuation_fn, base, credit)
    return (; base = sum(cv.base), credit = sum(cv.credit), cross = sum(cv.cross))
end
function convexity(base::AYM, credit::AYM, tenors, cfs, times)
    # static cashflows: the analytic helper computes the same blocks as the
    # (2n)×(2n) ForwardDiff Hessian the do-block form pays for, in O(N_cf)
    an = _keyrate_analytic(base, credit, tenors, cfs, times; order = 2)
    return (;
        base = sum(an.base_hessian) / an.value,
        credit = sum(an.credit_hessian) / an.value,
        cross = sum(an.cross_hessian) / an.value,
    )
end
convexity(base::AYM, credit::AYM, tenors, cfs::AbstractVector{<:FinanceCore.Cashflow}) = convexity(base, credit, tenors, _extract_cfs_times(cfs)...)

function convexity(kr::KeyRates, valuation_fn::Function, base::AYM, credit::AYM)
    ad = _keyrate_ad(base, credit, kr.tenors, valuation_fn; order = 2)
    return _conv_blocks(ad)
end
function convexity(kr::KeyRates, base::AYM, credit::AYM, cfs, times)
    an = _keyrate_analytic(base, credit, kr.tenors, cfs, times; order = 2)
    return _conv_blocks(an)
end
convexity(kr::KeyRates, base::AYM, credit::AYM, cfs::AbstractVector{<:FinanceCore.Cashflow}) = convexity(kr, base, credit, _extract_cfs_times(cfs)...)

# Multi-curve NamedTuple cashflow form. Per-role and per-role-pair convexities
# for static cashflows. All L² blocks alias one shared N×N matrix — values
# coincide by construction under multiplicative discount composition.
function convexity(kr::KeyRates, curves::NamedTuple, cfs, times)
    an = _ncurve_analytic(curves, kr.tenors, cfs, times; order = 2)
    roles = keys(curves)
    L = length(roles)
    normalized = an.hessian ./ an.value
    return NamedTuple{roles}(ntuple(_ -> NamedTuple{roles}(ntuple(_ -> normalized, L)), L))
end
convexity(kr::KeyRates, curves::NamedTuple, cfs::AbstractVector{<:FinanceCore.Cashflow}) =
    convexity(kr, curves, _extract_cfs_times(cfs)...)

# Do-block-first forwarders (support `f(args...) do x; ...; end` syntax)
convexity(vf::Function, kr::KeyRates, curve::AYM) = convexity(kr, vf, curve)
convexity(vf::Function, kr::KeyRates, base::AYM, credit::AYM) = convexity(kr, vf, base, credit)

"""
    sensitivities(kr::KeyRates, valuation_fn, curve::AbstractYieldModel) -> NamedTuple
    sensitivities(kr::KeyRates, curve::AbstractYieldModel, cfs, times) -> NamedTuple
    sensitivities(::DV01, kr::KeyRates, valuation_fn, curve::AbstractYieldModel) -> NamedTuple
    sensitivities(kr::KeyRates, base::AbstractYieldModel, credit::AbstractYieldModel, cfs, times) -> NamedTuple
    sensitivities(::DV01, kr::KeyRates, base, credit, cfs, times) -> NamedTuple

Bundled value + key-rate durations (or DV01s) + convexity matrix for any
`AbstractYieldModel` or pair, in a single AD pass. The knot grid is carried
by [`KeyRates`](@ref).
"""
function sensitivities(kr::KeyRates, valuation_fn::Function, curve::AYM)
    ad = _keyrate_ad(curve, kr.tenors, valuation_fn; order = 2)
    return (;
        value = ad.value,
        durations = -ad.gradient ./ ad.value,
        convexities = ad.hessian ./ ad.value,
    )
end
function sensitivities(kr::KeyRates, curve::AYM, cfs, times)
    an = _keyrate_analytic(curve, kr.tenors, cfs, times; order = 2)
    return (;
        value = an.value,
        durations = -an.gradient ./ an.value,
        convexities = an.hessian ./ an.value,
    )
end
sensitivities(kr::KeyRates, curve::AYM, cfs::AbstractVector{<:FinanceCore.Cashflow}) = sensitivities(kr, curve, _extract_cfs_times(cfs)...)

function sensitivities(::DV01, kr::KeyRates, valuation_fn::Function, curve::AYM)
    ad = _keyrate_ad(curve, kr.tenors, valuation_fn; order = 2)
    return (;
        value = ad.value,
        dv01s = -ad.gradient ./ 10_000,
        convexities = ad.hessian ./ ad.value,
    )
end
function sensitivities(::DV01, kr::KeyRates, curve::AYM, cfs, times)
    an = _keyrate_analytic(curve, kr.tenors, cfs, times; order = 2)
    return (;
        value = an.value,
        dv01s = -an.gradient ./ 10_000,
        convexities = an.hessian ./ an.value,
    )
end
sensitivities(::DV01, kr::KeyRates, curve::AYM, cfs::AbstractVector{<:FinanceCore.Cashflow}) = sensitivities(DV01(), kr, curve, _extract_cfs_times(cfs)...)

function sensitivities(kr::KeyRates, valuation_fn::Function, base::AYM, credit::AYM)
    ad = _keyrate_ad(base, credit, kr.tenors, valuation_fn; order = 2)
    return (;
        value = ad.value,
        base_durations = -ad.base_gradient ./ ad.value,
        credit_durations = -ad.credit_gradient ./ ad.value,
        convexities = _conv_blocks(ad),
    )
end
function sensitivities(kr::KeyRates, base::AYM, credit::AYM, cfs, times)
    an = _keyrate_analytic(base, credit, kr.tenors, cfs, times; order = 2)
    return (;
        value = an.value,
        base_durations = -an.base_gradient ./ an.value,
        credit_durations = -an.credit_gradient ./ an.value,
        convexities = _conv_blocks(an),
    )
end
sensitivities(kr::KeyRates, base::AYM, credit::AYM, cfs::AbstractVector{<:FinanceCore.Cashflow}) = sensitivities(kr, base, credit, _extract_cfs_times(cfs)...)

# Multi-curve NamedTuple cashflow form. One AD-free pass returns per-role
# durations + per-role-pair N×N convexity blocks. All per-role durations
# alias one shared vector; all L² Hessian blocks alias one shared matrix —
# values coincide by construction under multiplicative discount composition.
function sensitivities(kr::KeyRates, curves::NamedTuple, cfs, times)
    an = _ncurve_analytic(curves, kr.tenors, cfs, times; order = 2)
    roles = keys(curves)
    L = length(roles)
    dur_normalized = -an.gradient ./ an.value
    conv_normalized = an.hessian ./ an.value
    durations = NamedTuple{roles}(ntuple(_ -> dur_normalized, L))
    convexities = NamedTuple{roles}(ntuple(_ -> NamedTuple{roles}(ntuple(_ -> conv_normalized, L)), L))
    return (; value = an.value, durations, convexities)
end
sensitivities(kr::KeyRates, curves::NamedTuple, cfs::AbstractVector{<:FinanceCore.Cashflow}) =
    sensitivities(kr, curves, _extract_cfs_times(cfs)...)

function sensitivities(::DV01, kr::KeyRates, valuation_fn::Function, base::AYM, credit::AYM)
    ad = _keyrate_ad(base, credit, kr.tenors, valuation_fn; order = 2)
    return (;
        value = ad.value,
        base_dv01s = -ad.base_gradient ./ 10_000,
        credit_dv01s = -ad.credit_gradient ./ 10_000,
        convexities = _conv_blocks(ad),
    )
end
function sensitivities(::DV01, kr::KeyRates, base::AYM, credit::AYM, cfs, times)
    an = _keyrate_analytic(base, credit, kr.tenors, cfs, times; order = 2)
    return (;
        value = an.value,
        base_dv01s = -an.base_gradient ./ 10_000,
        credit_dv01s = -an.credit_gradient ./ 10_000,
        convexities = _conv_blocks(an),
    )
end
sensitivities(::DV01, kr::KeyRates, base::AYM, credit::AYM, cfs::AbstractVector{<:FinanceCore.Cashflow}) = sensitivities(DV01(), kr, base, credit, _extract_cfs_times(cfs)...)

# Do-block-first forwarders (support `f(args...) do x; ...; end` syntax)
sensitivities(vf::Function, kr::KeyRates, curve::AYM) = sensitivities(kr, vf, curve)
sensitivities(vf::Function, ::DV01, kr::KeyRates, curve::AYM) = sensitivities(DV01(), kr, vf, curve)
sensitivities(vf::Function, kr::KeyRates, base::AYM, credit::AYM) = sensitivities(kr, vf, base, credit)
sensitivities(vf::Function, ::DV01, kr::KeyRates, base::AYM, credit::AYM) = sensitivities(DV01(), kr, vf, base, credit)

"""
    sensitivities(valuation, curves::NamedTuple; tenors) -> (; value, duration, dv01, key_rate)
    sensitivities(target, tenors; discount::NamedTuple, index) -> same

Multi-curve sensitivities: differentiate `valuation(curves)` w.r.t. each named curve
in `curves` in a single AD pass, returning a per-role `duration`/`dv01`/`key_rate`
NamedTuple. The structured form assembles `discount = sum(discount layers)` and projects
the contract's coupons on `index` — e.g. `discount = (; rf, credit, ilp)` gives `r.duration.rf`
(≈ IR01), `.credit` (≈ CS01), `.ilp` ("ILP01"), and `.index` (the reset sensitivity). ILP /
matching-adjustment / basis are just additional named curves.
"""
function sensitivities(valuation, curves::NamedTuple; tenors)
    r = _ncurve_ad(valuation, curves, tenors; order = 1)
    v, grads = r.value, r.gradient
    roles = keys(curves)
    return (;
        value = v,
        duration = NamedTuple{roles}(map(g -> -sum(g) / v, values(grads))),
        dv01 = NamedTuple{roles}(map(g -> -sum(g) / 10_000, values(grads))),
        key_rate = NamedTuple{roles}(map(g -> -g ./ v, values(grads))),
    )
end
sensitivities(curves::NamedTuple, valuation::Function; tenors) = sensitivities(valuation, curves; tenors)  # do-block form
