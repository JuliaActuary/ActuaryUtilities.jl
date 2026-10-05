## Curve shifts
# The continuous-zero shifts every sensitivity applies to a yield model: triangular hats on a
# tenor grid, and a parallel shift as the one-knot grid.

const AYM = FinanceModels.Yield.AbstractYieldModel
const HW = FinanceModels.ShortRate.HullWhite

# Triangular hats with flat extrapolation outside the knot range: at a knot τᵢ the
# bump equals bᵢ; between τᵢ and τᵢ₊₁ it is linear. Returns (i, w_i, j, w_j) such
# that the hat sum at t equals `w_i * b[i] + w_j * b[j]`. At/beyond the endpoints
# only one hat is active (the other weight is 0 and j == i).
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

# The hat-function bump at `t`. A lone knot's hat is flat, so a one-knot grid is a parallel
# shift; beyond the endpoints, the end hats stay at their bumps.
function _hat_bump(tenors, bumps, t)
    length(bumps) == 1 && return only(bumps)
    i, wi, j, wj = _active_hats(tenors, t)
    return i == j ? bumps[i] : wi * bumps[i] + wj * bumps[j]
end

# Layer a hat-function zero-rate bump over `curve` lazily.
_bumped(curve, tenors, bumps) = FinanceModels.Yield.TenorShift(
    curve,
    (z, t) -> FinanceCore.Continuous(_hat_bump(tenors, bumps, t)) + z,
)
# A Hull–White model prices on the curve it is calibrated to, so a bump moves that curve and keeps
# the model, with its mean reversion, volatility and closed forms (caps, swaptions); a shift layered
# over the model would hide them. Its discount factors are the bumped curve's either way.
_bumped(hw::HW, tenors, bumps) =
    ConstructionBase.setproperties(hw; curve = _bumped(hw.curve, tenors, bumps))

# A one-knot grid is an exact parallel shift: its hat is flat everywhere. Every
# yield-model parallel shift uses it, so parallel results are ≈ the sums of key-rate results.
const _PARALLEL_GRID = 1.0:1.0

# Shift every continuous zero rate of a yield model by `shift`. Converting an annual-rate
# increment to continuous compounding would change the second derivative.
_parallel_bumped(curve::AYM, shift) = _bumped(curve, _PARALLEL_GRID, (shift,))

# A scalar or `Rate` moves in its own coordinate.
_parallel_bumped(yield, shift) = yield + shift
