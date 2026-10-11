# Interest-Rate Sensitivities

Calculate durations, DV01s, and convexities for scalar rates, FinanceCore `Rate`s,
FinanceModels [`AbstractYieldModel`](https://github.com/JuliaActuary/FinanceModels.jl)
curves, and contracts. Scalar measures describe a parallel shift. Key-rate measures
decompose risk by tenor with triangular continuous-zero bumps on the original curve.
Callbacks use ForwardDiff through `Yield.TenorShift`; fixed cashflows use analytic
derivatives. See the [autodiff ALM chapter](https://modernfinancialmodeling.com/autodiff_alm)
for background.

Explicit cashflow inputs that are empty or have all-zero amounts return zero value
and risk without valuing any payment. Nonzero amounts that offset to zero present
value retain their dollar exposures and undefined normalized risk. See
[Zero cashflow streams](@ref) for numeric types and portfolio aggregation.

## Shock coordinates

A one-basis-point shift must move some rate, and the dollar result depends on which
one. Each single-rate input moves in its own shock coordinate:

| Input | What moves by `s` | Modified duration | Convexity weight |
|:------|:------------------|:------------------|:-----------------|
| `Real` `y` (annual effective) | `y` | Macaulay / (1 + y) | `t(t+1) / (1+y)²` |
| `Periodic(y, m)` | the nominal rate `y` | Macaulay / (1 + y/m) | `t(t+1/m) / (1+y/m)²` |
| `Continuous(y)` | `y` | Macaulay | `t²` |
| Any `AbstractYieldModel`, including `Yield.Constant` | every continuous zero rate, in parallel | Macaulay | `t²` |
| `KeyRates(tenors)` | continuous zero rates, through a triangular bump at each tenor | per tenor | per tenor pair |
| Contracts with `Effective()`/`Spread()` | `Effective()`: continuous zero rates of the index and discount curves; `Spread()`: the discount curve only | — | — |

DV01 is `-∂V/∂s / 10000` in the same coordinate: it is the first-order value lost
for a rate increase of one basis point. Reversing a position reverses its dollar
risk. Ordinary liability payments entered as negative cashflows contribute negative
DV01, but the sign of a mixed position's present value alone does not determine its
DV01. Parallel curve risk is ≈ the sum of the corresponding key-rate results,
including convexity cross terms (see [Convexity Conventions](@ref)).

Wrapping a scalar in `Yield.Constant` keeps its discount factors but changes what
moves, so duration and DV01 change by a factor of `1 + y`:

```@example sensitivities
using ActuaryUtilities, FinanceModels, FinanceCore

cfs   = [5.0, 5.0, 5.0, 5.0, 105.0]
times = [1.0, 2.0, 3.0, 4.0, 5.0]

(scalar_dv01 = duration(DV01(), 0.03, cfs, times),
 curve_dv01  = duration(DV01(), Yield.Constant(0.03), cfs, times),
 ratio       = duration(DV01(), Yield.Constant(0.03), cfs, times) / duration(DV01(), 0.03, cfs, times))
```

To measure a scalar yield in the curve coordinate, pass the equivalent continuous
rate:

```@example sensitivities
duration(DV01(), Continuous(log1p(0.03)), cfs, times) ≈ duration(DV01(), Yield.Constant(0.03), cfs, times)
```

## Scalar Measures

Without a `KeyRates` marker, duration, DV01, and convexity return parallel risk.
No tenor grid is involved:

```@example sensitivities
rates  = [0.03, 0.03, 0.03, 0.03, 0.03]
tenors = [1.0, 2.0, 3.0, 4.0, 5.0]
zrc    = ZeroRateCurve(rates, tenors)

(duration  = duration(zrc, cfs, times),
 macaulay  = duration(Macaulay(), zrc, cfs, times),
 dv01      = duration(DV01(), zrc, cfs, times),
 convexity = convexity(zrc, cfs, times))
```

For a flat curve, the scalar measures match an explicitly continuous rate at the
same zero-rate level:

```@example sensitivities
(zrc_dur   = duration(zrc, cfs, times),
 rate_dur  = duration(Continuous(0.03), cfs, times),
 zrc_conv  = convexity(zrc, cfs, times),
 rate_conv = convexity(Continuous(0.03), cfs, times))
```

### Two curves: IR01 and CS01

For yield-model inputs, IR01 applies a parallel continuous-zero shift to the base
curve, holding the credit curve fixed; CS01 shifts the credit curve, holding the
base curve fixed. Both use the same sign and one-basis-point scaling as DV01.

!!! note "When rate and credit sensitivities coincide"
    For fixed cashflows discounted at `base + credit`, matching one-basis-point
    shifts to either curve produce the same change in the combined continuous
    zero rates. IR01, CS01 and combined-curve DV01 are therefore equal. The four
    convexity blocks are also equal when the value is nonzero.
    Here, credit is an additive discount spread.

    More generally, the equality requires the same shock coordinate, shape and
    scaling. If value depends only on the sum of two additive shifts,
    ``V(u, v) = F(u + v)``, both first derivatives equal ``F'`` and all three
    second derivatives equal ``F''``. Shifting **both** curves by one basis point
    shifts the combined curve by **two** basis points.

For fixed cashflows, use the combined curve's DV01,
`duration(DV01(), base + credit, cfs, times)`.

```@example sensitivities
base   = ZeroRateCurve([0.03, 0.03, 0.03, 0.03, 0.03], tenors)
credit = ZeroRateCurve([0.02, 0.02, 0.02, 0.02, 0.02], tenors)

duration(DV01(), base + credit, cfs, times)
```

The measures can differ when a valuation uses the two curves differently. Pass the
valuation first; it receives `(base, credit)`. IR01 shifts the base curve in parallel
and CS01 shifts the credit curve.
Compare the fixed cashflows with a five-year floater paying the one-year forward rate plus
a 2% margin. The floater's coupons reset on the base curve, while every payment is
discounted at `base + credit`:

```@example sensitivities
margin = 0.02

fixed_value(b, c) = pv(b + c, cfs, times)
floater_value(b, c) = sum(1:5) do t
    coupon = 100 * (1 / discount(b, t - 1, t) - 1 + margin)   # resets on the base curve
    (coupon + (t == 5 ? 100 : 0)) * discount(b + c, t)       # discounted at base + credit
end

(fixed   = (ir01 = duration(fixed_value, IR01(), base, credit),
            cs01 = duration(fixed_value, CS01(), base, credit)),
 floater = (ir01 = duration(floater_value, IR01(), base, credit),
            cs01 = duration(floater_value, CS01(), base, credit)))
```

The floater's IR01 is smaller because a base-rate move changes both projected coupons
and discount factors, with much of the exposure offsetting. Its CS01 is close to that
of a five-year bond because the credit curve changes only discounting in this example.
For a FinanceModels contract,
[Floating-Rate Instruments: Effective vs Spread Duration](@ref) makes the same
distinction between rate and credit risk.

```@example sensitivities
using CairoMakie

ir01s = [duration(v, IR01(), base, credit) for v in (fixed_value, floater_value)]
cs01s = [duration(v, CS01(), base, credit) for v in (fixed_value, floater_value)]

fig = Figure(size = (650, 320))
ax = Axis(fig[1, 1]; title = "Rate vs credit sensitivity", ylabel = "Value lost per +1bp",
    xticks = (1:2, ["5-year fixed cashflows", "5-year floater"]))
colors = Makie.wong_colors()[1:2]
barplot!(ax, [1, 2, 1, 2], [ir01s; cs01s]; dodge = [1, 1, 2, 2], color = colors[[1, 1, 2, 2]])
Legend(fig[1, 2], [PolyElement(polycolor = c) for c in colors], ["IR01 (base curve)", "CS01 (credit curve)"])
fig
```

Two-curve convexity returns the parallel blocks keyed by role: `base.base`, `credit.credit`, and
the cross block `base.credit`, equal to `credit.base`. They coincide for the fixed cashflows and can
differ for the floater:

```@example sensitivities
(fixed   = convexity(fixed_value, base, credit),
 floater = convexity(floater_value, base, credit))
```

The cross block is the mixed second derivative divided by the initial value;
it contains no extra factor of two. See [Two-curve convexity blocks](@ref) for
the second-order P&L formula.

### Curve shocks and market-quote risk

These are sensitivities to the specified curve shifts, not a universal definition
of market IR01 or CS01. For example, [CDS CS01 in OpenGamma Strata](https://strata.opengamma.io/apidocs/com/opengamma/strata/pricer/credit/SpreadSensitivityCalculator.html)
measures sensitivity to CDS par spreads. Bumping a market quote and recalibrating
a curve can produce a different curve change from a parallel zero-rate shift.
Different shock coordinates or key-rate grids can therefore give different risk
numbers even when the pricing function depends only on a combined discount curve.

A bond's contractual coupons can be fixed while its expected payments depend on
default probabilities and recovery. Such a valuation need not depend on
`base + credit` alone, so IR01 and CS01 can differ. The fixed-cashflow forms here
hold the supplied amounts and times fixed; use a callback or contract when payments
depend on the curves.
For sensitivities to calibration inputs, use [Market Inputs](@ref) with the curve
construction or calibration inside the valuation.

## Key-Rate Decomposition

Pass `KeyRates(tenors)` to decompose risk by tenor. Choose the grid independently of
the curve's own knots, for example the buckets your reporting requires. The grid must
be nonempty, finite, positive, and strictly increasing. The `KeyRates` constructor
validates these requirements, and calculations revalidate the grid in case it has
been mutated.

The first and last bumps stay constant beyond the grid, so sensitivity after the
last tenor belongs to its bucket. Extend the grid to separate exposures at longer
maturities.

```@example sensitivities
# Key-rate durations (modified): vector of -∂V/∂rᵢ / V
krds = duration(KeyRates(tenors), zrc, cfs, times)
```

```@example sensitivities
# Key-rate DV01s: vector of -∂V/∂rᵢ / 10000
dv01s = duration(DV01(), KeyRates(tenors), zrc, cfs, times)
```

```@example sensitivities
# Key-rate convexity matrix: ∂²V/∂rᵢ∂rⱼ / V
conv_matrix = convexity(KeyRates(tenors), zrc, cfs, times)
```

The parallel measures are ≈ the sums of the decomposition:

```@example sensitivities
(duration(zrc, cfs, times) ≈ sum(krds),
 duration(DV01(), zrc, cfs, times) ≈ sum(dv01s),
 convexity(zrc, cfs, times) ≈ sum(conv_matrix))
```

Use [`sensitivities`](@ref) to calculate the value, durations and DV01s together, and with
[`SecondOrder()`](@ref) the convexities and dollar convexities too, from one derivative calculation:

```@example sensitivities
result = sensitivities(SecondOrder(), KeyRates(tenors), zrc, cfs, times)
# result.value     — present value, a number
# result.duration  — key-rate durations (modified, vector)
# result.dv01      — key-rate DV01s (vector)
# result.convexity — key-rate convexity matrix
# result.dollar_convexity — its unnormalized second derivatives, defined at zero value
result
```

The default, [`FirstOrder()`](@ref), leaves out the convexity and takes no second
derivatives. Without `KeyRates`, every field is a number for one parallel shift:

```@example sensitivities
sensitivities(zrc, cfs, times)
```

The two-curve callback forms also accept `KeyRates`; the callback receives the bumped
`(base, credit)` curves. For the fixed cashflows, discount factors multiply, so continuous
zero rates add, and the per-tenor IR01s and CS01s are equal:

```@example sensitivities
(ir01s = duration(fixed_value, IR01(), KeyRates(tenors), base, credit),
 cs01s = duration(fixed_value, CS01(), KeyRates(tenors), base, credit))
```
 For the floater in [Two curves: IR01 and CS01](@ref), base-rate
changes also reset its coupons, so the base and credit key-rate durations differ in
every bucket:

```@example sensitivities
floater_krd = sensitivities(floater_value, KeyRates(tenors), base, credit)

(base_durations   = floater_krd.duration.base,
 credit_durations = floater_krd.duration.credit)
```

Base-rate changes affect coupons and discounting; credit changes affect discounting only.
Each vector sums to ≈ the floater's parallel modified duration for its curve:

```@example sensitivities
(base   = sum(floater_krd.duration.base),
 credit = sum(floater_krd.duration.credit))
```

## Market Inputs

When the valuation builds its own curves from market data, differentiate with
respect to named input vectors instead of curve bumps. Each input element is bumped
in the units you pass in:

```@example sensitivities
linear_curve(z) = ZeroRateCurve(z, tenors, Spline.Linear())

inputs = sensitivities((; zeros = [0.02, 0.025, 0.03, 0.035, 0.04], spread = [0.01])) do m
    curve = linear_curve(m.zeros) + Yield.Constant(Continuous(only(m.spread)))
    pv(curve, cfs, times)
end

(value         = inputs.value,
 zero_dv01s    = inputs.dv01.zeros,          # per input element
 spread_dv01   = only(inputs.dv01.spread))   # the one spread element
```

The result is keyed by input name, with one entry per element, as for curves on a
`KeyRates` grid.

With linear zero-rate interpolation and payments inside the knot range, the
per-element results equal the `KeyRates` decomposition on the same knots.

### Par-yield sensitivities: AD and bump-and-reprice

To measure sensitivity to par yields, rebuild the curve from par quotes inside the
valuation. `KeyRates(tenors)` instead bumps continuous zero rates on an existing
curve; these are different shock coordinates.

The following example uses annualized, semiannually compounded par yields in
decimal units. It bootstraps a curve with linear interpolation of continuous zero
rates, then values a seven-year bond with fixed annual payments:

```@example par_quote_risk
using ActuaryUtilities, FinanceModels, FinanceCore

par_tenors = [1.0, 2.0, 3.0, 5.0, 10.0]
par_yields = [0.02, 0.024, 0.028, 0.032, 0.035]
payments = [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 104.0]
payment_times = collect(1.0:7.0)

function curve_from_par(yields)
    quotes = ParYield.(yields, par_tenors; frequency = Periodic(2))
    return fit(Spline.Linear(), quotes, Fit.Bootstrap())
end
par_value(yields) = pv(curve_from_par(yields), payments, payment_times)

ad = sensitivities((; par_yields)) do inputs
    par_value(inputs.par_yields)
end

(tenors = par_tenors,
 par_durations = ad.duration.par_yields,
 par_dv01s = ad.dv01.par_yields)
```

Each entry differentiates one par quote while holding the other quotes fixed and
recalibrating the curve. `par_dv01s` is ``-\partial V/\partial q_i \times 10^{-4}``:
the first-order value loss per one-basis-point increase. `par_durations` is
``-\partial V/\partial q_i / V``; it divides by the value on the unbumped
**recalibrated** curve.
The sum of the buckets describes a parallel shift of these par quotes, not a
parallel continuous-zero shift. The calculation requires a differentiable fit;
FinanceModels' calibration checks still apply.

For a traditional finite bump, choose `bump_bps` and reprice both scenarios. This
example moves only the five-year par quote by ten basis points in each direction:

```@example par_quote_risk
bump_bps = 10.0                        # positive bump size in basis points
h = bump_bps / 10_000
i = findfirst(==(5.0), par_tenors)
up, down = copy(par_yields), copy(par_yields)
up[i] += h
down[i] -= h

v0 = par_value(par_yields)
vup, vdown = par_value(up), par_value(down)

bump_result = (
    up_change = vup - v0,              # actual signed P&L for the +10 bp scenario
    down_change = vdown - v0,          # actual signed P&L for the -10 bp scenario
    dv01 = (vdown - vup) / (2 * bump_bps),
    duration = (vdown - vup) / (2 * h * v0),
)
(bump_result = bump_result,
 ad_dv01 = ad.dv01.par_yields[i],
 ad_duration = ad.duration.par_yields[i])
```

The central-difference `dv01` and `duration` estimate the derivatives and approach
the AD results as the bump shrinks, until numerical precision limits the comparison.
At a finite bump, the up and down P&Ls need not be opposites. To shock every par quote together, use
`up = par_yields .+ h` and `down = par_yields .- h`; its AD comparison is
`sum(ad.dv01.par_yields)` or `sum(ad.duration.par_yields)`.

If starting from an existing `curve`, obtain synthetic quotes with
`[rate(par(curve, t; frequency = 2)) for t in par_tenors]` and use those as
`par_yields`. The rebuilt curve need not reproduce the original curve between
quote maturities. This measures risk to that synthetic par calibration, not
necessarily to the original market inputs. To measure original-input risk, use
those inputs and their actual calibration procedure in the callback.

This is the migration route for `KeyRatePar`. Its old default was a ±10 bp bump,
linear bootstrapping, and normalization by the **original** curve's price. The
finite-bump example has the same par-bump interpretation; to retain that original
normalization, replace `v0` in the duration denominator with
`pv(curve, payments, payment_times)`. Results can differ from v5 where fitting or
quote conventions changed.

## Callable Valuations

Every callback form takes the valuation first, then any marker, then the rate, curves,
or inputs. A valuation can be a function or a callable struct that holds its input data:

```@example sensitivities
struct CashflowValuation{C, T}
    amounts::C
    times::T
end
(v::CashflowValuation)(curve) = pv(curve, v.amounts, v.times)

valuation = CashflowValuation(cfs, tenors)
duration(valuation, zrc)
convexity(valuation, zrc)
sensitivities(valuation, KeyRates(tenors), zrc)
```

Because the valuation comes first, do-block syntax works for every input, including
scalar and `Rate` inputs:

```@example sensitivities
duration(DV01(), Periodic(0.03, 2)) do rate
    pv(rate, cfs, times)
end
```

## Using Cashflow Objects

Yield-model cashflow methods accept `Vector{Cashflow}` directly:

```@example sensitivities
cfs_obj = Cashflow.([5.0, 5.0, 5.0, 5.0, 105.0], [1.0, 2.0, 3.0, 4.0, 5.0])

# These are equivalent:
a = duration(zrc, cfs_obj)                                                            # using Cashflow objects
b = duration(zrc, [5.0, 5.0, 5.0, 5.0, 105.0], [1.0, 2.0, 3.0, 4.0, 5.0])  # using amounts + times
(a, b, a ≈ b)
```

This applies to duration, DV01, convexity, and `sensitivities`.

A `Cashflow` uses its own time even when you pass `times`; explicit times apply only
to numeric amounts. A time vector needs one entry per cashflow; extra trailing
entries are ignored:

```@example sensitivities
fallback_times = fill(10.0, length(cfs_obj))
duration(KeyRates(tenors), zrc, cfs_obj, fallback_times) ≈
    duration(KeyRates(tenors), zrc, cfs_obj)
```

To change payment dates, construct new `Cashflow` objects or pass numeric amounts with the
desired times.

## Other yield models

The same API works with parametric curves such as Nelson–Siegel:

```@example sensitivities
ns        = Yield.NelsonSiegel(1.0, 0.04, -0.02, 0.01)
ns_knots  = [1.0, 2.0, 5.0, 10.0, 20.0]
ns_result = sensitivities(KeyRates(ns_knots), ns,
                          [5.0, 5.0, 5.0, 5.0, 105.0], [1.0, 2.0, 5.0, 10.0, 20.0])
ns_result.duration
```

The Nelson–Siegel parameters stay fixed; only the layered zero-rate bumps move under AD.

## Interest-Sensitive Instruments

For instruments whose cashflows depend on the rate environment (callable bonds, floaters, etc.), use the do-block syntax to pass a custom valuation function:

```@example sensitivities
# Callable bond: key-rate durations (vector)
callable_krds = duration(KeyRates(tenors), zrc) do curve
    ncv = pv(curve, cfs, tenors)
    called_value = pv(curve, cfs[1:3], tenors[1:3]) + 102.0 * curve(3.0)
    min(ncv, called_value)
end
```

```@example sensitivities
# Scalar duration (default)
callable_dur = duration(zrc) do curve
    ncv = pv(curve, cfs, tenors)
    called_value = pv(curve, cfs[1:3], tenors[1:3]) + 102.0 * curve(3.0)
    min(ncv, called_value)
end
```

The function receives a curve object and must return a scalar value. ForwardDiff
differentiates the valuation as written, including branches such as `min`. At a kink
it returns the derivative of one branch.

## Floating-Rate Instruments: Effective vs Spread Duration

Pass a FinanceModels contract or portfolio directly. Contract measures differentiate
`present_value(Models(discount; index), contract)`. The bumped discount curve discounts
cashflows and values closed forms. Every key the contract reads, such as a floater's
`"SOFR"`, returns the bumped index curve. A closed form `present_value(ctx, c::MyContract)`
is used as written, also inside a `Composite` or a portfolio. The marker selects the risk:

- **Effective (rate) duration**: bump the curve; coupons re-fix. Small for a new floater
  (near zero at par), and about the time to the next reset once the current coupon is
  fixed (see [`locked_floater`](@ref)).
- **Spread (credit) duration**: bump the discount curve only; coupons stay fixed. Close
  to the duration of a fixed-rate bond with the same maturity.

```@example sensitivities
using FinanceModels: Bond
floater = Bond.Floating(0.015, Periodic(1), 5.0, "SOFR")   # SOFR + 150bp, 5y annual

(effective = duration(Effective(), zrc, floater),   # rate duration, yrs — small
 spread    = duration(Spread(),    zrc, floater),   # spread duration, yrs — near a 5-year bond's
 dv01      = duration(DV01(), Effective(), zrc, floater))   # effective DV01, $/bp
```

The curve comes before the contract. Calls without a marker default to `Effective()` for
`duration`, `duration(DV01(), …)` and `convexity`, including portfolios. Request spread risk explicitly with `Spread()`.
Coupons project on the keyword `index` curve, which defaults to the discount curve, and
every payment is discounted on the curve passed: `duration(Effective(), credit_curve,
floater; index = sofr_curve)`. The parallel measures take no tenor grid; use
`KeyRates(tenors)` for a key-rate decomposition.

```@example sensitivities
(duration(zrc, floater) ≈ duration(Effective(), zrc, floater),
 duration(DV01(), zrc, floater) ≈ duration(DV01(), Effective(), zrc, floater),
 convexity(zrc, floater) ≈ convexity(Effective(), zrc, floater))
```

`sensitivities` reports the two curve roles of `Models(discount; index)` separately, even when
`index` is the discount curve: `discount` shifts discounting only (spread risk), and `index`
shifts the projected coupons only. Effective risk is their sum:

```@example sensitivities
s = sensitivities(zrc, floater)
(spread = s.duration.discount, index = s.duration.index,
 effective = s.duration.discount + s.duration.index, eff_dv01 = s.dv01.discount + s.dv01.index)
```

With `SecondOrder()`, `s.convexity` holds the four blocks; effective convexity, which
`convexity(Effective(), ...)` returns, is the sum of all four, cross terms included.

For a fixed bond, effective and spread duration both equal its continuous-zero duration. For an **in-force** floater whose current coupon is already fixed, [`locked_floater`](@ref) gives the conventional rate duration ≈ time to next reset:

```@example sensitivities
duration(Effective(), zrc, locked_floater(floater, 0.04, 1.0))   # ≈ 1y, not ≈ 5
```

### Portfolios

For a portfolio, pass a vector of contracts. The calculation sums their values and dollar
derivatives before normalizing risk, so a portfolio's duration is the value-weighted average
of its contracts' durations:

```@example sensitivities
portfolio = [floater, Bond.Fixed(0.03, Periodic(1), 7.0)]
duration(zrc, portfolio)
```

A vector of `Cashflow`s is fixed cashflows rather than a portfolio; their measures agree.

### Multi-curve: risk-free + credit + ILP + index

Pass named discount layers and a coupon-projection `index` to obtain risk by role:

```@example sensitivities
rf     = zrc
credit = Yield.Constant(Continuous(0.01))
ilp    = Yield.Constant(Continuous(0.004))
r = sensitivities(KeyRates(tenors), (; rf, credit, ilp), floater; index = zrc)
r.dv01    # dollar risk per role: rf, credit and ilp move discounting only; index moves the coupons
```

Additional discount layers can represent liquidity or matching-adjustment spreads.
Use a callback with a FinanceModels valuation context for custom valuations:

```@example sensitivities
sensitivities(KeyRates(tenors), (; rf, credit, ilp, index = zrc)) do c
    present_value(Models(c.rf + c.credit + c.ilp; index = c.index), floater)
end
```

Use [`zspread`](@ref) to fit the discount margin to a market price. The margin is a
`Continuous` rate, so `zrc + z.zspread` is the discount curve that reprices; `zspread_dv01` is a
number:

```@example sensitivities
z = zspread(zrc, floater, 0.99)
```

```@example sensitivities
present_value(Models(zrc + z.zspread; index = zrc), floater)
```

## Portfolio Sensitivity

DV01s are additive across positions, so a portfolio's DV01 vector is ≈ the sum of the
positions' DV01 vectors on the same curve:

```@example sensitivities
# Two bonds: 5-year 5% coupon and 5-year 3% coupon
bond1_cfs   = [5.0, 5.0, 5.0, 5.0, 105.0]
bond1_times = [1.0, 2.0, 3.0, 4.0, 5.0]
bond2_cfs   = [3.0, 3.0, 3.0, 3.0, 103.0]
bond2_times = [1.0, 2.0, 3.0, 4.0, 5.0]

# Portfolio DV01 vector
portfolio_dv01 = duration(DV01(), KeyRates(tenors), zrc) do curve
    pv(curve, bond1_cfs, bond1_times) + pv(curve, bond2_cfs, bond2_times)
end

# Equivalently, sum the individual DV01 vectors:
dv01_1 = duration(DV01(), KeyRates(tenors), zrc, bond1_cfs, bond1_times)
dv01_2 = duration(DV01(), KeyRates(tenors), zrc, bond2_cfs, bond2_times)

(portfolio_dv01, dv01_1 .+ dv01_2, portfolio_dv01 ≈ dv01_1 .+ dv01_2)
```

### Example: Portfolio of Floating Rate Bonds

For a floater portfolio, differentiate both coupon amounts and discount factors:

```@example sensitivities
flt_rates  = [0.02, 0.025, 0.03, 0.035, 0.04, 0.042, 0.044, 0.046, 0.048, 0.05]
flt_tenors = collect(1.0:10.0)
flt_zrc    = ZeroRateCurve(flt_rates, flt_tenors)

# 10 floating rate bonds: maturities 1yr to 10yr, face 100 each,
# annual coupons = 1yr forward rate + 50bp credit spread
notionals  = fill(100.0, 10)
maturities = 1:10
flt_spread = 0.005

# The do-block receives the curve and returns the total present value
# of all cashflows across the portfolio.
floater_portfolio = sensitivities(KeyRates(flt_tenors), flt_zrc) do curve
    total = 0.0
    for (notional, mat) in zip(notionals, maturities)
        # For each bond, loop over annual payment dates t = 1, 2, ..., maturity
        for t in 1:mat
            df      = curve(Float64(t))
            df_prev = t == 1 ? 1.0 : curve(Float64(t - 1))

            # 1yr simple forward rate from t-1 to t: F = P(0,t-1)/P(0,t) - 1
            fwd = df_prev / df - 1.0

            # Floating coupon PV: notional × (forward rate + spread) × P(0,t)
            total += notional * (fwd + flt_spread) * df

            # Return principal at maturity
            t == mat && (total += notional * df)
        end
    end
    total
end

(value = floater_portfolio.value,
 total_duration = sum(floater_portfolio.duration))
```

A par floater at reset has near-zero effective duration. A fixed coupon spread
adds duration because those payments do not reset with the curve.

## Stochastic Model Sensitivities

ForwardDiff can differentiate the simulated valuation through Hull–White path
generation. These derivatives describe the Monte Carlo estimate, which remains
subject to sampling and time-discretization error.

A bare `HullWhite` model is a curve: a bump moves `hw.curve`, cashflows are discounted
on it, and caps and swaptions keep their Hull–White closed forms. Wrap it in
[`Scenarios`](@ref) to simulate.

### What is being differentiated?

The curve-risk API differentiates continuous-zero bumps at the supplied tenors.
For a stochastic valuation:

```julia
hw = ShortRate.HullWhite(0.1, 0.01, zrc)
sensitivities(KeyRates(tenors), Scenarios(hw; horizon = 6.0, n_scenarios = 500, rng = Xoshiro(42)), cfs, times)
```

the chain of differentiation is:

1. ForwardDiff perturbs the zero-rate bump at tenor `i`
2. The perturbed `curve` changes the forward curve `f(0, t)`
3. Hull–White recalibrates `θ(t)` from the new forwards
4. All simulated paths shift (same random draws, different drift)
5. The value derivative, divided by `-V`, gives KRD at tenor `i`

Mean reversion `a` and volatility `σ` stay fixed. These KRDs measure the
portfolio's response to curve shocks.

### Model parameter sensitivities (vega, mean-reversion sensitivity)

Parameter sensitivities measure changes in `a` or `σ`:

| | Curve KRDs (`-(∂V/∂rᵢ)/V`) | Parameter sensitivities (`∂V/∂a`, `∂V/∂σ`) |
|---|---|---|
| **What moves** | Market zero rates | Model calibration parameters |
| **Use case** | Hedging with bonds/swaps | Model risk, calibration stability |

The curve-risk methods do not calculate parameter sensitivities. One way to
estimate them is finite differences with matched random draws:

```@example sensitivities
using FinanceModels: ShortRate, simulate
using FinanceCore: discount
using Random: Xoshiro

mc_rates  = [0.03, 0.03, 0.03, 0.03, 0.03]
mc_tenors = [1.0, 2.0, 3.0, 4.0, 5.0]
mc_cfs    = [5.0, 5.0, 5.0, 5.0, 105.0]

function mc_value(a, σ)
    curve     = ZeroRateCurve(mc_rates, mc_tenors)
    hw        = ShortRate.HullWhite(a, σ, curve)
    scenarios = simulate(hw; n_scenarios = 1000, timestep = 1/12, horizon = 6.0, rng = Xoshiro(42))
    sum(pv(sc, mc_cfs, mc_tenors) for sc in scenarios) / 1000
end

# Finite-difference sensitivities
ε       = 1e-5
dV_da   = (mc_value(0.1 + ε, 0.01) - mc_value(0.1 - ε, 0.01)) / (2ε)   # mean reversion
dV_dσ   = (mc_value(0.1, 0.01 + ε) - mc_value(0.1, 0.01 - ε)) / (2ε)   # volatility (vega)

(dV_da, dV_dσ)
```

### Hull–White: sensitivities w.r.t. the initial term structure

A Hull–White model calibrates its drift θ(t) to the initial yield curve. Calculate
the simulated value's sensitivity to curve bumps:

```@example sensitivities
# Key-rate sensitivities of E[V] under Hull–White dynamics
hw_curve  = ZeroRateCurve(mc_rates, mc_tenors)
hw        = ShortRate.HullWhite(0.1, 0.01, hw_curve)
scenarios = Scenarios(hw; n_scenarios = 500, timestep = 1/12, horizon = 6.0, rng = Xoshiro(42))
hw_result = sensitivities(KeyRates(mc_tenors), scenarios, mc_cfs, mc_tenors)

(durations = hw_result.duration,
 sum_durations = sum(hw_result.duration))
```

This uses nested AD: curve-risk derivatives pass through the forward-rate
derivatives used to calibrate Hull–White drift. ForwardDiff's
[tag system](https://github.com/JuliaDiff/ForwardDiff.jl/issues/83) separates them.

For a valuation that is not a fixed cashflow stream, pass a callback; it receives the
vector of simulated paths:

```@example sensitivities
sensitivities(KeyRates(mc_tenors), scenarios) do paths
    sum(pv(p, mc_cfs, mc_tenors) for p in paths) / length(paths)
end
```

### Comparison: deterministic vs model-based sensitivities

Compare simulated sensitivities with direct discounting for fixed cashflows:

```@example sensitivities
# Deterministic: discount directly off the initial curve (or `hw` itself)
det_result = sensitivities(KeyRates(mc_tenors), hw_curve, mc_cfs, mc_tenors)

# Model-based: average across simulated rate paths (computed above as hw_result)
(det_durations  = det_result.duration,
 hw_durations   = hw_result.duration,
 sum_det        = sum(det_result.duration),
 sum_hw         = sum(hw_result.duration))
```

For fixed cashflows, exact risk-neutral valuation gives
`V = Σ cf_i × P(0, t_i)` ([Glasserman, 2003, Ch. 7](https://link.springer.com/book/10.1007/978-0-387-21617-1)).
A model calibrated to each shocked curve should therefore reproduce both total
and bucket risk under those same shocks. Sampling, time discretization, and the
implementation of curve shocks can cause differences in numerical estimates.
Agreement in total alone does not validate the individual buckets.

Pathwise AD differentiates the simulated estimate through drift calibration,
path generation, and valuation. See
[Giles & Glasserman (2006)](https://people.maths.ox.ac.uk/~gilesm/files/mc_greeks.pdf).

!!! note
    `Scenarios` fixes its horizon and time grid and draws one seed when it is constructed.
    Every valuation and AD evaluation reuses that seed, so the value and its derivatives
    share random draws, and results for several positions add. Do the same in your own
    simulation code. The horizon must be a whole number of time steps, and a payment after it
    throws.

## Choosing Interpolation

`ZeroRateCurve` accepts an optional third argument for the interpolation method:

```@example sensitivities
interp_rates  = [0.02, 0.03, 0.04, 0.05]
interp_tenors = [1.0, 3.0, 5.0, 10.0]

zrc_default = ZeroRateCurve(interp_rates, interp_tenors)                       # default: MonotoneConvex
zrc_pchip   = ZeroRateCurve(interp_rates, interp_tenors, Spline.PCHIP())       # PCHIP
zrc_lin     = ZeroRateCurve(interp_rates, interp_tenors, Spline.Linear())      # linear
zrc_cub     = ZeroRateCurve(interp_rates, interp_tenors, Spline.Cubic())       # cubic spline
zrc_aki     = ZeroRateCurve(interp_rates, interp_tenors, Spline.Akima())       # Akima

# All can be passed into the same sensitivities API:
interp_cfs = [3.0, 3.0, 3.0, 103.0]
(default_durs = sensitivities(KeyRates(interp_tenors), zrc_default, interp_cfs, interp_tenors).duration,
 linear_durs  = sensitivities(KeyRates(interp_tenors), zrc_lin,     interp_cfs, interp_tenors).duration)
```

Interpolation controls the base curve between its knots. The supported
choices include monotone convex, PCHIP, linear, Akima, and cubic interpolation.
See the [FinanceModels interpolation guide](https://docs.juliaactuary.org/FinanceModels/dev/interpolation/)
for their smoothness and shape constraints.

`KeyRates` applies the same triangular bumps for every interpolation method.
Changing the base interpolation can change discounted cashflow weights, but it
does not change the bump shape or tenor grid.

## Validating AD vs Bump-and-Reprice

Compare AD with central finite differences. Bumping a knot rate moves the zero rate
at that knot exactly as a `KeyRates` hat does, so with payments on the knots, as
here, the two agree to finite-difference error. The finite difference has O(ε²)
truncation error:

```@example sensitivities
val_rates  = [0.02, 0.03, 0.04, 0.05]
val_tenors = [1.0, 3.0, 5.0, 10.0]
val_zrc    = ZeroRateCurve(val_rates, val_tenors)
val_cfs    = [3.0, 3.0, 3.0, 103.0]

# AD (exact) — use KeyRates(tenors) for the per-tenor vector
ad_dv01 = duration(DV01(), KeyRates(val_tenors), val_zrc, val_cfs, val_tenors)

# Finite difference (bump-and-reprice)
ε = 1e-5
fd_dv01 = map(1:4) do i
    rates_up      = copy(val_rates); rates_up[i] += ε
    rates_dn      = copy(val_rates); rates_dn[i] -= ε
    v_up = pv(ZeroRateCurve(rates_up, val_tenors), val_cfs, val_tenors)
    v_dn = pv(ZeroRateCurve(rates_dn, val_tenors), val_cfs, val_tenors)
    -(v_up - v_dn) / (2ε) / 10_000
end

(; ad_dv01, fd_dv01, max_abs_diff = maximum(abs.(ad_dv01 .- fd_dv01)))
```

## Validating AD with TenorShift

Use `TenorShift` to compare the observed price change with the derivative's
prediction for a small finite shock:

```@example sensitivities
# AD: total DV01 across all tenors — already in dollar-per-1bp units
total_dv01 = sum(duration(DV01(), KeyRates(val_tenors), val_zrc, val_cfs, val_tenors))

# TenorShift: actual PV change under a +1 bp parallel shift
pv_base    = present_value(val_zrc, val_cfs, val_tenors)
shifted    = val_zrc + (z, t) -> z + Continuous(0.0001)  # +1 bp
pv_shifted = present_value(shifted, val_cfs, val_tenors)
actual_change = -(pv_shifted - pv_base)

# DV01 is per-1bp, so it directly predicts the 1bp PV change
predicted_change = total_dv01

(; predicted_change = round(predicted_change, digits = 6),
   actual_change    = round(actual_change,    digits = 6),
   ratio            = round(actual_change / predicted_change, digits = 6))
```

The ratio approaches one as the shock shrinks. DV01 is a first-order estimate;
finite price changes also include convexity and higher-order terms.
