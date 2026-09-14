# Convexity Conventions

Convexity is the second derivative of value with respect to a rate shock, divided
by the initial value. Annual-yield and continuous-zero shocks produce different
convexities, even when they start from the same price.
See [Coleman (2011)](https://closemountain.com/papers/risktransform1_brief.pdf),
pp. 3–4, for the role of compounding in rate sensitivities.

Yield-model inputs use additive **continuously compounded zero-rate shifts**.
Plain scalar inputs use annual compounding; explicit `Rate` inputs use their
specified compounding. A curve's initial discount factors are preserved.

In v6.0, scalar convexity changes for every yield-model type. Both
`convexity(curve, cfs, times)` and `convexity(curve, valuation_function)` agree
with the tenor-aware form and the sum of the full key-rate matrix.

## Why the analytic formula contains t²

For fixed cashflows ``CF_i`` paid at fixed times ``t_i`` in years, write their
initial present values as ``PV_i = CF_i D(0,t_i)``. A parallel continuous-zero
shift ``s`` changes the price to

```math
P(s) = \sum_i PV_i e^{-s t_i}.
```

Differentiating twice gives

```math
P''(0) = \sum_i t_i^2 PV_i,
\qquad
C = \frac{P''(0)}{P(0)} = \sum_i w_i t_i^2,
\qquad
w_i = \frac{PV_i}{P(0)}.
```

Each derivative contributes a factor ``-t_i``. Payments at years 1, 2, and 3
therefore receive time weights 1, 4, and 9. The formula uses squared payment
times, weighted by present value, and applies to nonflat curves too.
[Nawalkha, Soto, and Beliaeva (2005)](https://catalogimages.wiley.com/images/db/pdf/0471427241.excerpt.pdf),
*Interest Rate Risk Modeling*, chapter 1, pp. 5–6, equation (1.2), gives this
continuous-rate definition.

The callback API derives convexity by differentiating the shocked valuation.
The cashflow API evaluates the formula directly. For rate-dependent cashflows,
use a callback or contract so the derivative includes changes in the payments.

## Worked example: 11.26, 8.40, and annual-yield convexity

Consider cashflows `[5, 5, 105]` at years `[1, 2, 3]`, discounted at a 4% annual
yield. Their present value is **102.775091**. The equivalent continuous zero
rate is ``\log(1.04) \approx 0.039220713``.

```jldoctest convexity_conventions
julia> using ActuaryUtilities, FinanceModels, FinanceCore

julia> cfs = [5.0, 5.0, 105.0]; times = [1.0, 2.0, 3.0];

julia> curve = Yield.Constant(Periodic(0.04, 1));

julia> valuation(c) = pv(c, cfs, times);

julia> kr = KeyRates(times);

julia> results = (convexity(curve, cfs, times),       # analytic fast path
                 convexity(curve, valuation),       # scalar AutoDiff
                 convexity(curve, times, cfs, times),
                 sum(convexity(kr, curve, cfs, times)),
                 sum(convexity(kr, valuation, curve)));

julia> round.(results; digits=6)
(8.400872, 8.400872, 8.400872, 8.400872, 8.400872)
```

For this example, with ``y=0.04`` and the same present-value weights:

| Quantity | Formula | Value |
|:--|:--|--:|
| Continuous-zero convexity | ``\sum_i w_i t_i^2`` | 8.400872 |
| Annual-yield convexity | ``\sum_i w_i t_i(t_i+1)/(1+y)^2`` | 10.412662 |
| Former constant-curve statistic | ``\sum_i w_i t_i(t_i+1)`` | 11.262335 |

The former statistic omits the ``(1+y)^2`` divisor required for annual-yield
convexity. [Clarke, de Silva, and Thorley (2013)](https://www.cfainstitute.org/sites/default/files/-/media/documents/book/rf-publication/2013/rf-v2013-n3-1-pdf.pdf),
*Fundamentals of Futures and Options*, appendix p. 127, equation (A.15) and the
following modification, distinguishes these annual-compounding quantities.
For annual-yield convexity, pass a scalar yield or explicit annual rate:

```jldoctest convexity_conventions
julia> round.((convexity(0.04, cfs, times),
               convexity(Periodic(0.04, 1), cfs, times)); digits=6)
(10.412662, 10.412662)

julia> weights = (cfs ./ 1.04 .^ times) ./ valuation(curve);

julia> round(sum(weights .* times .* (times .+ 1)); digits=6)
11.262335
```

## Summing key-rate convexities includes cross terms

The key-rate bumps interpolate linearly between knots, following the localized
spot-rate approach of [Ho (1992)](https://doi.org/10.3905/jfi.1992.408049),
“Key Rate Durations: Measures of Interest Rate Risks,” *Journal of Fixed Income*
2(2), pp. 29–44. The hats, including the flat endpoint extrapolations, sum to one.
An equal shift to every key rate therefore produces the parallel shock.

For ``K_{jk}=P^{-1}\partial^2 P/\partial s_j\partial s_k``, the chain rule gives

```math
C_{\mathrm{parallel}} = \mathbf{1}^{\mathsf T} K \mathbf{1}
                     = \sum_{j,k} K_{jk}.
```

Sum **every matrix entry**, including mixed derivatives. See
[Reitano (1991), “Multivariate Duration Analysis”](https://www.soa.org/globalassets/assets/library/monographs/50th-anniversary/investment-section/1999/january/m-as99-2-05.pdf),
section 3(c), definitions 3.3–3.5 and equation (3.28) (SOA monograph reprint).

With knots at years 1 and 3, the year-2 cashflow responds to both bumps, producing
nonzero off-diagonal entries:

```jldoctest convexity_conventions
julia> K = convexity(KeyRates([1.0, 3.0]), curve, cfs, times);

julia> (sum(K) ≈ convexity(curve, valuation),
        K[1, 2] > 0,
        K[1, 1] + K[2, 2] < sum(K))
(true, true, true)
```

The identity holds for derivatives under the specified hat shocks. Finite-bump
estimates have approximation error.
