module AtomicMeasures

import ..Distributions

# Shared sorted, normalized atoms for exact risk and transport calculations.
struct FiniteAtoms{X, P}
    values::X
    probabilities::P
    function FiniteAtoms(xs, ps)
        length(xs) == length(ps) || throw(DimensionMismatch("atoms and probabilities must have equal length"))
        all(p -> isfinite(p) && p >= 0, ps) || throw(ArgumentError("atom probabilities must be finite and nonnegative"))
        total = sum(ps)
        (isfinite(total) && total > 0) || throw(ArgumentError("atom probabilities must have a positive finite sum, got $total"))
        order = sortperm(xs)
        filter!(i -> !iszero(ps[i]), order)
        values = xs[order]
        probabilities = ps[order] ./ total
        return new{typeof(values), typeof(probabilities)}(values, probabilities)
    end
end

finite_atoms(d) = nothing
function finite_atoms(d::Distributions.DiscreteUnivariateDistribution)
    # Atom count can be finite even when the support contains infinity.
    (d isa Union{Distributions.DiscreteNonParametric, Distributions.Dirac} || Distributions.hasfinitesupport(d)) || return nothing
    xs = collect(Distributions.support(d))
    return FiniteAtoms(xs, Distributions.pdf.(d, xs))
end
function finite_atoms(xs::AbstractVector{<:Real})
    isempty(xs) && throw(ArgumentError("an empirical measure needs at least one observation"))
    # Rational ranks preserve exact i/n breakpoints, even for unequal samples.
    return FiniteAtoms(collect(xs), fill(1 // length(xs), length(xs)))
end

function probability_breaks(atoms::FiniteAtoms)
    breaks = cumsum(atoms.probabilities)
    clamp!(breaks, zero(eltype(breaks)), one(eltype(breaks)))
    breaks[end] = one(eltype(breaks))
    return breaks
end

end
