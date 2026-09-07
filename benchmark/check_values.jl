using Serialization

equivalent(a::Number, b::Number) = isapprox(a, b; rtol = 1.0e-10, atol = 1.0e-10)
equivalent(a::AbstractArray, b::AbstractArray) = axes(a) == axes(b) && all(equivalent.(a, b))
equivalent(a::NamedTuple, b::NamedTuple) = keys(a) == keys(b) && all(equivalent.(values(a), values(b)))

baseline, candidate = deserialize.(ARGS)
@assert keys(baseline) == keys(candidate)
for name in sort!(collect(keys(baseline)))
    @assert equivalent(baseline[name], candidate[name]) "Different benchmark output: $name"
end
println("All $(length(baseline)) benchmark outputs agree (rtol = atol = 1e-10).")
