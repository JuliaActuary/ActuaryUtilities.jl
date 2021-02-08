function _precompile_()
    precompile(irr,(Vector{Float64},UnitRange{Int64}))
    precompile(irr,(Vector{Float64},Vector{Int64}))
    precompile(irr,(Vector{Float64},Vector{Float64}))
    precompile(irr,(Vector{Int},Vector{Int}))
    
    precompile(present_value,(Float64,Vector{Float64}))
    precompile(present_value,(Float64,Vector{Float64},Vector{Float64}))
end