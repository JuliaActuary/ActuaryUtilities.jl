
# created with the help of SnoopCompile.jl
function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(internal_rate_of_return),Vector{Int64}})   
    Base.precompile(Tuple{typeof(internal_rate_of_return),Vector{Float64}})
    Base.precompile(Tuple{typeof(internal_rate_of_return),Vector{Int64},Vector{Float64}})
    Base.precompile(Tuple{typeof(internal_rate_of_return),Vector{Float64},Vector{Float64}})   


    Base.precompile(Tuple{typeof(present_value),Float64,Vector{Int64},Vector{Float64}}) 
    Base.precompile(Tuple{typeof(present_value),Float64,Vector{Float64},Vector{Float64}}) 
    Base.precompile(Tuple{typeof(present_value),Float64,Vector{Float64},Vector{Int64}}) 

    Base.precompile(Tuple{typeof(duration),Float64,Vector{Float64}})
    Base.precompile(Tuple{typeof(duration),Float64,Vector{Int64}})

    Base.precompile(Tuple{typeof(convexity),Float64,Vector{Float64}})
    Base.precompile(Tuple{typeof(convexity),Float64,Vector{Int64}})
    
end