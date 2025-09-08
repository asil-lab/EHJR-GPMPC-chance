module GPModelTypes

using GaussianProcesses
using LinearAlgebra
using Statistics

export GPModel

mutable struct GPModel
    models::Vector{GPE}  # Vector of standard GPs from GaussianProcesses.jl
    
    function GPModel(input_dim::Int, output_dim::Int)
        # Initialize empty vector of GPs
        models = Vector{GPE}(undef, output_dim)
        return new(models)
    end
end

end
