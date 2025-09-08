module SystemDynamics

using LinearAlgebra
using Distributions
using Random

export nominal_dynamics, true_dynamics, get_linear_system_matrices

"""
Nominal linear dynamics model for prediction
"""
function nominal_dynamics(x::AbstractVector, u::AbstractVector)
    A = [1.0 0.0; 0.0 1.0]  # Simple position dynamics
    B = [0.1 0.0; 0.0 0.1]  # Control effectiveness
    return A * x + B * u
end

"""
Nonlinear dynamics component
"""
function nonlinear_dynamics(x::AbstractVector)
    return [0.1 * sin(x[2]); 0.1 * cos(x[1])]  # Coupling terms
end

"""
True system dynamics with nonlinearities
"""
function true_dynamics(x::AbstractVector, u::AbstractVector; noise_scale::Float64=0.1)
    next_state = nominal_dynamics(x, u) .+ nonlinear_dynamics(x)
    
    if noise_scale > 0
        noise = noise_scale * randn(length(x))
        return next_state .+ noise
    end
    return next_state
end

"""
Get system matrices for MPC
"""
function get_linear_system_matrices()
    A = [1.0 0.0; 0.0 1.0]  # Position dynamics
    B = [0.1 0.0; 0.0 0.1]  # Control mapping
    return A, B
end

end  # module SystemDynamics