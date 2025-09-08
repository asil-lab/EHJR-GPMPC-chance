module MPCController

using JuMP
using Ipopt
using LinearAlgebra
using GaussianProcesses
using ..GPModelTypes
using ..AgentTypes
using ..SystemDynamics
using ..ControllerTypes
using ..GPUtils

export AbstractMPCController, StandardMPC, GPBasedMPC, compute_control, compute_decentralized_control, compute_centralized_control

# Abstract type for all MPC controllers
abstract type AbstractMPCController end

# Standard MPC without GP
struct StandardMPC <: AbstractMPCController
    config::ControllerConfig
end

# MPC with GP-based dynamics learning
struct GPBasedMPC <: AbstractMPCController
    config::ControllerConfig
    use_linearized::Bool
end

"""
Add nominal (linear) dynamics constraints
"""
function add_nominal_dynamics!(model, x, u, A, B)
    horizon = size(x, 2)
    nx = size(x, 1)
    nu = size(B, 2)
    
    for t in 1:horizon-1
        for dim in 1:nx
            @constraint(model, x[dim, t+1] == 
                sum(A[dim, j] * x[j, t] for j in 1:nx) + 
                sum(B[dim, j] * u[j, t] for j in 1:nu))
        end
    end
end

"""
Add full GP dynamics constraints
"""
function add_full_gp_dynamics!(model, x, u, A, B, gp_models)
    horizon = size(x, 2)
    nx = size(x, 1)
    nu = size(B, 2)

    # Handle dimension 1
    if nx >= 1 && length(gp_models) >= 1 && !isnothing(gp_models[1])
        # Create prediction function for dimension 1
        gp1 = gp_models[1]
        function predict1(x1, x2)
            x_v = [x1 x2]'
            μ, σ² = predict_y(gp1, x_v)
            return μ[1]
        end
        
        # Register operator for dimension 1
        @operator(model, op_predict1, 2, predict1)
        
        # Add constraint for dimension 1
        for t in 1:horizon-1
            @constraint(model, x[1, t+1] == 
                sum(A[1, j] * x[j, t] for j in 1:nx) + 
                sum(B[1, j] * u[j, t] for j in 1:nu) + 
                op_predict1(x[1, t], x[2, t]))
        end
    else
        # Use linear model for dimension 1
        for t in 1:horizon-1
            @constraint(model, x[1, t+1] == 
                sum(A[1, j] * x[j, t] for j in 1:nx) + 
                sum(B[1, j] * u[j, t] for j in 1:nu))
        end
    end
    
    # Handle dimension 2 separately
    if nx >= 2 && length(gp_models) >= 2 && !isnothing(gp_models[2])
        # Create prediction function for dimension 2
        gp2 = gp_models[2]
        function predict2(x1, x2)
            x_v = [x1 x2]'
            μ, σ² = predict_y(gp2, x_v)
            return μ[1]
        end
        
        # Register operator for dimension 2
        @operator(model, op_predict2, 2, predict2)
        
        # Add constraint for dimension 2
        for t in 1:horizon-1
            @constraint(model, x[2, t+1] == 
                sum(A[2, j] * x[j, t] for j in 1:nx) + 
                sum(B[2, j] * u[j, t] for j in 1:nu) + 
                op_predict2(x[1, t], x[2, t]))
        end
    else
        # Use linear model for dimension 2
        for t in 1:horizon-1
            @constraint(model, x[2, t+1] == 
                sum(A[2, j] * x[j, t] for j in 1:nx) + 
                sum(B[2, j] * u[j, t] for j in 1:nu))
        end
    end
end

"""
Add linearized GP dynamics constraints
"""
function add_linearized_gp_dynamics!(model, x, u, A, B, gp_models, linearization_point)
    horizon = size(x, 2)
    nx = size(x, 1)
    nu = size(B, 2)
    
    # Handle dimension 1
    if nx >= 1 && length(gp_models) >= 1 && !isnothing(gp_models[1])
        μ1, grad1 = predict_y_and_gradient(gp_models[1], linearization_point)
        for t in 1:horizon-1
            @constraint(model, x[1, t+1] == 
                sum(A[1, j] * x[j, t] for j in 1:nx) + 
                sum(B[1, j] * u[j, t] for j in 1:nu) + 
                μ1[1] + sum(grad1[j] * (x[j, t] - linearization_point[j]) for j in 1:nx))
        end
    end
    
    # Handle dimension 2
    if nx >= 2 && length(gp_models) >= 2 && !isnothing(gp_models[2])
        μ2, grad2 = predict_y_and_gradient(gp_models[2], linearization_point)
        for t in 1:horizon-1
            @constraint(model, x[2, t+1] == 
                sum(A[2, j] * x[j, t] for j in 1:nx) + 
                sum(B[2, j] * u[j, t] for j in 1:nu) + 
                μ2[1] + sum(grad2[j] * (x[j, t] - linearization_point[j]) for j in 1:nx))
        end
    end
end

"""
Main dynamics constraint function that selects appropriate dynamics
"""
function add_dynamics_constraints!(model, x, u, A, B, dynamics_mode::DynamicsMode, gp_model=nothing, linearization_point=nothing, reference=Matrix{Float64}(undef,0,0))
    gp_models = isnothing(gp_model) ? Any[] : gp_model.models
    
    if !isempty(gp_models) && dynamics_mode == FullGP
        add_full_gp_dynamics!(model, x, u, A, B, gp_models)
    elseif dynamics_mode == LinearizedGP && !isnothing(gp_model) && !isnothing(linearization_point)
        add_linearized_gp_dynamics!(model, x, u, A, B, gp_models, linearization_point)
    else
        add_nominal_dynamics!(model, x, u, A, B)
    end
end

"""
Add collision avoidance constraints between agents.

# Arguments
- model: JuMP optimization model
- x: State trajectory variables for current agent (2×horizon Matrix for decentralized, 2×horizon×num_agents Array for centralized)
- other_trajectories: List of predicted trajectories for other agents
- safety_distance: Minimum allowed distance between agents
"""
function add_collision_avoidance!(model, x, other_trajectories, safety_distance=1.0)
    horizon = size(x, 2)
    # Handle both centralized (3D array) and decentralized (2D array) cases
    if ndims(x) == 3
        println("Using centralized collision avoidance")
        # Centralized case: x is 2×horizon×num_agents
        num_agents = size(x, 3)
        for t in 1:horizon
            for i in 1:num_agents
                for j in (i+1):num_agents
                    # Add quadratic distance constraint between each pair of agents
                    @constraint(model, 
                        (x[1,t,i] - x[1,t,j])^2 + 
                        (x[2,t,i] - x[2,t,j])^2 >= 
                        safety_distance^2
                    )
                end
            end
        end
    else
        println("Using decentralized collision avoidance")
        # Decentralized case: x is 2×horizon
        for t in 1:horizon
            for other_traj in other_trajectories
                if size(other_traj, 2) >= t  # Check if prediction exists for this timestep
                    @constraint(model,
                        (x[1,t] - other_traj[1,t])^2 + 
                        (x[2,t] - other_traj[2,t])^2 >= 
                        safety_distance^2
                    )
                end
            end
        end
    end
end

# Main control computation interface
function compute_control(
    controller::AbstractMPCController,
    state::Vector{Float64},
    reference::Matrix{Float64},
    gp_model::Union{Nothing, GPModel}=nothing;
    dynamics_mode::DynamicsMode=Nominal
)
    # Create optimization model with Ipopt
    model = Model(optimizer_with_attributes(
        Ipopt.Optimizer,
        "print_level" => 0,
        "max_iter" => 1000,
        "tol" => 1e-4
    ))

    # Extract parameters
    horizon = controller.config.mpc_params.horizon
    nx = length(state)
    nu = 2

    # Define variables
    @variable(model, x[1:nx, 1:horizon])
    @variable(model, u[1:nu, 1:horizon-1])

    # Initial condition
    @constraint(model, x[:, 1] .== state)

    # Dynamics constraints
    A, B = get_linear_system_matrices()
    add_dynamics_constraints!(model, x, u, A, B, dynamics_mode, gp_model, state, reference)

    # Objective function
    @objective(model, Min,
        sum(sum((x[:, t] - reference[:, min(t, size(reference,2))]).^2) for t in 1:horizon)
    )

    # Solve and handle results
    try
        JuMP.optimize!(model)
        status = termination_status(model)
        
        if status == MOI.OPTIMAL || status == MOI.LOCALLY_SOLVED
            return value.(u[:, 1])
        elseif has_values(model)
            @warn "Suboptimal solution found. Status: $status"
            return value.(u[:, 1])
        else
            @warn "Optimization failed. Status: $status. Using fallback control."
            return compute_fallback_control(state, reference[:, 1])
        end
    catch e
        @error "Optimization error: $e"
        return compute_fallback_control(state, reference[:, 1])
    end
end

# Fallback control computation
function compute_fallback_control(state::Vector{Float64}, reference::Vector{Float64})
    error = reference - state
    K_fallback = [0.5 0.0; 0.0 0.5]  # Simple proportional control
    return clamp.(K_fallback * error, -2.0, 2.0)
end

"""
Compute decentralized control for a single agent
"""
function compute_decentralized_control(
    controller::AbstractMPCController,
    agent::Agent,
    reference::Matrix{Float64},
    other_refs::Vector{Matrix{Float64}};  # Changed to accept reference trajectories
    dynamics_mode::DynamicsMode=Nominal
)
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    
    horizon = controller.config.mpc_params.horizon
    nx = length(agent.state)
    nu = 2

    # Variables and initial condition
    @variable(model, x[1:nx, 1:horizon])
    @variable(model, u[1:nu, 1:horizon-1])
    @constraint(model, x[:, 1] .== agent.state)

    # Add dynamics constraints based on mode
    A, B = get_linear_system_matrices()
    add_dynamics_constraints!(
        model, x, u, A, B,
        dynamics_mode,
        agent.gp_model,
        agent.state,
        reference
    )

    # Use reference trajectories for collision avoidance
    add_collision_avoidance!(model, x, other_refs)

    # Objective
    @objective(model, Min, sum(sum((x[:, t] - reference[:, t]).^2) for t in 1:horizon))

    JuMP.optimize!(model)
    return termination_status(model) in [MOI.OPTIMAL, MOI.LOCALLY_SOLVED] ? 
           value.(u[:, 1]) : 
           compute_fallback_control(agent.state, reference[:, 1])
end

"""
Compute centralized control for all agents
"""
function compute_centralized_control(
    controller::AbstractMPCController,
    agents::Vector{Agent},
    references::Vector{Matrix{Float64}};
    dynamics_mode::DynamicsMode=Nominal
)
    model = Model(Ipopt.Optimizer)
    set_silent(model)

    num_agents = length(agents)
    horizon = controller.config.mpc_params.horizon
    nx = 2  # State dimension per agent
    nu = 2  # Control dimension per agent

    # Create 3D arrays for states and controls
    @variable(model, x[1:nx, 1:horizon, 1:num_agents])
    @variable(model, u[1:nu, 1:horizon-1, 1:num_agents])
    
    # Initial conditions
    for i in 1:num_agents
        @constraint(model, x[:, 1, i] .== agents[i].state)
    end

    # Add dynamics constraints for each agent
    A, B = get_linear_system_matrices()
    for i in 1:num_agents
        add_dynamics_constraints!(
            model,
            view(x, :, :, i),  # Get 2D slice for current agent
            view(u, :, :, i),
            A, B,
            dynamics_mode,
            agents[i].gp_model,
            agents[i].state,
            references[i]
        )
    end

    # Add collision avoidance for all agents at once
    add_collision_avoidance!(model, x, [])  # Pass empty other_trajectories for centralized case

    # Combined objective
    @objective(model, Min,
        sum(sum(sum((x[:, t, i] - references[i][:, t]).^2) for t in 1:horizon)
            for i in 1:num_agents)
    )

    JuMP.optimize!(model)
    return termination_status(model) in [MOI.OPTIMAL, MOI.LOCALLY_SOLVED] ? 
           [value.(u[:, 1, i]) for i in 1:num_agents] : 
           [compute_fallback_control(agents[i].state, references[i][:, 1]) 
            for i in 1:num_agents]
end

end