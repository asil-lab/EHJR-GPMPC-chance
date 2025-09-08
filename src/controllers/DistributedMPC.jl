module DistributedMPC

using JuMP
using Ipopt
using Juniper
using LinearAlgebra
using GaussianProcesses
using Statistics
using Distributions
using ..AgentTypes
using ..GPModelTypes
using ..SystemDynamics
using ..ControllerTypes  # Import the ControllerConfig and MPCParams types
using ..GPUtils

# Export types and functions
export DistributedMPCController, Edge
export solve_distributed_mpc!, solve_local_problem!
export initialize_edges!, update_edge_variables!
export update_consensus_variables!, update_dual_variables!



function add_nominal_dynamics!(model::Model, x::Matrix{VariableRef}, u::Matrix{VariableRef}, A::Matrix{Float64}, B::Matrix{Float64})
    horizon = size(x, 2)
    for t in 1:(horizon-1)
        @constraint(model, x[:, t+1] .== A * x[:, t] + B * u[:, t])
    end
end


function add_linearized_gp_dynamics!(
    model::Model, 
    x::Matrix{VariableRef}, 
    u::Matrix{VariableRef}, 
    A::Matrix{Float64}, 
    B::Matrix{Float64}, 
    gp_model::GPModel,
    x_prev::Matrix{Float64})
    
    horizon = size(x, 2)
    nx = size(x, 1)
    nu = size(B, 2)
    
    # Handle dimension 1 (x₁)
    if !isnothing(gp_model.models[1])
        for t in 1:horizon-1
            # First compute nominal dynamics
            x_nominal = sum(A[1, j] * x[j, t] for j in 1:nx) + 
                       sum(B[1, j] * u[j, t] for j in 1:nu)
            
            # Get GP prediction and gradient at linearization point
            μ1, grad1 = predict_y_and_gradient(gp_model.models[1], x_prev[:, t])
            
            # Add constraint with linearized GP correction
            @constraint(model, x[1, t+1] == x_nominal + μ1[1] + 
                sum(grad1[j] * (x[j, t] - x_prev[j, t]) for j in 1:2))
        end
    else
        add_nominal_dynamics!(model, x, u, A, B)
    end
    
    # Handle dimension 2 (x₂) similarly
    if !isnothing(gp_model.models[2])
        for t in 1:horizon-1
            x_nominal = sum(A[2, j] * x[j, t] for j in 1:nx) + 
                       sum(B[2, j] * u[j, t] for j in 1:nu)
            
            μ2, grad2 = predict_y_and_gradient(gp_model.models[2], x_prev[:, t])
            
            @constraint(model, x[2, t+1] == x_nominal + μ2[1] + 
                sum(grad2[j] * (x[j, t] - x_prev[j, t]) for j in 1:2))
        end
    else
        add_nominal_dynamics!(model, x, u, A, B)
    end
end

function add_full_gp_dynamics!(model, x, u, A, B, gp_models)
    horizon = size(x, 2)
    nx = size(x, 1)
    nu = size(B, 2)

    # Handle dimension 1
    if nx >= 1 && length(gp_models.models) >= 1 && !isnothing(gp_models.models[1])
        # Create prediction function for dimension 1
        gp1 = gp_models.models[1]
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
    if nx >= 2 && length(gp_models.models) >= 2 && !isnothing(gp_models.models[2])
        # Create prediction function for dimension 2
        gp2 = gp_models.models[2]
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

function add_nonlinear_dynamics!(model::Model, x::Matrix{VariableRef}, u::Matrix{VariableRef}, A, B)
    horizon = size(x, 2)
    for t in 1:(horizon-1)
        @constraint(model, x[1, t+1] == A[1,:]' * x[:, t] + B[1,:]' * u[:, t] + 0.1 * sin(x[2,t]))
        @constraint(model, x[2, t+1] == A[2,:]' * x[:, t] + B[2,:]' * u[:, t] + 0.1 * cos(x[1,t]))
    end
end

"""
Shift the last solution by one step and extend the final point
"""
function shift_and_extend_solution(x::Matrix{Float64}, u::Matrix{Float64})
    horizon = size(x, 2)
    
    # Shift state trajectory
    x_shifted = hcat(x[:, 2:end], x[:, end])  # Repeat last state
    
    # Shift control trajectory
    u_shifted = hcat(u[:, 2:end], u[:, end])  # Repeat last control
    
    return x_shifted, u_shifted
end



mutable struct Edge
    agent_i::Int
    agent_j::Int
    y_i::Matrix{Float64}  # Agent i's view of relative position (i - j)
    y_j::Matrix{Float64}  # Agent j's view of relative position (i - j)
    z::Matrix{Float64}    # Consensus relative position
    lambda_i::Matrix{Float64}  # Dual variable for agent i
    lambda_j::Matrix{Float64}  # Dual variable for agent j
    safety_distance::Float64
end

struct DistributedMPCController
    config::ControllerConfig
    dynamics_mode::DynamicsMode  # No change needed, kept as DynamicsMode_D
    rho::Float64
    max_iterations::Int
    tolerance::Float64
    edges::Dict{Tuple{Int,Int}, Edge}
    last_solutions::Dict{Int, Tuple{Matrix{Float64}, Matrix{Float64}}}
end

# Fix the constructor to use DynamicsMode_D
function DistributedMPCController(
    config::ControllerConfig, 
    dynamics_mode::DynamicsMode,  # Changed from DynamicsMode to DynamicsMode_D
    rho::Float64,
    max_iterations::Int,
    tolerance::Float64)
    
    return DistributedMPCController(
        config,
        dynamics_mode,
        rho,
        max_iterations,
        tolerance,
        Dict{Tuple{Int,Int}, Edge}(),
        Dict{Int, Tuple{Matrix{Float64}, Matrix{Float64}}}()
    )
end

function initialize_edges!(controller::DistributedMPCController, agents::Vector{Agent})
    horizon = controller.config.mpc_params.horizon
    
    # Create edges between all pairs of agents
    for i in 1:length(agents)
        for j in (i+1):length(agents)
            key = (i, j)
            
            agentipos = agents[i].state
            agentiinit = repeat(agentipos, horizon)
            agentjpos = agents[j].state
            agentjinit = repeat(agentjpos, horizon)
            zinit = [agentiinit agentjinit]'            
            
            # Create edge with better initialization
            controller.edges[key] = Edge(
                i,
                j,
                zeros(2, horizon*2),  # y_i - will be set in first ADMM iteration
                zeros(2, horizon*2),  # y_j - will be set in first ADMM iteration
                zinit, 
                zeros(2, horizon*2),  # lambda_i
                zeros(2, horizon*2),  # lambda_j
                controller.config.safety_distance
            )
        end
    end
end

function add_collision_constraints!(model,
                                    x,
                                    y,
                                    neighbor_edges,
                                    controller,
                                    agent,
                                    horizon::Int;
                                    use_chance_constraints::Bool=false,
                                    confidence_level::Float64=0.95)
    # println(fieldnames(typeof(agent)))
    for (n, (_edge_key, edge)) in enumerate(neighbor_edges)
        for t in 1:horizon
            # relative direction between the two consensus trajectories
            rel   = edge.z[:, t] .- edge.z[:, horizon + t]
            dist  = norm(rel)
            grad  = dist < 1e-9 ? [1.0, 0.0] : rel ./ dist
            if use_chance_constraints && !isnothing(agent.gp_model)
                if edge.agent_i == agent.id
                    # agent i's state vs. agent j's planned path
                    x_prev = edge.y_i[:, t]
                else
                    # agent j's state vs. agent i's planned path
                    x_prev = edge.y_j[:, horizon + t]
                end
                # Chance constraints using GP predictions
                μ1, σ1, grad_σ1 = get_prediction_with_grad_sigma(agent.gp_model.models[1], x_prev)
                μ2, σ2, grad_σ2 = get_prediction_with_grad_sigma(agent.gp_model.models[2], x_prev)
                
                # Safe distance with linearized uncertainty
                z_score = quantile(Normal(), confidence_level)
                safe_distance = controller.config.safety_distance + 
                                z_score * sqrt(grad'*(σ1^2 + σ2^2)*grad)
                # println("safe_distance: ", safe_distance)
            else
                safe_distance = controller.config.safety_distance
            end
            # println("safe_distance: ", safe_distance)
            if edge.agent_i == agent.id
                # agent i's state vs. agent j's planned path
                rel  = edge.y_i[:, t]    .- edge.y_i[:, horizon + t]
                dist = norm(rel)
                grad = dist < 1e-9 ? [1.0, 0.0] : rel ./ dist

                @constraint(model, y[n, :, t] .== x[:, t])
                @constraint(model,
                    dist + dot(grad, x[:, t] .- y[n, :, horizon + t] .- rel)
                    ≥ safe_distance)
            else
                # agent j vs. agent i's path
                rel  = edge.y_j[:, t]    .- edge.y_j[:, horizon + t]
                dist = norm(rel)
                grad = dist < 1e-9 ? [1.0, 0.0] : rel ./ dist

                @constraint(model, y[n, :, horizon + t] .== x[:, t])
                @constraint(model,
                    dist + dot(grad, y[n, :, t] .- x[:, t] .- rel)
                    ≥ safe_distance)
            end
        end
    end
    return model
end

function add_dynamics_constraints!(
    model::Model,
    controller::DistributedMPCController,
    x::Matrix{VariableRef},
    u::Matrix{VariableRef},
    agent::Agent,
    x_prev::Matrix{Float64})

    A, B = get_linear_system_matrices()

    if controller.dynamics_mode == FullGP && !isnothing(agent.gp_model)
        # Full GP dynamics using operator registration
        add_full_gp_dynamics!(model, x, u, A, B, agent.gp_model)
    
    elseif controller.dynamics_mode == LinearizedGP && !isnothing(agent.gp_model)
        # Linearized GP dynamics
        add_linearized_gp_dynamics!(model, x, u, A, B, agent.gp_model, x_prev)
    
    elseif controller.dynamics_mode == Nominal
        # Nominal linear dynamics
        add_nominal_dynamics!(model, x, u, A, B)
    
    elseif controller.dynamics_mode == Nonlinear
        # Known nonlinear dynamics
        add_nonlinear_dynamics!(model, x, u, A, B)
    end
end

function solve_local_problem!(
    controller::DistributedMPCController,
    agent::Agent,
    reference::Matrix{Float64},
    x_previous::Matrix{Float64})

    model = Model(optimizer_with_attributes(Ipopt.Optimizer, 
        "print_level" => 0
    ))

    if x_previous === nothing
        x_previous = reference
    elseif any(isnan, x_previous)
        x_previous .= reference
    end
    
    
    horizon = controller.config.mpc_params.horizon
    nx = 2
    nu = 2
    # find number of neighbors
    neighbor_edges = [
        (k, controller.edges[k]) for k in sort(collect(keys(controller.edges)))
        if controller.edges[k].agent_i == agent.id ||
           controller.edges[k].agent_j == agent.id
    ]
    num_neighbors = length(neighbor_edges)
    # Variables
    @variable(model, x[1:nx, 1:horizon])
    @variable(model, u[1:nu, 1:horizon-1])
    @variable(model, y[1:num_neighbors,1:2, 1:horizon*2]) 
    # Initial condition and input constraints
    @constraint(model, x[:, 1] .== agent.state)
    u_min, u_max = controller.config.mpc_params.control_bounds
    # @constraint(model, [t=1:horizon-1], u_min .<= u[:, t] .<= u_max)
    A, B = get_linear_system_matrices()
    # Linearized collision avoidance constraints
    add_collision_constraints!(
        model, x, y, neighbor_edges, controller, agent, horizon,
        use_chance_constraints=controller.config.use_chance_constraints,  # Use setting from controller
        confidence_level=controller.config.confidence_level
    )


    add_dynamics_constraints!(model, controller, x, u, agent, x_previous)
    # Objective function (without slack penalty)
    tracking_obj = sum(sum((x[:, t] - reference[:, t]).^2) for t in 1:horizon)
    
    
    # ADMM terms for relative positions
    coupling_obj = 0.0

    for (n, (edge_key, edge)) in enumerate(neighbor_edges)
        if edge.agent_i == agent.id
            for t in 1:horizon*2
                rel_traj = y[n, :, t] - edge.z[:, t]
                coupling_obj += dot(edge.lambda_i[:, t], rel_traj) +
                               (controller.rho/2) * sum(rel_traj.^2)
            end
        else  # edge.agent_j == agent.id
           for t in 1:horizon*2
                rel_traj = y[n,:, t] - edge.z[:, t]
                coupling_obj += dot(edge.lambda_j[:, t], rel_traj) +
                               (controller.rho/2) * sum(rel_traj.^2)
            end
        end
    end

    obj_u = 0.001*sum(sum((u[:, t]).^2) for t in 2:horizon-1) 
    total_effort_bound = 10.0  # Adjust this value based on your system
    # @constraint(model, sum(sum(u[:, t].^2) for t in 1:horizon-1) <= total_effort_bound)
    # @constraint(model, [t=1:horizon-1], sum(u[:, t]).<= u_max)
    for t in 1:horizon-1
        # @constraint(model, u[:,t] .<= u_max)
        @constraint(model, u[:,t]'* u[:,t] <= u_max^2)
    end
    @objective(model, Min, tracking_obj + coupling_obj + obj_u)
    # @objective(model, Min, tracking_obj)
    JuMP.optimize!(model)
    
    # More detailed error handling
    status = termination_status(model)
    if status ∉ [MOI.OPTIMAL, MOI.LOCALLY_SOLVED]
        @warn "Local problem for agent $(agent.id) failed to solve optimally (status: $status)"
        if has_values(model)
            return value.(x), value.(u)  # Return best solution found
        else
            return zeros(nx, horizon), zeros(nu, horizon-1)
        end
    end
    y_val = value.(y)
    x_val = value.(x)
    u_val = value.(u)
    for (n,(key,edge)) in enumerate(neighbor_edges)
        if edge.agent_i==agent.id
            edge.y_i = y_val[n,:,:]
        else
            edge.y_j = y_val[n,:,:]
        end
    end
    return x_val, u_val
end


function update_consensus_variables!(controller::DistributedMPCController)
    all_edges = [(k, controller.edges[k]) for k in sort(collect(keys(controller.edges)))]
    for (n, (edge_key, edge)) in enumerate(all_edges)
        # # collision avoidance for z_old
        # model = Model(optimizer_with_attributes(Ipopt.Optimizer, 
        # "print_level" => 0))  
        # # println("edge: ", edge)
            
        # lengthz = size(edge.z, 2)
        # H = controller.config.mpc_params.horizon
        # @variable(model, z[1:2, 1:lengthz])
        # objectivez = 0.0
        # for t in 1:H*2
        #     rel_traj_i = edge.y_i[:,t] - z[:,t]
        #     rel_traj_j = edge.y_j[:,t] - z[:,t]
        #     objectivez += dot(edge.lambda_i[:, t], rel_traj_i) +
        #                        (controller.rho/2) * sum(rel_traj_i.^2)
        #     objectivez += dot(edge.lambda_j[:, t], rel_traj_j) +
        #                          (controller.rho/2) * sum(rel_traj_j.^2)
        # end
            
        # @objective(model, Min, objectivez)
        # for t in 1:H
        #     dist = norm(edge.z[:,t] - edge.z[:,t+H])
        #     grad = dist < 1e-9 ? [1.0,0.0] : (edge.z[:,t] - edge.z[:,t+H]) ./ dist
        #     @constraint(model, dist + dot(grad, z[:,t] - z[:,t+H]-edge.z[:,t] - edge.z[:,t+H]) ≥ controller.config.safety_distance)
        # end
        # # Solve the optimization problem
        # JuMP.optimize!(model)
        # # set z
        # edge.z = value.(z)

        edge.z = (edge.y_i + edge.y_j) / 2
    end
end

function update_dual_variables!(controller::DistributedMPCController)
    # Initialize error metrics
    consensus_errors = Dict(
        "primal_i" => Float64[],  # y_i - z errors
        "primal_j" => Float64[],  # y_j - z errors
        "dual" => Float64[]       # z - z_old errors
    )
    
    for edge in values(controller.edges)
        # Primal residuals (consensus constraint violations)
        primal_i = edge.y_i - edge.z
        primal_j = edge.y_j - edge.z
        
        # Store current z for dual residual
        z_old = copy(edge.z)
        
        # Update dual variables
        edge.lambda_i += controller.rho * primal_i
        edge.lambda_j += controller.rho * primal_j
        
        # Store errors
        push!(consensus_errors["primal_i"], norm(primal_i))
        push!(consensus_errors["primal_j"], norm(primal_j))
        push!(consensus_errors["dual"], norm(edge.z - z_old))
    end
    
    # Compute maximum residuals for ADMM convergence check
    primal_residual = mean([consensus_errors["primal_i"]; consensus_errors["primal_j"]])
    dual_residual = mean(consensus_errors["dual"])
    
    return primal_residual, dual_residual, consensus_errors
end

function solve_distributed_mpc!(
    controller::DistributedMPCController,
    agents::Vector{Agent},
    references::Vector{Matrix{Float64}},
    previous_path::Vector{Matrix{Float64}}=nothing; 
    verbose::Bool=false)
    # Initialize edges if not already done
    initialize_edges!(controller, agents)
    
    # Tracking convergence history
    history = Dict(
        "primal_residuals" => Float64[],
        "dual_residuals" => Float64[],
        "iterations" => Int[],
        "consensus_errors" => Vector{Dict{String,Vector{Float64}}}()  # Store errors for each iteration
    )

    for iter in 1:controller.max_iterations
        # Step 1: Local optimization for each agent
        local_solutions = Dict{Int, Matrix{Float64}}()

        for agent in agents
            x, u_local = solve_local_problem!(controller, agent, references[agent.id], previous_path[agent.id])
            previous_path[agent.id] = x
            local_solutions[agent.id] = u_local
        end

        
        # Step 3: Update consensus variables z
        update_consensus_variables!(controller)
        
        # Step 4: Update dual variables lambda and get detailed errors
        primal_res, dual_res, consensus_errors = update_dual_variables!(controller)
        
        # Store convergence history
        push!(history["primal_residuals"], primal_res)
        push!(history["dual_residuals"], dual_res)
        push!(history["iterations"], iter)
        push!(history["consensus_errors"], consensus_errors)
        
        if verbose
            @info "Iteration $iter: primal_res = $primal_res, dual_res = $dual_res"
        end

        # Check both primal and dual convergence
        if primal_res < controller.tolerance && dual_res < controller.tolerance
            println("ADMM converged after $iter iterations")
            if verbose
                @info "ADMM converged after $iter iterations"
            end
            break
        end
    end
    
    # Extract controls and return with convergence history
    controls = extract_controls(controller, agents, references, previous_path)  # Now returns 2×num_agents matrix
    return controls, history, previous_path  # Return both controls and history
end

function extract_controls(
    controller::DistributedMPCController, 
    agents::Vector{Agent},
    references::Vector{Matrix{Float64}},
    previous::Vector{Matrix{Float64}})
    # Initialize control matrix (2×num_agents)
    num_agents = length(agents)
    controls = zeros(2, num_agents)
    
    # Use the final ADMM iteration results
    for (i, agent) in enumerate(agents)
        _, u = solve_local_problem!(controller, agent, references[agent.id], previous[agent.id])
        controls[:, i] = u[:, 1]  # Store first control input as column
    end
    
    return controls  # Return 2×num_agents matrix
end

function solve_local_problem_chance_constrained!(
    controller::DistributedMPCController,
    agent::Agent,
    reference::Matrix{Float64},
    x_previous::Matrix{Float64})
    model = Model(optimizer_with_attributes(Ipopt.Optimizer, 
        "print_level" => 0
    ))
    if x_previous === nothing
        x_previous = reference
    elseif any(isnan, x_previous)
        x_previous .= reference
    end
    
    horizon = controller.config.mpc_params.horizon
    nx = 2
    nu = 2
    
    # Find number of neighbors
    neighbor_edges = [
        (k, controller.edges[k]) for k in sort(collect(keys(controller.edges)))
        if controller.edges[k].agent_i == agent.id ||
           controller.edges[k].agent_j == agent.id
    ]
    num_neighbors = length(neighbor_edges)
    
    # Variables
    @variable(model, x[1:nx, 1:horizon])
    @variable(model, u[1:nu, 1:horizon-1])
    @variable(model, y[1:num_neighbors, 1:2, 1:horizon*2])
    
    # Initial condition
    @constraint(model, x[:, 1] .== agent.state)
    
    # Input constraints
    u_min, u_max = controller.config.mpc_params.control_bounds
    @constraint(model, [t=1:horizon-1], u_min .<= u[:, t] .<= u_max)
    
    # Get system matrices
    A, B = get_linear_system_matrices()
    
    # Add GP-based chance constraints for collision avoidance
    confidence_level = 0.95  # 95% confidence level
    for (n, (edge_key, edge)) in enumerate(neighbor_edges)
        for t in 1:horizon
            if edge.agent_i == agent.id
                # Agent i's state vs agent j's planned path
                @constraint(model, y[n,:,t] .== x[:,t])
                
                # Get GP predictions for agent j's position
                if !isnothing(agent.gp_model.models[1]) && !isnothing(agent.gp_model.models[2])
                    # Predict mean and variance for both dimensions
                    μ1, σ1 = predict_y_and_std(agent.gp_model.models[1], y[n,:,horizon+t])
                    μ2, σ2 = predict_y_and_std(agent.gp_model.models[2], y[n,:,horizon+t])
                    
                    # Calculate the minimum safe distance considering uncertainty
                    # Using normal distribution properties for 95% confidence
                    z_score = quantile(Normal(), confidence_level)
                    safe_distance = controller.config.safety_distance + 
                                  z_score * sqrt(σ1[1]^2 + σ2[1]^2)
                    
                    # Add chance constraint
                    rel = x[:,t] - [μ1[1], μ2[1]]
                    dist = norm(rel)
                    grad = dist < 1e-9 ? [1.0,0.0] : rel ./ dist
                    @constraint(model, dot(grad, x[:,t] - [μ1[1], μ2[1]]) ≥ safe_distance)
                end
            else
                # Agent j vs. agent i's path
                @constraint(model, y[n,:,horizon+t] .== x[:,t])
                
                # Get GP predictions for agent i's position
                if !isnothing(agent.gp_model.models[1]) && !isnothing(agent.gp_model.models[2])
                    # Predict mean and variance for both dimensions
                    μ1, σ1 = predict_y_and_std(agent.gp_model.models[1], y[n,:,t])
                    μ2, σ2 = predict_y_and_std(agent.gp_model.models[2], y[n,:,t])
                    
                    # Calculate the minimum safe distance considering uncertainty
                    z_score = quantile(Normal(), confidence_level)
                    safe_distance = controller.config.safety_distance + 
                                  z_score * sqrt(σ1[1]^2 + σ2[1]^2)
                    
                    # Add chance constraint
                    rel = [μ1[1], μ2[1]] - x[:,t]
                    dist = norm(rel)
                    grad = dist < 1e-9 ? [1.0,0.0] : rel ./ dist
                    @constraint(model, dot(grad, [μ1[1], μ2[1]] - x[:,t]) ≥ safe_distance)
                end
            end
        end
    end
    
    # Dynamics
    if controller.use_gp
        linearization_point = [agent.state; zeros(nu)]
        add_linearized_gp_dynamics!(model, x, u, A, B, agent.gp_model, linearization_point)
    else
        add_nominal_dynamics!(model, x, u, A, B)
    end
    
    # Objective function
    tracking_obj = sum(sum((x[:, t] - reference[:, t]).^2) for t in 1:horizon)
    
    # ADMM terms for relative positions
    coupling_obj = 0.0
    for (n, (edge_key, edge)) in enumerate(neighbor_edges)
        if edge.agent_i == agent.id
            for t in 1:horizon*2
                rel_traj = y[n, :, t] - edge.z[:, t]
                coupling_obj += dot(edge.lambda_i[:, t], rel_traj) +
                               (controller.rho/2) * sum(rel_traj.^2)
            end
        else
            for t in 1:horizon*2
                rel_traj = y[n,:, t] - edge.z[:, t]
                coupling_obj += dot(edge.lambda_j[:, t], rel_traj) +
                               (controller.rho/2) * sum(rel_traj.^2)
            end
        end
    end
    
    @objective(model, Min, tracking_obj + coupling_obj)
    JuMP.optimize!(model)
    
    # Error handling
    status = termination_status(model)
    if status ∉ [MOI.OPTIMAL, MOI.LOCALLY_SOLVED]
        @warn "Local problem for agent $(agent.id) failed to solve optimally (status: $status)"
        if has_values(model)
            return value.(x), value.(u)
        else
            return zeros(nx, horizon), zeros(nu, horizon-1)
        end
    end
    
    y_val = value.(y)
    x_val = value.(x)
    u_val = value.(u)
    
    # Update edge variables
    for (n,(key,edge)) in enumerate(neighbor_edges)
        if edge.agent_i==agent.id
            edge.y_i = y_val[n,:,:]
        else
            edge.y_j = y_val[n,:,:]
        end
    end
    
    return x_val, u_val
end

end  # module

