using Plots
using Random
using Statistics
using GaussianProcesses: predict_f
using LinearAlgebra
using JLD2
using Dates
include("../src/CGPMPC_src.jl")
using .CGPMPC_src

function shift_solution(paths)
    # Shift the solution to the next time step
    for agent_path in paths
        agent_path[1:end-1, :] = agent_path[2:end, :]
    end
    
    return paths
end

function create_trajectory_animation(state_histories, ref_trajs, num_steps)
    # Create the animation
    anim = @animate for t in 1:num_steps
        p = plot(
            title="Agent Trajectories (t = $t)",
            xlabel="x₁",
            ylabel="x₂",
            aspect_ratio=:equal,
            legend=:outerright,
            xlims=(-3, 3),  # Adjust these limits based on your scenario
            ylims=(-3, 3)   # Adjust these limits based on your scenario
        )
        
        colors = [:blue, :red, :green]
        
        # Plot reference trajectories
        for i in 1:length(ref_trajs)
            plot!(p, ref_trajs[i][1,:], ref_trajs[i][2,:],
                label="Reference $i",
                color=colors[i],
                linestyle=:dash,
                alpha=0.5
            )
        end
        
        # Plot actual trajectories up to current time
        for i in 1:length(state_histories)
            # Plot the path up to current time
            plot!(p, state_histories[i][1,1:t], state_histories[i][2,1:t],
                label="Agent $i Path",
                color=colors[i],
                alpha=0.3
            )
            
            # Plot current position with a marker
            scatter!(p, [state_histories[i][1,t]], [state_histories[i][2,t]],
                label="Agent $i",
                color=colors[i],
                marker=:circle,
                markersize=8
            )
        end
    end
    
    # Save the animation
    gif(anim, "agent_trajectories.gif", fps=5)
end

function run_distributed_comparison()
    # Configuration
    num_agents = 3
    num_steps = 30  # Total simulation steps
    horizon = 10    # MPC prediction horizon
    
    # Create crossing trajectories that end at desired positions
    ref_trajs = generate_crossing_trajectories(num_steps, num_agents, radius=2.0, spacing=1.0)
    
    # repeat the endpoint of the trajectories as the full trajectory
    for i in 1:num_agents
        endpoint = ref_trajs[i][:, end]
        # Create a full trajectory with the same endpoint
        for t in 2:num_steps
            ref_trajs[i][:, t] = endpoint
        end
    end

    # println("Reference Trajectories:")
    # for i in 1:num_agents
    #     println("Agent $i: ", ref_trajs[i])
    # end
    # Create agents
    agents = [
        Agent(
            AgentConfig(i, horizon, 0.4, (-2.0, 2.0), (-100.0, 100.0)),
            zeros(2)
        ) for i in 1:num_agents
    ]
    
    # Initialize agents at their starting positions
    for (agent, ref_traj) in zip(agents, ref_trajs)
        reset_state!(agent, ref_traj[:, 1])
    end
    
    
    # Train GPs for all agents
    for (agent, ref_traj) in zip(agents, ref_trajs)
        collect_training_data!(agent, ref_traj, 100)
        train_gp!(agent)
    end
    
    # Create distributed MPC controller with LINEAR_GP mode
    controller = DistributedMPCController(
        ControllerConfig(
            MPCParams(horizon, (-2.0, 2.0), (-100.0, 100.0), [1.0, 1.0], Dict("print_level" => 0)),
            true, 0.5, 0.5  # Set safety_distance to 0.0
        ),
        LinearizedGP,  # Use LINEAR_GP mode
        20.0,    # rho
        200,   # max_iterations
        1e-9    # tolerance
    )
    
    # Storage for results
    state_histories = [zeros(2, num_steps) for _ in 1:num_agents]
    control_histories = [zeros(2, num_steps) for _ in 1:num_agents]
    mse_histories = [zeros(num_steps) for _ in 1:num_agents]
    linearization_points = [zeros(2, horizon) for _ in 1:num_agents]
    path_histories = [zeros(2, horizon, num_steps) for _ in 1:num_agents]  # Changed dimensions
    convergence_history = Dict("iterations" => [], "primal_residuals" => [])

    # Main simulation loop
    for t in 1:num_steps
        println("Step $t/$num_steps")
        # Store current states
        for i in 1:num_agents
            state_histories[i][:, t] = get_state(agents[i])
        end
        
        # Create reference windows for each agent
        ref_windows = [
            create_reference_window(ref_trajs[i], t, horizon)
            for i in 1:num_agents
        ]

        
        # Solve distributed MPC problem
        controls, history, paths = solve_distributed_mpc!(controller, agents, ref_windows, linearization_points)
        # Store paths for plotting
        for i in 1:num_agents
            path_histories[i][:, :, t] = paths[i]  # Store full horizon prediction at time t
        end
        # Store convergence history
        push!(convergence_history["iterations"], history["iterations"])
        push!(convergence_history["primal_residuals"], history["primal_residuals"])
        linearization_points = shift_solution(paths)
        # Apply controls and update states
        for i in 1:num_agents
            control_histories[i][:, t] = controls[:, i]  # Use only the controls part
            next_state = true_dynamics(get_state(agents[i]), controls[:, i])  # Use column vector
            reset_state!(agents[i], next_state)

            # Compute MSE
            mse_histories[i][t] = mean((get_state(agents[i]) - ref_trajs[i][:, t]).^2)
        end

    end

    # Find the maximum length of primal residuals
    max_length = maximum(length.(convergence_history["primal_residuals"]))
    
    # Pad shorter arrays with their last value
    padded_residuals = []
    for residuals in convergence_history["primal_residuals"]
        if length(residuals) < max_length
            # Pad with the last value
            padded = vcat(residuals, fill(residuals[end], max_length - length(residuals)))
            push!(padded_residuals, padded)
        else
            push!(padded_residuals, residuals)
        end
    end
    
    
    # what happens if they are not the same size? (errors apparently)
    mean_primal_residuals = mean(padded_residuals, dims=1)
    # reduce the mean to a vector
    mean_primal_residuals = mean_primal_residuals[1]
    # Plotting
    p = plot(
        title="Distributed MPC with GP Dynamics",
        xlabel="x₁", ylabel="x₂",
        aspect_ratio=:equal,
        legend=:outerright
    )
    
    colors = [:blue, :red, :green]
    
    for i in 1:num_agents
        # Plot reference
        plot!(p, ref_trajs[i][1,:], ref_trajs[i][2,:],
            label="Reference $i",
            color=colors[i],
            linestyle=:dash
        )
        
        # Plot actual trajectory
        plot!(p, state_histories[i][1,:], state_histories[i][2,:],
            label="Agent $i",
            color=colors[i]
        )
    end
    # save this plot
    savefig(p, "agent_trajectories.png")
    display(p)
    
    # Print MSE statistics
    println("\nMSE Summary:")
    for i in 1:num_agents
        mean_mse = mean(mse_histories[i])
        println("Agent $i: $(round(mean_mse, digits=6))")
    end

    # plot convergence of the primal
    p = plot(
        title="ADMM Convergence",
        xlabel="Iteration",
        ylabel="Residual (log scale)",
        yscale=:log10,
        legend=:topright
    )
    plot!(p, 1:length(mean_primal_residuals),
        mean_primal_residuals,
        label="Primal Residual",
        linewidth=2,
        marker=:circle,
        markersize=4
    )
    # save the plot
    savefig(p, "admm_convergence.png")

    # Create the snapshot plots with path_histories instead of path
    create_snapshot_plots(state_histories, ref_trajs, path_histories, agents, num_steps)
    

# Save the simulation results
    save_simulation_results((states=state_histories, controls=control_histories, mse=mse_histories), ref_trajs, agents, controller)
    return (
        states=state_histories,
        controls=control_histories,
        mse=mse_histories
    ), ref_trajs
end

# First, add JLD2 to your dependencies
using JLD2

# Add this function after run_distributed_comparison()
function save_simulation_results(results, ref_trajs, agents, controller; filename="simulation_results.jld2")
    violations = check_collisions(results.states, controller.config.safety_distance)
    
    # Collect all simulation parameters
    simulation_params = Dict(
        "num_agents" => length(agents),
        "num_steps" => size(results.states[1], 2),
        "horizon" => controller.config.mpc_params.horizon,
        "safety_distance" => controller.config.safety_distance,
        "rho" => controller.rho,
        "max_iterations" => controller.max_iterations,
        "tolerance" => controller.tolerance,
        "dynamics_mode" => string(controller.dynamics_mode),
        "control_bounds" => controller.config.mpc_params.control_bounds,
        "state_bounds" => controller.config.mpc_params.state_bounds
    )

    # Collect GP parameters for each agent
    gp_params = []
    for agent in agents
        if !isnothing(agent.gp_model)
            push!(gp_params, Dict(
                "agent_id" => agent.id,
                "kernel_x" => agent.gp_model.models[1].kernel,
                "kernel_y" => agent.gp_model.models[2].kernel,
                "noise_x" => exp(agent.gp_model.models[1].logNoise.value),
                "noise_y" => exp(agent.gp_model.models[2].logNoise.value)
            ))
        end
    end

    # Prepare data dictionary
    data = Dict(
        "simulation_params" => simulation_params,
        "gp_params" => gp_params,
        "results" => Dict(
            "states" => results.states,
            "controls" => results.controls,
            "mse" => results.mse
        ),
        "reference_trajectories" => ref_trajs,
        "violations" => violations,
        "timestamp" => now(),
        "git_commit" => try
            read(`git rev-parse HEAD`, String)
        catch
            "Git info not available"
        end
    )

    # Save to file
    save(filename, data)
    println("\nResults saved to $filename")
end

# Check collision distance violations
function check_collisions(state_histories, safety_distance)
    num_agents = length(state_histories)
    num_steps = size(state_histories[1], 2)
    violations = []
    
    for t in 1:num_steps
        for i in 1:num_agents
            for j in (i+1):num_agents
                dist = norm(state_histories[i][:, t] - state_histories[j][:, t])
                if dist < safety_distance
                    push!(violations, (t, i, j, dist))
                end
            end
        end
    end
    return violations
end

function create_snapshot_plots(state_histories, ref_trajs, paths, agents, num_steps)
    # Select timesteps for plots
    plot_times = [1, num_steps ÷ 3, 2num_steps ÷ 3, num_steps]
    
    # Create 2x2 subplot layout
    p = plot(
        layout=(2,2),
        size=(1200,1000),
        title=["t=$(plot_times[1])" "t=$(plot_times[2])" "t=$(plot_times[3])" "t=$(plot_times[4])"],
        legend=false  # Remove legends from all subplots
    )
    
    colors = [:blue, :red, :green]
    
    for (idx, t) in enumerate(plot_times)
        subplot = p[idx]
        
        # Plot reference trajectories without labels
        for i in 1:length(ref_trajs)
            plot!(subplot, 
                ref_trajs[i][1,:], ref_trajs[i][2,:],
                color=colors[i],
                linestyle=:dash,
                alpha=0.5
            )
        end
        
        # For each agent
        for i in 1:length(state_histories)
            # Plot past trajectory without labels
            plot!(subplot,
                state_histories[i][1,1:t], state_histories[i][2,1:t],
                color=colors[i],
                alpha=0.3
            )
            
            # Current position without label
            scatter!(subplot,
                [state_histories[i][1,t]], [state_histories[i][2,t]],
                color=colors[i],
                marker=:circle,
                markersize=6
            )
            
            # Plot MPC predictions with uncertainty
            if !isnothing(agents[i].gp_model)
                current_pred = paths[i][:,:,t]
                
                for h in 1:size(current_pred, 2)
                    μ1, σ1 = predict_f(agents[i].gp_model.models[1], reshape(current_pred[:, h], :, 1))
                    μ2, σ2 = predict_f(agents[i].gp_model.models[2], reshape(current_pred[:, h], :, 1))
                    
                    θ = range(0, 2π, length=50)
                    ellipse_x = current_pred[1, h] .+ 2σ1[1] .* cos.(θ)
                    ellipse_y = current_pred[2, h] .+ 2σ2[1] .* cos.(θ)
                    
                    # Prediction point without label
                    scatter!(subplot,
                        [current_pred[1, h]], [current_pred[2, h]],
                        color=colors[i],
                        marker=:diamond,
                        markersize=4
                    )
                    
                    # Uncertainty ellipse without label
                    plot!(subplot,
                        ellipse_x, ellipse_y,
                        color=colors[i],
                        alpha=0.1,
                        fill=true
                    )
                end
                
                # Connect predictions without label
                plot!(subplot,
                    current_pred[1, :], current_pred[2, :],
                    color=colors[i],
                    alpha=0.6,
                    linestyle=:solid
                )
            end
        end
        
        plot!(subplot,
            xlabel="x₁",
            ylabel="x₂",
            aspect_ratio=:equal,
            xlims=(-3, 3),
            ylims=(-3, 3)
        )
    end
    
    savefig(p, "mpc_snapshots.png")
    display(p)
end

# Run simulation
Random.seed!(1234)
results, ref_trajs = run_distributed_comparison()

# Check violations with safety distance
safety_distance = 0.5  # Same as in controller
violations = check_collisions(results.states, safety_distance)

# Print violation summary
if isempty(violations)
    println("\nNo collision distance violations found!")
else
    println("\nFound $(length(violations)) collision distance violations:")
    for (t, i, j, dist) in violations
        println("Time $(t): Agents $i and $j got too close (distance: $(round(dist, digits=3)) < $(safety_distance))")
    end
end
