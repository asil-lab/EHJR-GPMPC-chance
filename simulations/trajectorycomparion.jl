using Plots
using Random
using Statistics
using GaussianProcesses: predict_f
using LinearAlgebra
using JLD2
using Dates
using StatsPlots
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

# Add a function to check if all agents have reached their goals
function all_agents_at_goal(agents, ref_trajs, goal_threshold=0.1)
    for (agent, ref_traj) in zip(agents, ref_trajs)
        goal_position = ref_traj[:, end]
        current_position = get_state(agent)
        if norm(current_position - goal_position) > goal_threshold
            return false
        end
    end
    return true
end

# Modify run_distributed_comparison function
function run_distributed_comparison(controller, agents, ref_trajs; max_steps=100, goal_threshold=0.1)
    # Configuration
    num_agents = length(agents)
    horizon = controller.config.mpc_params.horizon
    ref_length = size(ref_trajs[1], 2)
    
    # Storage for results (preallocate for max_steps)
    state_histories = [zeros(2, max_steps) for _ in 1:num_agents]
    control_histories = [zeros(2, max_steps) for _ in 1:num_agents]
    mse_histories = [zeros(max_steps) for _ in 1:num_agents]
    linearization_points = [zeros(2, horizon) for _ in 1:num_agents]
    path_histories = [zeros(2, horizon, max_steps) for _ in 1:num_agents]
    
    # Initialize convergence history
    convergence_history = Dict(
        "iterations" => Int[],
        "primal_residuals" => Vector{Float64}[]
    )

    # Main simulation loop
    actual_steps = 0
    for t in 1:max_steps
        println("Step $t/$max_steps")
        actual_steps = t
        
        # Store current states
        for i in 1:num_agents
            state_histories[i][:, t] = get_state(agents[i])
        end

        # Check if all agents have reached their goals
        if all_agents_at_goal(agents, ref_trajs, goal_threshold)
            println("\nAll agents reached their goals at step $t !")
            break
        end
        
        # Create reference windows for each agent (with bounds checking)
        ref_windows = [
            create_reference_window(ref_trajs[i], min(t, ref_length), horizon)
            for i in 1:num_agents
        ]

        
        # Solve distributed MPC problem
        controls, history, paths = solve_distributed_mpc!(controller, agents, ref_windows, linearization_points)
        # Store paths for plotting (always, not conditionally)
        for i in 1:num_agents
            path_histories[i][:, :, t] = paths[i]
        end
        
        # Store convergence history
        push!(convergence_history["iterations"], history["iterations"][end])
        push!(convergence_history["primal_residuals"], history["primal_residuals"])
        linearization_points = shift_solution(paths)
        # Apply controls and update states
        for i in 1:num_agents
            control_histories[i][:, t] = controls[:, i]
            next_state = true_dynamics(get_state(agents[i]), controls[:, i])
            reset_state!(agents[i], next_state)
            
            # Use last reference point if beyond reference trajectory
            ref_idx = min(t, ref_length)
            mse_histories[i][t] = mean((get_state(agents[i]) - ref_trajs[i][:, ref_idx]).^2)
        end

    end

    # Trim results to actual number of steps used
    for i in 1:num_agents
        state_histories[i] = state_histories[i][:, 1:actual_steps]
        control_histories[i] = control_histories[i][:, 1:actual_steps]
        mse_histories[i] = mse_histories[i][1:actual_steps]
        path_histories[i] = path_histories[i][:, :, 1:actual_steps]
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
    # p = plot(
    #     title="Distributed MPC with GP Dynamics",
    #     xlabel="x₁", ylabel="x₂",
    #     aspect_ratio=:equal,
    #     legend=:outerright
    # )
    
    # colors = [:blue, :red, :green]
    
    # for i in 1:num_agents
    #     # Plot reference
    #     plot!(p, ref_trajs[i][1,:], ref_trajs[i][2,:],
    #         label="Reference $i",
    #         color=colors[i],
    #         linestyle=:dash
    #     )
        
    #     # Plot actual trajectory
    #     plot!(p, state_histories[i][1,:], state_histories[i][2,:],
    #         label="Agent $i",
    #         color=colors[i]
    #     )
    # end
    # # save this plot
    # savefig(p, "agent_trajectories.png")
    # display(p)
    
    # Print MSE statistics
    # println("\nMSE Summary:")
    # for i in 1:num_agents
    #     mean_mse = mean(mse_histories[i])
    #     println("Agent $i: $(round(mean_mse, digits=6))")
    # end

    # plot convergence of the primal
    # p = plot(
    #     title="ADMM Convergence",
    #     xlabel="Iteration",
    #     ylabel="Residual (log scale)",
    #     yscale=:log10,
    #     legend=:topright
    # )
    # plot!(p, 1:length(mean_primal_residuals),
    #     mean_primal_residuals,
    #     label="Primal Residual",
    #     linewidth=2,
    #     marker=:circle,
    #     markersize=4
    # )
    # # save the plot
    # savefig(p, "admm_convergence.png")

    # Create the snapshot plots with path_histories instead of path
    create_snapshot_plots(state_histories, ref_trajs, path_histories, agents, actual_steps)
    

# Save the simulation results
    save_simulation_results((states=state_histories, controls=control_histories, mse=mse_histories), ref_trajs, agents, controller)
    return (
        states=state_histories,
        controls=control_histories,
        mse=mse_histories,
        paths=path_histories  # Add paths to returned results
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
    
    # savefig(p, "mpc_snapshots.png")
    display(p)
end

function calculate_path_length(states)
    # Calculate total distance traveled
    length = 0.0
    for t in 2:size(states, 2)
        length += norm(states[:, t] - states[:, t-1])
    end
    return length
end

function calculate_agent_distances(states)
    num_agents = length(states)
    num_steps = size(states[1], 2)
    distances = Dict()
    
    for i in 1:num_agents
        for j in (i+1):num_agents
            key = "$(i)-$(j)"
            distances[key] = zeros(num_steps)
            for t in 1:num_steps
                distances[key][t] = norm(states[i][:, t] - states[j][:, t])
            end
        end
    end
    return distances
end

function compare_dynamics_modes()
    # Basic setup
    num_agents = 3
    num_steps = 50
    max_steps = 150  # Changed from num_steps
    horizon = 13
    
    # Define dynamics modes to compare, each with and without chance constraints
    modes = Dict(
        "Nominal-NoChance" => (Nominal, false),
        "Nominal-Chance" => (Nominal, true),
        "Nonlinear-NoChance" => (Nonlinear, false),
        "Nonlinear-Chance" => (Nonlinear, true),
        "LinearizedGP-NoChance" => (LinearizedGP, false),
        "LinearizedGP-Chance" => (LinearizedGP, true)
    )
    
    # Generate reference trajectories (same for all modes)
    ref_trajs = generate_crossing_trajectories(num_steps, num_agents, radius=4.0, spacing=1.0)
    
    # Results storage
    all_results = Dict()
    
    # Train GP models once for all modes
    trained_agents = [Agent(AgentConfig(i, horizon, 0.4, (-2.0, 2.0), (-100.0, 100.0)), zeros(2)) 
                     for i in 1:num_agents]
                     
    # Train all agents with GP models
    for (agent, ref_traj) in zip(trained_agents, ref_trajs)
        reset_state!(agent, ref_traj[:, 1])
        collect_training_data!(agent, ref_traj, 100)
        train_gp!(agent)
    end
    
    for (mode_name, (dynamics_mode, use_chance)) in modes
        println("\nTesting dynamics mode: $mode_name (Chance Constraints: $use_chance)")
        
        # Create controller with specific chance constraint setting
        controller = DistributedMPCController(
            ControllerConfig(
                MPCParams(horizon, (-2.0, 2.0), (-100.0, 100.0),[1.0,1.0], Dict("print_level" => 0)),
                use_chance,  # Use the specific chance constraint setting
                0.95,       # confidence level
                0.5,        # communication radius
                0.5         # safety distance
            ),
            dynamics_mode,
            20.0,    # rho
            200,     # max_iterations
            1e-9     # tolerance
        )
        
        # Create new agents with copied GP models
        agents = [Agent(AgentConfig(i, horizon, 0.4, (-2.0, 2.0), (-100.0, 100.0)), zeros(2)) 
                 for i in 1:num_agents]
        
        # Copy trained GP models and initial states
        for (agent, trained_agent, ref_traj) in zip(agents, trained_agents, ref_trajs)
            agent.gp_model = trained_agent.gp_model  # Keep trained GP
            reset_state!(agent, ref_traj[:, 1])
        end
        
        # Run simulation with max_steps
        results, _ = run_distributed_comparison(
            controller, 
            agents, 
            ref_trajs; 
            max_steps=max_steps, 
            goal_threshold=0.1
        )
        all_results[mode_name] = results
        
        # Save individual results
        timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
        filename = "simulation_results_$(mode_name)_$timestamp.jld2"
        save_simulation_results(results, ref_trajs, agents, controller, filename=filename)
    end
    
    # Create comparison plots
    plot_dynamics_comparison(all_results, ref_trajs, num_agents)
    
    return all_results, ref_trajs
end

function plot_dynamics_comparison(all_results, ref_trajs, num_agents)
    # Create timestamp for saving
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    colors = [:blue, :red, :green]
    
    # Plot separate figure for each dynamics mode
    for (mode_name, results) in all_results
        # Select 4 timesteps for snapshots
        num_steps = size(results.states[1], 2)
        plot_times = [1, num_steps ÷ 3, 2num_steps ÷ 3, num_steps]
        
        # Create 1x4 subplot layout
        p = plot(
            layout=(1,4),
            size=(1600,400),
            title=["t=$(plot_times[1])" "t=$(plot_times[2])" "t=$(plot_times[3])" "t=$(plot_times[4])"],
            legend=false
        )
        
        # Create snapshots
        for (idx, t) in enumerate(plot_times)
            subplot = p[idx]
            
            # Plot reference trajectories
            for i in 1:num_agents
                plot!(subplot, 
                    ref_trajs[i][1,:], ref_trajs[i][2,:],
                    color=colors[i],
                    linestyle=:dash,
                    alpha=0.5
                )
            end
            
            # Plot actual trajectories up to current time
            for i in 1:num_agents
                # Past trajectory
                plot!(subplot,
                    results.states[i][1,1:t], results.states[i][2,1:t],
                    color=colors[i],
                    alpha=0.3
                )
                
                # Current position
                scatter!(subplot,
                    [results.states[i][1,t]], [results.states[i][2,t]],
                    color=colors[i],
                    marker=:circle,
                    markersize=6
                )
                
                # Plot predicted future path if available
                if haskey(results, :paths) && t <= size(results.paths[i], 3)  # Changed from hasfield
                    future_path = results.paths[i][:,:,t]
                    plot!(subplot,
                        future_path[1,:], future_path[2,:],
                        color=colors[i],
                        linestyle=:dot,
                        alpha=0.6
                    )
                    
                    # Plot endpoints of predictions
                    scatter!(subplot,
                        [future_path[1,end]], [future_path[2,end]],
                        color=colors[i],
                        marker=:diamond,
                        markersize=4
                    )
                end
            end
            
            plot!(subplot,
                xlabel="x₁",
                ylabel="x₂",
                aspect_ratio=:equal,
                xlims=(-4.5, 4.5),
                ylims=(-4.5, 4.5)
            )
        end
        
        # Add overall title
        plot!(plot_title="$mode_name Dynamics")
        
        # Save figure
        filename_base = lowercase(replace(mode_name, " " => "_"))
        # savefig(p, "snapshots_$(filename_base)_$timestamp.pdf")
        # savefig(p, "snapshots_$(filename_base)_$timestamp.png")
        display(p)
    end
    
    # Save complete dataset for reproducibility
    save("dynamics_comparison_data_$timestamp.jld2", Dict(
        "all_results" => all_results,
        "ref_trajs" => ref_trajs,
        "timestamp" => timestamp,
        "plot_metadata" => Dict(
            "num_agents" => num_agents,
            "colors" => colors,
            "git_commit" => try
                read(`git rev-parse HEAD`, String)
            catch
                "Git info not available"
            end
        )
    ))
    
    # Create comparison summary plot with boxplot
    p_summary = plot(
        layout=(2,1),
        size=(800,1000),
        title="Dynamics Mode Comparison"
    )
    
    # MSE evolution
    subplot = p_summary[1]
    plot!(subplot,
        title="MSE Evolution",
        xlabel="Time Step",
        ylabel="MSE (log)",
        yscale=:log10,
        legend=:topright
    )
    
    for (mode_name, results) in all_results
        mean_mse = mean([results.mse[i] for i in 1:num_agents])
        plot!(subplot, mean_mse,
            label=mode_name,
            linewidth=2
        )
    end
    
    # MSE boxplot comparison
    subplot = p_summary[2]
    
    # Prepare data for boxplot
    mode_names = collect(keys(all_results))
    mse_data = []
    for mode in mode_names
        # Collect MSE values across all agents and timesteps
        mode_mse = vcat([all_results[mode].mse[i] for i in 1:num_agents]...)
        push!(mse_data, mode_mse)
    end
    
    # Create boxplot
    # boxplot!(subplot,
    #     repeat(mode_names, inner=length(first(mse_data))),
    #     vcat(mse_data...),
    #     title="MSE Distribution by Mode",
    #     ylabel="MSE ",
    #     yscale=:linear,
    #     legend=false,
    #     fillalpha=0.75,
    #     outliers=true,
    #     whisker_width=0.5,
    #     notch=true  # Add confidence interval around median
    # )
    
    # Save summary plot
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    savefig(p_summary, "dynamics_comparison_summary_$timestamp.png")
    display(p_summary)
    
    # Calculate and plot path lengths
    p_lengths = plot(
        layout=(num_agents, 1),  # One subplot per agent
        size=(1000, 300 * num_agents),
        title="Path Lengths by Method and Agent"
    )
    
    # Calculate reference path lengths
    ref_lengths = [calculate_path_length(ref_traj) for ref_traj in ref_trajs]
    
    # Calculate actual path lengths for each method
    method_lengths = Dict()
    mode_names = collect(keys(all_results))
    
    for (i, mode) in enumerate(mode_names)
        agent_lengths = [calculate_path_length(all_results[mode].states[j]) for j in 1:num_agents]
        method_lengths[mode] = agent_lengths
    end
    
    # Create separate subplot for each agent
    for agent_idx in 1:num_agents
        subplot = p_lengths[agent_idx]
        
        # Get path lengths for this agent across all methods
        agent_data = [method_lengths[mode][agent_idx] for mode in mode_names]
        
        # Create bar plot for this agent
        bar!(subplot, mode_names, agent_data,
            title="Agent $agent_idx",
            ylabel="Path Length",
            label="Actual Path",
            color=:blue,
            alpha=0.7
        )
        
        # Add reference line
        hline!(subplot, [ref_lengths[agent_idx]],
            label="Reference",
            color=:red,
            linestyle=:dash,
            linewidth=2
        )
        
        # Customize subplot
        if agent_idx == num_agents
            plot!(subplot, xlabel="Method")
        end
        plot!(subplot, 
            legend=:topright,
            rotation=45
        )
    end
    
    # Save and display
    savefig(p_lengths, "path_lengths_by_agent_$timestamp.png")
    display(p_lengths)
    
    # Add path lengths to saved data
    # save("dynamics_comparison_data_$timestamp.jld2", Dict(
    #     "all_results" => all_results,
    #     "ref_trajs" => ref_trajs,
    #     "path_lengths" => method_lengths,
    #     "reference_lengths" => ref_lengths,
    #     "timestamp" => timestamp,
    #     "plot_metadata" => Dict(
    #         "num_agents" => num_agents,
    #         "colors" => colors,
    #         "git_commit" => try
    #             read(`git rev-parse HEAD`, String)
    #         catch
    #             "Git info not available"
    #         end
    #     )
    # ))
    
    # Add inter-agent distance plot
    p_distances = plot(
        layout=(length(all_results), 1),
        size=(800, 200 * length(all_results)),
        title="Inter-agent Distances"
    )
    
    for (idx, (mode_name, results)) in enumerate(all_results)
        subplot = p_distances[idx]
        
        # Calculate distances between agents
        distances = calculate_agent_distances(results.states)
        
        # Plot distances
        for (pair, dist) in distances
            plot!(subplot, dist,
                label="Agents $pair",
                linewidth=2
            )
        end
        
        # Add safety distance line
        hline!(subplot, [0.5],
            label="Safety Distance",
            color=:black,
            linestyle=:dash,
            alpha=0.5
        )
        
        plot!(subplot,
            title=mode_name,
            xlabel= idx == length(all_results) ? "Time Step" : "",
            ylabel="Distance",
            legend= idx == 1 ? :right : false,
            ylims=(0, maximum(maximum.(values(distances))) * 1.1)
        )
    end
    
    # Save distance plot
    savefig(p_distances, "agent_distances_$timestamp.pdf")
    savefig(p_distances, "agent_distances_$timestamp.png")
    display(p_distances)
    
    # Add distances to saved data
    save("dynamics_comparison_data_$timestamp.jld2", Dict(
        "all_results" => all_results,
        "ref_trajs" => ref_trajs,
        "path_lengths" => method_lengths,
        "reference_lengths" => ref_lengths,
        "timestamp" => timestamp,
        "plot_metadata" => Dict(
            "num_agents" => num_agents,
            "colors" => colors,
            "git_commit" => try
                read(`git rev-parse HEAD`, String)
            catch
                "Git info not available"
            end
        ),
        "distances" => Dict(
            mode_name => calculate_agent_distances(results.states)
            for (mode_name, results) in all_results
        )
    ))
    
    return timestamp
end

n_runs = 6


# Run dynamics mode comparison
println("Starting dynamics mode comparison...")
for run in 1:2:Int(n_runs/2)
    println("Run $run/$n_runs")
    Random.seed!(1234+15+run)  # For reproducibility
    results, ref_trajs = compare_dynamics_modes()
end


println("\nComparison complete! Check the generated plots and saved data.")
