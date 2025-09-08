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

function run_distributed_comparison(controller, agents, ref_trajs)
    # Configuration
    num_agents = length(agents)
    num_steps = size(ref_trajs[1], 2)  # Get num_steps from reference trajectory
    horizon = controller.config.mpc_params.horizon
    
    # Storage for results
    state_histories = [zeros(2, num_steps) for _ in 1:num_agents]
    control_histories = [zeros(2, num_steps) for _ in 1:num_agents]
    mse_histories = [zeros(num_steps) for _ in 1:num_agents]
    linearization_points = [zeros(2, horizon) for _ in 1:num_agents]
    path_histories = [zeros(2, horizon, num_steps) for _ in 1:num_agents]
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
            next_state = true_dynamics(CGPMPC_src.get_state(agents[i]), controls[:, i])  # Use column vector
            reset_state!(agents[i], next_state)

            # Compute MSE
            mse_histories[i][t] = mean((CGPMPC_src.get_state(agents[i]) - ref_trajs[i][:, t]).^2)
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
    # create_snapshot_plots(state_histories, ref_trajs, path_histories, agents, num_steps)
    

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
function run_monte_carlo_analysis(num_runs=5)
    # Basic setup
    num_agents = 3
    num_steps = 80
    horizon = 13
    
    # Define dynamics modes to compare
    modes = Dict(
        "Nominal" => (Nominal, false),
        "Nonlinear" => (Nonlinear, false),
        "LinearizedGP" => (LinearizedGP, false)
    )
    
    # Monte Carlo results storage
    monte_carlo_results = Dict(
        mode_name => Dict(
            "path_lengths" => [],
            "mse_values" => [],
            "collisions" => [],
            "final_distances" => []
        ) for mode_name in keys(modes)
    )
    
    for run in 1:num_runs
        println("\nMonte Carlo Run $run/$num_runs")
        
        # Generate new reference trajectories for each run
        ref_trajs = generate_crossing_trajectories(num_steps, num_agents, radius=4.0, spacing=1.0)
        
        for (mode_name, (dynamics_mode, use_chance)) in modes
            println("  Testing $mode_name...")
            
            # Create controller
            controller = DistributedMPCController(
                ControllerConfig(
                    MPCParams(horizon, (-2.0, 2.0), (-100.0, 100.0), [1.0,1.0], Dict("print_level" => 0)),
                    use_chance, 0.5, 0.5
                ),
                dynamics_mode,
                20.0,    # rho
                100,     # max_iterations
                1e-9     # tolerance
            )
            
            # Initialize agents
            agents = [Agent(AgentConfig(i, horizon, 0.4, (-2.0, 2.0), (-100.0, 100.0)), zeros(2)) 
                     for i in 1:num_agents]
            
            # Initialize and train agents
            for (agent, ref_traj) in zip(agents, ref_trajs)
                reset_state!(agent, ref_traj[:, 1])
                collect_training_data!(agent, ref_traj, 100)
                train_gp!(agent)
            end
            
            # Run simulation
            results, _ = run_distributed_comparison(controller, agents, ref_trajs)
            
            # Calculate metrics
            path_lengths = [calculate_path_length(results.states[i]) for i in 1:num_agents]
            mse_values = [mean(results.mse[i]) for i in 1:num_agents]
            violations = check_collisions(results.states, controller.config.safety_distance)
            distances = calculate_agent_distances(results.states)
            
            # Store results
            push!(monte_carlo_results[mode_name]["path_lengths"], path_lengths)
            push!(monte_carlo_results[mode_name]["mse_values"], mse_values)
            push!(monte_carlo_results[mode_name]["collisions"], length(violations))
            push!(monte_carlo_results[mode_name]["final_distances"], 
                  [distances[key][end] for key in keys(distances)])
        end
    end
    
    # Plot and save results
    plot_monte_carlo_results(monte_carlo_results, num_agents, num_runs)
    
    return monte_carlo_results
end

function plot_monte_carlo_results(results, num_agents, num_runs)
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    
    # Create subplots for statistical analysis
    p = plot(
        layout=(2,2),
        size=(1200,800),
        title="Monte Carlo Analysis ($num_runs runs)"
    )
    
    # 1. Average Path Lengths (top left)
    subplot = p[1]
    plot!(subplot,
        title="Average Path Length by Agent",
        ylabel="Path Length",
        xlabel="Method",
        legend=:topright
    )
    
    mode_names = collect(keys(results))
    for (i, mode) in enumerate(mode_names)
        agent_means = [mean([run[j] for run in results[mode]["path_lengths"]]) for j in 1:num_agents]
        agent_stds = [std([run[j] for run in results[mode]["path_lengths"]]) for j in 1:num_agents]
        
        # Plot with error bars
        scatter!(subplot, fill(i, num_agents), agent_means,
            yerror=agent_stds,
            label=mode,
            markersize=6
        )
    end
    xticks!(subplot, 1:length(mode_names), mode_names)
    
    # 2. MSE Distribution (top right)
    subplot = p[2]
    boxplot!(subplot,
        title="MSE Distribution",
        ylabel="MSE",
        xlabel="Method",
        yscale=:log10,
        legend=false
    )
    
    for mode in mode_names
        mse_values = vcat([values for values in results[mode]["mse_values"]]...)
        boxplot!(subplot, [mode], mse_values)
    end
    
    # 3. Collision Statistics (bottom left)
    subplot = p[3]
    bar!(subplot,
        title="Average Collisions per Run",
        ylabel="Number of Collisions",
        xlabel="Method",
        legend=false
    )
    
    collision_means = [mean(results[mode]["collisions"]) for mode in mode_names]
    collision_stds = [std(results[mode]["collisions"]) for mode in mode_names]
    bar!(subplot, mode_names, collision_means, yerror=collision_stds)
    
    # 4. Final Distance Distribution (bottom right)
    subplot = p[4]
    boxplot!(subplot,
        title="Final Inter-agent Distances",
        ylabel="Distance",
        xlabel="Method",
        legend=false
    )
    
    for mode in mode_names
        distances = vcat(results[mode]["final_distances"]...)
        boxplot!(subplot, [mode], distances)
    end
    
    # Save plots and data
    savefig(p, "monte_carlo_summary_$timestamp.pdf")
    savefig(p, "monte_carlo_summary_$timestamp.png")
    display(p)
    
    # Save numerical data
    save("monte_carlo_results_$timestamp.jld2", Dict(
        "results" => results,
        "num_runs" => num_runs,
        "num_agents" => num_agents,
        "timestamp" => timestamp
    ))
    
    # Print summary statistics
    println("\nMonte Carlo Summary Statistics ($num_runs runs):")
    for mode in mode_names
        println("\n$mode:")
        
        # Path lengths
        println("  Average Path Lengths:")
        for i in 1:num_agents
            lengths = [run[i] for run in results[mode]["path_lengths"]]
            println("    Agent $i: $(round(mean(lengths), digits=3)) ± $(round(std(lengths), digits=3))")
        end
        
        # MSE
        mse_values = vcat([values for values in results[mode]["mse_values"]]...)
        println("  MSE: $(round(mean(mse_values), digits=6)) ± $(round(std(mse_values), digits=6))")
        
        # Collisions
        println("  Collisions: $(round(mean(results[mode]["collisions"]), digits=2)) ± $(round(std(results[mode]["collisions"]), digits=2))")
    end
end

# Run Monte Carlo analysis
Random.seed!(1234)  # For reproducibility
results = run_monte_carlo_analysis(5)  # 50 runs