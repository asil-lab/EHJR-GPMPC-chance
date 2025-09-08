using Plots
using JLD2
using Dates
using Statistics
using LinearAlgebra
using StatsPlots
using LaTeXStrings

function load_and_plot_results(filename)
    # Load the data
    data = load(filename)
    all_results = data["all_results"]
    ref_trajs = data["ref_trajs"]
    num_agents = length(ref_trajs)
    path_lengths = data["path_lengths"]
    reference_lengths = data["reference_lengths"]
    
    # Create timestamp for new plots
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    
   
    
    # 2. Path Length Comparison
    p_lengths = plot(
        layout=(num_agents, 1),
        size=(1000, 300 * num_agents),
        title="Path Lengths by Method and Agent"
    )
    
    # Define the desired order of modes
    mode_order = [
        "Nominal-NoChance",
        "Nominal-Chance",
        "Nonlinear-NoChance",
        "Nonlinear-Chance",
        "LinearizedGP-NoChance",
        "LinearizedGP-Chance"
    ]
    
    # Path Lengths Plot
    p_lengths = plot(
        size=(800, 600),
        # title="Path Lengths by Method",
        xlabel="Method",
        ylabel="Path Length",
        legend=:topright
    )

    # Calculate reference path lengths
    ref_lengths = reference_lengths[1:num_agents]
    
    # Define colors and labels
    agent_colors = [:royalblue, :crimson, :forestgreen]
    agent_labels = ["Agent 1", "Agent 2", "Agent 3"]
    
    # Prepare data in the specified order
    agent_data = zeros(length(mode_order), num_agents)
    
    # Fill data matrix using ordered modes
    for (mode_idx, mode) in enumerate(mode_order)
        for agent_idx in 1:num_agents
            agent_data[mode_idx, agent_idx] = path_lengths[mode][agent_idx]
        end
    end

    # Create stacked bar plot with ordered data
    groupedbar!(p_lengths, agent_data,
        bar_position=:stack,
        xticks=(1:length(mode_order), mode_order),
        xrotation=10,
        label=permutedims(agent_labels),
        color=permutedims(agent_colors),
        width=0.7,
        alpha=0.7
    )

    # Add reference total path length line
    hline!([sum(reference_lengths)],
        label="Total Reference Path",
        color=:black,
        linestyle=:dash,
        alpha=0.5
    )
    
    # Save and display
    savefig(p_lengths, "path_lengths_$timestamp.pdf")
    savefig(p_lengths, "path_lengths_$timestamp.png")
    display(p_lengths)

    # 3. Inter-agent Distance Plot
    p_distances = plot(
        layout=(Int(length(all_results)/2), 2),
        size=(800, 200 * Int(length(all_results)/2)),
        # title="Inter-agent Distances"
    )
    legend_labels = []
    # Plot distances for each mode in specified order
    for (idx, mode_name) in enumerate(mode_order)
        subplot = p_distances[idx]
        
        # Calculate distances for this mode
        distances = data["distances"][mode_name]
        
        # Plot distances for each agent pair
        for (pair, dist) in distances
            plot!(subplot, dist,
                label="Agents $pair",
                linewidth=2
            )
            if idx == 1
                push!(legend_labels, "Agents $pair")
            end
        end
        
        # Add safety distance line
        hline!(subplot, [0.5],
            label=L"r_\mathrm{safe}",
            color=:black,
            linestyle=:dash,
            alpha=0.5
        )
        
        # Customize subplot
        plot!(subplot,
            title=mode_name,
            xlabel=(idx == length(mode_order)||idx ==length(mode_order)-1) ? "Time Step" : "",
            ylabel=(idx == 1||idx ==3||idx==5) ? "Distance" : "",
        #    legend=false,
            legend=idx == 2 ? :topright : false,
            xlims=(20, 35),  # Adjust x-limits as needed
            ylims=(0, 2)  # Set fixed y-limits for better comparison
        )
    end
    # 
    # Add a single legend to the right of the figure
    # plot!(p_distances,
    #     legend=:outerright,
    #     # legendtitle="Agent Pairs",
    #     labels=vcat(legend_labels, ["Safety Distance"]),
    # )
    
    savefig(p_distances, "distances_comparison_$timestamp.png")
    display(p_distances)
    
    # Print statistics
    println("\nSummary Statistics:")
    println("\nPath Length Statistics:")
    for (mode, lengths) in path_lengths
        println("\n$mode:")
        for (i, length) in enumerate(lengths)
            deviation = ((length - reference_lengths[i]) / reference_lengths[i]) * 100
            println("  Agent $i: $(round(length, digits=3)) ($(round(deviation, digits=2))% vs reference)")
        end
    end
    
    println("\nMSE Statistics:")
    for (mode, results) in all_results
        println("\n$mode:")
        for i in 1:num_agents
            mean_mse = mean(results.mse[i])
            println("  Agent $i: $(round(mean_mse, digits=6))")
        end
    end

    # 4. Create snapshot plots
    for (mode_name, results) in all_results
        num_steps = size(results.states[1], 2)
        plot_times = [1, num_steps ÷ 3, 2num_steps ÷ 3, num_steps]
        
        p_snapshots = plot(
            layout=(1,4),
            size=(1600,400),
            title=["t=$(plot_times[1])" "t=$(plot_times[2])" "t=$(plot_times[3])" "t=$(plot_times[4])"],
            legend=false
        )
        
        for (idx, t) in enumerate(plot_times)
            subplot = p_snapshots[idx]
            
            # Plot reference trajectories
            for i in 1:num_agents
                plot!(subplot, 
                    ref_trajs[i][1,:], ref_trajs[i][2,:],
                    color=agent_colors[i],
                    linestyle=:dash,
                    alpha=0.5
                )
            end
            
            # Plot actual trajectories up to current time
            for i in 1:num_agents
                # Past trajectory
                plot!(subplot,
                    results.states[i][1, 1:t], results.states[i][2, 1:t],
                    color=agent_colors[i],
                    linewidth=2
                )
                
                # Future predicted path if available
                if haskey(results, :paths) && t <= size(results.paths[i], 3)
                    future_path = results.paths[i][:,:,t]
                    plot!(subplot,
                        future_path[1,:], future_path[2,:],
                        color=agent_colors[i],
                        linestyle=:dot,
                        alpha=0.6
                    )
                    
                    # Plot endpoints of predictions
                    scatter!(subplot,
                        [future_path[1,end]], [future_path[2,end]],
                        color=agent_colors[i],
                        marker=:diamond,
                        markersize=4
                    )
                end
                
                # Current position
                scatter!(subplot,
                    [results.states[i][1, t]], [results.states[i][2, t]],
                    color=agent_colors[i],
                    markersize=6
                )
            end
            
            plot!(subplot,
                xlabel="x₁",
                ylabel="x₂",
                aspect_ratio=:equal
            )
        end
        
        savefig(p_snapshots, "snapshots_$(mode_name)_$timestamp.png")
        display(p_snapshots)
    end
end

# Example usage:
# Replace with your actual filename

filename = "dynamics_comparison_data_2025-06-14_12-55-57.jld2"
load_and_plot_results(filename)