using Plots, JLD2, Dates
function plot_saved_snapshots(data)
    # Get required data
    all_results = data["all_results"]
    ref_trajs = data["ref_trajs"]
    num_agents = length(ref_trajs)
    agent_colors = [:royalblue, :crimson, :forestgreen]
    
    # Define mode order
    mode_order = [
        "Nominal-NoChance",
        "Nominal-Chance",
        "Nonlinear-NoChance",
        "Nonlinear-Chance",
        "LinearizedGP-NoChance",
        "LinearizedGP-Chance"
    ]
    
    # Create snapshots for each mode
    for mode_name in mode_order
        results = all_results[mode_name]
        num_steps = size(results.states[1], 2)
        plot_times = [1, num_steps ÷ 3, 2num_steps ÷ 3, num_steps]
        
        p = plot(
            layout=(1,4),
            size=(1600,400),
            title=["t=$(plot_times[1])" "t=$(plot_times[2])" "t=$(plot_times[3])" "t=$(plot_times[4])"],
            legend=false,
            plot_title=""  # Remove main title
        )
        
        for (idx, t) in enumerate(plot_times)
            subplot = p[idx]
            
            # Plot reference trajectories
            for i in 1:num_agents
                plot!(subplot, 
                    ref_trajs[i][1,:], ref_trajs[i][2,:],
                    color=agent_colors[i],
                    linestyle=:dash,
                    alpha=0.5,
                    linewidth=2  # Increased line width
                )
            end
            
            # Plot actual trajectories
            for i in 1:num_agents
                # Past trajectory
                plot!(subplot,
                    results.states[i][1,1:t], results.states[i][2,1:t],
                    color=agent_colors[i],
                    alpha=0.3,
                    linewidth=3  # Increased line width
                )
                
                # Current position
                scatter!(subplot,
                    [results.states[i][1,t]], [results.states[i][2,t]],
                    color=agent_colors[i],
                    marker=:circle,
                    markersize=6
                )
                
                # Plot predicted future path if available
                if haskey(results, :paths) && t <= size(results.paths[i], 3)
                    future_path = results.paths[i][:,:,t]
                    plot!(subplot,
                        future_path[1,:], future_path[2,:],
                        color=agent_colors[i],
                        linestyle=:dot,
                        alpha=0.6,
                        linewidth=2  # Increased line width
                    )
                    
                    # Plot endpoints of predictions
                    scatter!(subplot,
                        [future_path[1,end]], [future_path[2,end]],
                        color=agent_colors[i],
                        marker=:diamond,
                        markersize=4
                    )
                end
            end
            
            # Customize subplot axes
            plot!(subplot,
                xlabel=(idx == 2 || idx == 3) ? "" : "x₁",  # Only show x₁ on outer plots
                ylabel=(idx == 1) ? "x₂" : "",  # Only show x₂ on leftmost plot
                aspect_ratio=:equal,
                xlims=(-4.5, 4.5),
                ylims=(-4.5, 4.5)
            )
        end
        
        # Remove overall title
        plot!(plot_title="")
        
        # Save and display
        timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
        filename_base = lowercase(replace(mode_name, " " => "_"))
        savefig(p, "snapshots_$(filename_base)_$timestamp.pdf")
        savefig(p, "snapshots_$(filename_base)_$timestamp.png")
        display(p)
    end
end

# Update load_and_plot_results to include snapshot plotting
function load_and_plot_results(filename)
    # Load the data
    data = load(filename)
    
    # Plot snapshots
    plot_saved_snapshots(data)
    
    # ... rest of existing plotting code ...
end
# In Julia REPL:
filename = "archive/dynamics_comparison_data_2025-06-14_12-55-57.jld2"  # Replace with your saved file
load_and_plot_results(filename)