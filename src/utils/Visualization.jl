module Visualization

using Plots
export visualize_agents, visualize_multi_agents

"""
Visualize agent trajectories and reference path
"""
function visualize_agents(results, ref_traj)
    # Create plot
    p = plot(
        xlabel="x position",
        ylabel="y position",
        title="Agent Trajectories",
        aspect_ratio=:equal,
        legend=:topleft
    )
    
    # Plot reference trajectory
    plot!(p, ref_traj[1, :], ref_traj[2, :],
        label="Reference",
        color=:black,
        linestyle=:dash,
        linewidth=2
    )
    
    # Plot agent trajectories for each phase
    colors = [:blue, :red, :green]
    labels = ["Standard MPC", "GP-MPC", "Linear GP-MPC"]
    
    for (i, history) in enumerate(results.state_histories)
        plot!(p, history[1, :], history[2, :],
            label=labels[i],
            color=colors[i],
            linewidth=2
        )
    end
    
    # Create MSE subplot
    p2 = plot(
        xlabel="Time step",
        ylabel="MSE",
        title="Tracking Error",
        yscale=:log10,
        legend=:topleft
    )
    
    for (i, mse) in enumerate(results.mse_histories)
        plot!(p2, mse,
            label=labels[i],
            color=colors[i],
            linewidth=2
        )
    end
    
    # Combine plots
    final_plot = plot(p, p2, layout=(2,1), size=(800,1000))
    
    # Save plot
    savefig(final_plot, "trajectory_comparison.png")
    
    return final_plot
end

"""
Visualize multiple agents' trajectories and their reference paths
"""
function visualize_multi_agents(results, ref_trajs)
    # Create trajectory plot
    p1 = plot(
        xlabel="x position",
        ylabel="y position",
        title="Multi-Agent Trajectories",
        aspect_ratio=:equal,
        legend=:topleft
    )
    
    # Define colors for different agents
    colors = [:blue, :red, :green]
    
    # Plot reference trajectories and agent paths
    for (i, (history, ref_traj)) in enumerate(zip(results.state_histories, ref_trajs))
        # Plot reference
        plot!(p1, ref_traj[1, :], ref_traj[2, :],
            label="Reference $(i)",
            color=colors[i],
            linestyle=:dash,
            linewidth=2
        )
        
        # Plot actual trajectory
        plot!(p1, history[1, :], history[2, :],
            label="Agent $(i)",
            color=colors[i],
            linewidth=2
        )
    end
    
    # Create MSE plot
    p2 = plot(
        xlabel="Time step",
        ylabel="MSE",
        title="Tracking Errors",
        yscale=:log10,
        legend=:topleft
    )
    
    # Plot MSE for each agent
    for (i, mse) in enumerate(results.mse_histories)
        plot!(p2, mse,
            label="Agent $(i)",
            color=colors[i],
            linewidth=2
        )
    end
    
    # Combine plots
    final_plot = plot(p1, p2, layout=(2,1), size=(800,1000))
    
    # Save plot
    savefig(final_plot, "multi_agent_comparison.png")
    
    return final_plot
end

end