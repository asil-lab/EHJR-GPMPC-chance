using Plots
using JLD2
using Dates
using Statistics
using LinearAlgebra
using StatsPlots
using LaTeXStrings

function load_and_average_results(filenames)
    # Assume all files have the same modes, agents, and agent pairs
    n_files = length(filenames)
    println("Averaging over $n_files files")
    all_path_lengths = Dict{String, Vector{Vector{Float64}}}()
    all_distances = Dict{String, Dict{String, Vector{Vector{Float64}}}}()
    all_final_times = Dict{String, Vector{Int64}}()
    reference_lengths = nothing
    ref_trajs = nothing

    for (file_idx, filename) in enumerate(filenames)
        println("Loading $filename")
        data = load(filename)
        path_lengths = data["path_lengths"]
        distances = data["distances"]
        if file_idx == 1
            reference_lengths = data["reference_lengths"]
            ref_trajs = data["ref_trajs"]
        end

        # Collect path lengths
        for (mode, lengths) in path_lengths
            if !haskey(all_path_lengths, mode)
                all_path_lengths[mode] = []
            end
            push!(all_path_lengths[mode], lengths)
        end

        # Collect distances
        for (mode, dist_dict) in distances
            if !haskey(all_distances, mode)
                all_distances[mode] = Dict{String, Vector{Vector{Float64}}}()
            end
            for (pair, dist_vec) in dist_dict
                if !haskey(all_distances[mode], pair)
                    all_distances[mode][pair] = []
                end
                push!(all_distances[mode][pair], dist_vec)
            end
        end
        mode_order = [
                "Nominal-NoChance",
                "Nominal-Chance",
                "Nonlinear-NoChance",
                "Nonlinear-Chance",
                "LinearizedGP-NoChance",
                "LinearizedGP-Chance"
            ]
        all_results = data["all_results"]
        for mode_name in mode_order
            if !haskey(all_final_times, mode_name)
                all_final_times[mode_name] = []
            end
            results = all_results[mode_name]
            num_steps = size(results.states[1], 2)
            # Collect final times for each mode
            push!(all_final_times[mode_name], num_steps)
        end
    end

    # Average path lengths
    avg_path_lengths = Dict{String, Vector{Float64}}()
    for (mode, runs) in all_path_lengths
        # runs is a Vector of Vectors, each inner Vector is for all agents
        avg_path_lengths[mode] = mean(reduce(hcat, runs), dims=2)[:]
    end

    # Average distances
    avg_distances = Dict{String, Dict{String, Vector{Float64}}}()
    for (mode, pair_dict) in all_distances
        avg_distances[mode] = Dict{String, Vector{Float64}}()
        for (pair, runs) in pair_dict
            # Find the maximum length among all runs for this pair
            maxlen = maximum(length.(runs))
            avg_vec = Float64[]
            for t in 1:maxlen
                vals = [run[t] for run in runs if length(run) >= t]
                push!(avg_vec, mean(vals))
            end
            avg_distances[mode][pair] = avg_vec
        end
    end

    return avg_path_lengths, avg_distances, reference_lengths, ref_trajs, all_distances, all_final_times
end

# Example usage:
# List your filenames here (replace with your actual file names)
filenames = [
    "TuesdayMC/dynamics_comparison_data_2025-06-16_20-29-47.jld2",
    "TuesdayMC/dynamics_comparison_data_2025-06-17_00-03-34.jld2",
    "TuesdayMC/dynamics_comparison_data_2025-06-17_15-18-00.jld2",
    "TuesdayMC/dynamics_comparison_data_2025-06-18_22-10-13.jld2",
    "TuesdayMC/dynamics_comparison_data_2025-06-19_19-25-54.jld2",
    "TuesdayMC/dynamics_comparison_data_2025-06-20_03-20-12.jld2"
]

avg_path_lengths, avg_distances, reference_lengths, ref_trajs, all_distances, all_final_times = load_and_average_results(filenames)


mode_order = [
    "Nominal-NoChance",
    "Nominal-Chance",
    "Nonlinear-NoChance",
    "Nonlinear-Chance",
    "LinearizedGP-NoChance",
    "LinearizedGP-Chance"
]
agent_labels = ["Agent 1", "Agent 2", "Agent 3"]
agent_colors = [:royalblue, :crimson, :forestgreen]
num_agents = length(agent_labels)
timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")

# --- Stacked Bar Plot for Path Lengths ---
agent_data = zeros(length(mode_order), num_agents)
for (mode_idx, mode) in enumerate(mode_order)
    for agent_idx in 1:num_agents
        agent_data[mode_idx, agent_idx] = avg_path_lengths[mode][agent_idx]
    end
end

p_lengths = plot(
    size=(800, 600),
    legend=:topleft,
    xlabel="Method",
    ylabel="Path Length"
)
groupedbar!(p_lengths, agent_data,
    bar_position=:stack,
    xticks=(1:length(mode_order), mode_order),
    xrotation=10,
    label=permutedims(agent_labels),
    color=permutedims(agent_colors),
    width=0.7,
    alpha=0.7
)
hline!([sum(reference_lengths)],
    label="Total Reference Path",
    color=:black,
    linestyle=:dash,
    alpha=0.5
)
savefig(p_lengths, "path_lengths_avg_$timestamp.pdf")
savefig(p_lengths, "path_lengths_avg_$timestamp.png")
display(p_lengths)

# --- Inter-agent Distance Plots (6 subplots, one per mode) ---
p_distances = plot(
    layout=(3,2),
    size=(1200, 900),
    left_margin=10Plots.mm
)
legend_labels = String[]
for (idx, mode_name) in enumerate(mode_order)
    subplot = p_distances[idx]
    distances = avg_distances[mode_name]
    all_pair_runs = all_distances[mode_name]  # This is Dict{String, Vector{Vector{Float64}}}
    for (pair, dist) in distances
        # Get all runs for this pair
        runs = all_pair_runs[pair]
        maxlen = maximum(length.(runs))
        # Build matrix: each column is a run, each row is a timestep (pad with NaN)
        mat = fill(NaN, maxlen, length(runs))
        for (j, run) in enumerate(runs)
            mat[1:length(run), j] .= run
        end
        # Compute mean and std, ignoring NaN
        mean_vec = [mean(skipmissing(mat[t, :])) for t in 1:maxlen]
        std_vec  = [std(skipmissing(mat[t, :])) for t in 1:maxlen]
        tvec = 1:maxlen
        # Assign a color for this pair (cycle through agent_colors or use a color map)
        color_idx = findfirst(x -> x == pair, collect(keys(distances)))
        color = agent_colors[mod1(color_idx, length(agent_colors))]
        # Plot band for ±1 std, using the same color as the line
        plot!(subplot, tvec, mean_vec .+ std_vec, fillrange=mean_vec .- std_vec,
            fillalpha=0.18, color=color, label="")
        # Plot mean line
        plot!(subplot, tvec, mean_vec,
            label="Agents $pair",
            linewidth=2,
            color=color
        )
        if idx == 1
            push!(legend_labels, "Agents $pair")
        end
    end
    hline!(subplot, [0.5],
        label=idx == 1 ? L"r_\mathrm{safe}" : "",
        color=:black,
        linestyle=:dash,
        alpha=0.5
    )
    plot!(subplot,
        title=mode_name,
        xlabel=(idx > 4 ? "Time Step" : ""),
        ylabel=(idx % 2 == 1 ? "Distance" : ""),
        legend= false,
        ylims=(0, 2),
        xlims=(20, 35),
        guidefontsize=15
    )
end
savefig(p_distances, "agent_distances_avg_$timestamp.pdf")
savefig(p_distances, "agent_distances_avg_$timestamp.png")
display(p_distances)


# Compute average final time for each mode
avg_final_times = Dict{String, Float64}()
for mode in mode_order
    if haskey(all_final_times, mode) && !isempty(all_final_times[mode])
        println(all_final_times[mode])
        avg_final_times[mode] = mean(all_final_times[mode])
    else
        avg_final_times[mode] = NaN
    end
end

final_times_vec = [avg_final_times[mode] for mode in mode_order]

p_final_times = bar(
    mode_order,
    final_times_vec,
    xlabel="Method",
    ylabel="Average Final Time",
    legend=false,
    color=:lightblue,
    width=0.7,
    xticks=(1:length(mode_order), mode_order),
    xrotation=10
)
savefig(p_final_times, "avg_final_times_$timestamp.png")
display(p_final_times)
