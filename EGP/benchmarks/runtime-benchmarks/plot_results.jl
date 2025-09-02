using JLD2
using Plots
using Statistics
using Printf

plotlyjs()

function logspace(start_value, end_value, num_points; digits=2)
    log_start = log10(start_value)
    log_end = log10(end_value)
    log_range = range(log_start, stop=log_end, length=num_points)
    values = Float32.(round.(10 .^ log_range, digits=digits))
    return values
end

SNRs = logspace(0.05, 60000, 20)
SNRs = SNRs[17]
rhos = Float32.([0, 0.35, 0.7, 0.9])
rhos = rhos[1]

method_names = ["EGP", "Forward Stepwise", "Relaxed Lasso", "Lasso", "IHT"]
repetitions = 10
markershapes = [:circle, :utriangle, :rect, :diamond, :xcross]
color_names = [:blue, :green, :orange, :violet, :red]
linestyles = [:solid, :dash, :dot, :dashdot, :solid]

settings = [
    (ids = 1, n = 100, p = 100, s1 = 10, s2 = 1000000),
    (ids = 1, n = 1000, p = 100, s1 = 10, s2 = 1000000),
    (ids = 1, n = 2000, p = 100, s1 = 10, s2 = 1000000),
    (ids = 1, n = 3000, p = 100, s1 = 10, s2 = 1000000),
    (ids = 1, n = 4000, p = 100, s1 = 10, s2 = 1000000),
    (ids = 1, n = 5000, p = 100, s1 = 10, s2 = 1000000),
    (ids = 1, n = 6000, p = 100, s1 = 10, s2 = 1000000),
    (ids = 1, n = 7000, p = 100, s1 = 10, s2 = 1000000),
    (ids = 1, n = 8000, p = 100, s1 = 10, s2 = 1000000),
    (ids = 1, n = 9000, p = 100, s1 = 10, s2 = 1000000),
    (ids = 1, n = 10000, p = 100, s1 = 10, s2 = 1000000),
    (ids = 1, n = 11000, p = 100, s1 = 10, s2 = 1000000),
    (ids = 2, n = 100, p = 100, s1 = 10, s2 = 1000000),
    (ids = 2, n = 100, p = 1000, s1 = 10, s2 = 1000000),
    (ids = 2, n = 100, p = 2000, s1 = 10, s2 = 1000000),
    (ids = 2, n = 100, p = 3000, s1 = 10, s2 = 1000000),
    (ids = 2, n = 100, p = 4000, s1 = 10, s2 = 1000000),
    (ids = 2, n = 100, p = 5000, s1 = 10, s2 = 1000000),
    (ids = 2, n = 100, p = 6000, s1 = 10, s2 = 1000000),
    (ids = 2, n = 100, p = 7000, s1 = 10, s2 = 1000000),
    (ids = 2, n = 100, p = 8000, s1 = 10, s2 = 1000000),
    (ids = 2, n = 100, p = 9000, s1 = 10, s2 = 1000000),
    (ids = 2, n = 100, p = 10000, s1 = 10, s2 = 1000000),
    (ids = 2, n = 100, p = 11000, s1 = 10, s2 = 1000000),
]

absolute_dataPath = joinpath(dirname(@__DIR__),"results/")
date_string = "2025-09-01"

function plot_runtimes_interquartile(setting_id, absolute_dataPath, date_string; path_addendum = "_timeBenchmarks", settings = settings, repetitions=10, method_names = ["EGP", "Forward Stepwise", "Relaxed Lasso", "Lasso", "IHT"], width=800, height=600, plot_mean=false, save_fig=false, plot_ribbon=true, figname_addendum = "", legend=true, xscale=:identity, yscale=:identity, yticks=:auto)

    filtered_settings = filter(entry -> entry.ids == setting_id, settings)
    if setting_id == 1
        x_values = [set.n for set in filtered_settings]
    elseif setting_id == 2
        x_values = [set.p for set in filtered_settings]
    end
    ticks = x_values
    formatter(x) = begin
        if x >= 1000
            @sprintf("%.0fk", x/1000)
        else
            @sprintf("%.0f", x)
        end
    end

    println("1")
    if legend
        p1 = Plots.plot(title="", legend=:topleft, size=(width,height), xticks=ticks, xformatter=formatter, width=width, height=height, legendfontsize=14, tickfontsize=14, 
        right_margin=1Plots.cm, framestyle=:box, xscale=xscale, yscale=yscale, yticks=yticks)
    else
        p1 = Plots.plot(title="", legend=false, size=(width,height), xticks=ticks, xformatter=formatter, width=width, height=height, legendfontsize=14, tickfontsize=14, 
        right_margin=1Plots.cm, framestyle=:box, xscale=xscale, yscale=yscale, yticks=yticks) 
    end
    println("2")
    if plot_ribbon
        medians_EGP = Array{Float32}(undef, length(x_values), 1)
        medians_RL = Array{Float32}(undef, length(x_values), 1)
        for (method_name, markershape, color_name) in zip(method_names, markershapes, color_names)
            value_matrix = Array{Float32}(undef, repetitions, length(x_values))
            for (i,set) in enumerate(filtered_settings)
                
                fr = load(absolute_dataPath * "results_" * date_string * path_addendum * "_setting_" * string(setting_id) * "_method_" * method_name * "_n_" * string(set.n) * "_p_" * string(set.p) * ".jld2")

                value_matrix[:,i] .= [v.total_time for v in values(fr)]

            end
            if plot_mean
                means = Statistics.mean(value_matrix, dims=1)'
                stds = Statistics.std(value_matrix, dims=1)'


                # 2d ribbon
                p1 = plot!(p1, x_values, means, ribbon = stds, label="", z_order=1, color=color_name, fillalpha=0.2)
            else
                medians = Statistics.median(value_matrix, dims=1)'
                q1s = mapslices(x -> quantile(x, 0.3), value_matrix, dims=1)'
                q3s = mapslices(x -> quantile(x, 0.7), value_matrix, dims=1)'


                if method_name == "EGP"
                    medians_EGP .= medians
                elseif method_name == "Relaxed Lasso"
                    medians_RL .= medians
                end

                # Asymmetric ribbon: (lower_error, upper_error)
                lower_errors = medians .- q1s  # distance from median down to Q1
                upper_errors = q3s .- medians  # distance from median up to Q3

                p1 = plot!(p1, x_values, medians, ribbon = (lower_errors, upper_errors), label="", z_order=1, color=color_name, fillalpha=0.2)
            end
        end
        medianFactor = medians_EGP ./ medians_RL
        medianFactorMean = Statistics.mean(medianFactor)
        medianFactorStd = Statistics.std(medianFactor)
        println("EGP/RL mean: ",medianFactorMean,"\nEGP/RL std: ",medianFactorStd)
    end
    for (method_name, markershape, color_name) in zip(method_names, markershapes, color_names)
        value_matrix = Array{Float32}(undef, repetitions, length(x_values))
        value_matrix2 = Array{Float32}(undef, repetitions, length(x_values))
        for (i,set) in enumerate(filtered_settings)
            
            fr = load(absolute_dataPath * "results_" * date_string * path_addendum * "_setting_" * string(setting_id) * "_method_" * method_name * "_n_" * string(set.n) * "_p_" * string(set.p) * ".jld2")

            value_matrix[:,i] .= [v.total_time for v in values(fr)]
            value_matrix2[:,i] .= [v.RTE for v in values(fr)]

        end
        if plot_mean
            means = Statistics.mean(value_matrix, dims=1)'
            p1 = plot!(p1, x_values, means, markershape=markershape, lw=3, label=method_name, z_order=2, color=color_name, markersize=6)
        else
            medians = Statistics.median(value_matrix, dims=1)'
            p1 = plot!(p1, x_values, medians, markershape=markershape, lw=3, label=method_name, z_order=2, color=color_name, markersize=6)
        end
    end
    pf = Plots.plot(p1, 
        size=(width,height), 
        titlefontsize = 18,
    )
    display(pf)
    if save_fig
        Plots.savefig(pf, "benchmarks_" * " s_" * string(setting_id) * figname_addendum * ".pdf")
    end
end

plot_runtimes_interquartile(1, absolute_dataPath, date_string; method_names = method_names, repetitions=10, yscale=:identity, save_fig=false, yticks=0:25:300)
plot_runtimes_interquartile(2, absolute_dataPath, date_string; method_names = method_names, repetitions=10, yscale=:identity, save_fig=false, yticks=0:25:300)