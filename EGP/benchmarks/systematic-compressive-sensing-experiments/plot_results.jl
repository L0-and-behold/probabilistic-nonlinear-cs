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
SNRs = logspace(0.05, 20, 10)
rhos = Float32.([0, 0.35, 0.7, 0.9])

method_names = ["EGP", "Forward Stepwise", "Relaxed Lasso", "Lasso", "IHT"]
repetitions = 10
repetitions = 2

markershapes = [:circle,:utriangle,:rect,:diamond,:xcross]
color_names = [:blue, :green, :orange, :violet, :red]
linestyles = [:solid, :dash, :dot, :dashdot, :solid]


function filter_results(rho, results_for_setting_and_method)
    return filter(entry -> entry[2].setting.rho == rho, results_for_setting_and_method)
end

absolute_dataPath = joinpath(dirname(@__DIR__),"results/")
date_string = "2025-08-31"

function plot_metric_interquartile(rho, metric, setting, absolute_dataPath, date_string, metric2=:ASRE; ylims=(1.0, 1.2), path_addendum = "", ylims2=:auto, repetitions=10, method_names = ["EGP", "Lasso"], width=800, height=600, plot_mean=true, save_fig=false, plot_ribbon=true, plot_error=true, figname_addendum = "", legend=true)

    ticks = [(SNRs[i]>20 ? round(Int,SNRs[i]) : round(SNRs[i], digits=2)) for i in 1:2:length(SNRs)]
    
    formatter(x) = begin
        if x >= 10000
            @sprintf("%.0fk", x/1000)
        elseif x >= 1000
            k_val = x/1000
            k_val == round(k_val) ? @sprintf("%.0fk", k_val) : @sprintf("%.1fk", k_val)
        elseif x > 20 
            @sprintf("%d", round(Int, x))
        elseif x > 10 
            @sprintf("%.1f", round(x,digits=1))
        else
            @sprintf("%.2f", x)
        end
    end
    
    println("1")
    if legend
        p1 = Plots.plot(title="", xscale=:log10, legend=(0.68, 0.98), size=(width,height), ylim=ylims, xticks=ticks, xformatter=formatter, width=width, height=height, legendfontsize=14, tickfontsize=14, 
        right_margin=1Plots.cm, framestyle=:box)
    else
        p1 = Plots.plot(title="", xscale=:log10, legend=false, size=(width,height), ylim=ylims, xticks=ticks, xformatter=formatter, width=width, height=height, legendfontsize=14, tickfontsize=14, 
        right_margin=1Plots.cm, framestyle=:box)
    end
    println("2")
    if plot_ribbon
        for (method_name, markershape, color_name) in zip(method_names, markershapes, color_names)

            results = load(absolute_dataPath * "results_" * date_string * path_addendum * "_setting_" * string(setting) * "_method_" * method_name * ".jld2")

            fr = filter_results(rho, results)

            value_matrix = Array{Float32}(undef, repetitions, length(SNRs))
            value_matrix2 = Array{Float32}(undef, repetitions, length(SNRs))
            for (i,nu) in enumerate(SNRs)
                fr_snr = filter(entry -> entry[2].setting.nu == nu, fr)
                value_matrix[:,i] .= [v[metric] for v in values(fr_snr)]
                value_matrix2[:,i] .= [v[metric2] for v in values(fr_snr)]
            end

            if plot_mean
                means = Statistics.mean(value_matrix, dims=1)'
                stds = Statistics.std(value_matrix, dims=1)'

                # 2d ribbon
                p1 = plot!(p1, SNRs, means, ribbon = stds, label="", z_order=1, color=color_name, fillalpha=0.2)
            else
                medians = Statistics.median(value_matrix, dims=1)'
                q1s = mapslices(x -> quantile(x, 0.3), value_matrix, dims=1)'
                q3s = mapslices(x -> quantile(x, 0.7), value_matrix, dims=1)'

                # Asymmetric ribbon: (lower_error, upper_error)
                lower_errors = medians .- q1s  # distance from median down to Q1
                upper_errors = q3s .- medians  # distance from median up to Q3

                p1 = plot!(p1, SNRs, medians, ribbon = (lower_errors, upper_errors), label="", z_order=1, color=color_name, fillalpha=0.2)
            end
        end
    end
    println("3")
    if plot_error
        for (method_name, markershape, color_name) in zip(method_names, markershapes, color_names)

            results = load(absolute_dataPath * "results_" * date_string * path_addendum * "_setting_" * string(setting) * "_method_" * method_name * ".jld2")

            fr = filter_results(rho, results)

            value_matrix = Array{Float32}(undef, repetitions, length(SNRs))
            value_matrix2 = Array{Float32}(undef, repetitions, length(SNRs))
            for (i,nu) in enumerate(SNRs)
                fr_snr = filter(entry -> entry[2].setting.nu == nu, fr)
                value_matrix[:,i] .= [v[metric] for v in values(fr_snr)]
                value_matrix2[:,i] .= [v[metric2] for v in values(fr_snr)]
            end

            if plot_mean
                means = Statistics.mean(value_matrix, dims=1)'
                stds = Statistics.std(value_matrix, dims=1)'
                p1 = plot!(p1, SNRs, means, yerr=stds, label="", markershape=markershape, color=color_name, lw=3, markerstrokecolor=color_name)
            else
                medians = Statistics.median(value_matrix, dims=1)'
                q1s = mapslices(x -> quantile(x, 0.3), value_matrix, dims=1)'
                q3s = mapslices(x -> quantile(x, 0.7), value_matrix, dims=1)'

                # Asymmetric ribbon: (lower_error, upper_error)
                lower_errors = medians .- q1s  # distance from median down to Q1
                upper_errors = q3s .- medians  # distance from median up to Q3

                p1 = plot!(p1, SNRs, medians, yerr=(lower_errors, upper_errors), label="", markershape=markershape, color=color_name, lw=3, markerstrokecolor=color_name)
            end
        end
    end
    println("4")
    for (method_name, markershape, color_name) in zip(method_names, markershapes, color_names)
        
        results = load(absolute_dataPath * "results_" * date_string * path_addendum * "_setting_" * string(setting) * "_method_" * method_name * ".jld2")

        fr = filter_results(rho, results)

        value_matrix = Array{Float32}(undef, repetitions, length(SNRs))
        value_matrix2 = Array{Float32}(undef, repetitions, length(SNRs))
        for (i,nu) in enumerate(SNRs)
            fr_snr = filter(entry -> entry[2].setting.nu == nu, fr)
            value_matrix[:,i] .= [v[metric] for v in values(fr_snr)]
            value_matrix2[:,i] .= [v[metric2] for v in values(fr_snr)]
        end

        if plot_mean
            means = Statistics.mean(value_matrix, dims=1)'
            p1 = plot!(p1, SNRs, means, markershape=markershape, label=method_name, lw=3, z_order=2, color=color_name, markersize=6)
        else
            medians = Statistics.median(value_matrix, dims=1)'

            if metric==:ASRE
                if method_name == "EGP"
                    markersize = 10
                elseif method_name == "Forward Stepwise"
                    markersize = 8
                elseif method_name == "Relaxed Lasso" || method_name == "Lasso"
                    markersize = 6
                elseif method_name == "IHT"
                    markersize = 5
                end
                lw = 4
            else
                markersize = 6
                lw = 3
            end

            if !plot_error && !plot_ribbon
                p1 = plot!(p1, SNRs, medians, markershape=markershape, label="", lw=lw+0.5, color=:black, markersize=markersize)
                p1 = plot!(p1, SNRs, medians, markershape=markershape, label=method_name, lw=lw, color=color_name, markersize=markersize)
            else
                p1 = plot!(p1, SNRs, medians, markershape=markershape, label="", lw=lw+0.5, z_order=2, color=:black, markersize=markersize)
                p1 = plot!(p1, SNRs, medians, markershape=markershape, label=method_name, lw=lw, z_order=2, color=color_name, markersize=markersize)
            end
        end
    end
    println("5")
    if metric==:RTE
        snr_points = logspace(SNRs[1], SNRs[end], 1000)
        zero_fit_line = snr_points .+ 1f0
        p1 = plot!(p1,snr_points, zero_fit_line, linecolor=:black, linestyle=:dot, label="β = 0", lw=3)
    end

    println("6")
    pf = Plots.plot(p1, 
        size=(width,height), 
        titlefontsize = 18,
    )
    println("7")
    display(pf)
    println("8")
    if save_fig
        Plots.savefig(pf, String(metric) * " s=" * string(setting) * " rho=" * replace(string(rho), "." => "") * figname_addendum * ".pdf")
    end
    println("9")
end

plot_metric_interquartile(rhos[1], :RTE, 1, absolute_dataPath, date_string; ylims=(1.0, 1.2), repetitions=2, method_names = method_names, width=800, height = 600, plot_mean=false, plot_ribbon=true, plot_error=false, save_fig=true, figname_addendum = "", legend=true)