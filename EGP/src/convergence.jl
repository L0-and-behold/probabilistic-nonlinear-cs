
using Statistics
using DiffEqFlux: collocate_data, EpanechnikovKernel
using Plots, Revise
plotlyjs()

function is_saturated(mylog::Vector, smoothing_window::Int; plot_graph::Bool = false, min_possible_deviation=1e-10, return_plot::Bool=false, twosided=false)
    if length(mylog) < smoothing_window + 2
        return false
    else
        # divide the smoothing window in 2 parts:
        th = cld(smoothing_window,2) # cld(a,b) outputs the smallest integer bigger or equal to a/b
        
        # convert the mylogs in those two parts into matrices:
        w1 = reshape(mylog[end-2*th+1:end-th], 1, th)
        w2 = reshape(mylog[end-th+1:end], 1, th)
        
        # fit smooth curves to w1 and w2:
        _, u1 = collocate_data(w1, 1:th, EpanechnikovKernel())
        _, u2 = collocate_data(w2, 1:th, EpanechnikovKernel())
        # other kernels: https://docs.sciml.ai/DiffEqFlux/stable/utilities/Collocation/
        # example: https://docs.sciml.ai/DiffEqFlux/stable/examples/collocation/
        
        # compute mean and standard deviation:
        m1 = mean(w1)
        m2 = mean(w2)
        st1 = std(w1 .- u1)
        st2 = std(w2 .- u2)

        if plot_graph
            xs = collect(1:length(mylog))
            xss = xs[end-2*th+1:end]
            mx = [xss[argmin(abs.(u1 .- m1))[2]], xss[th + argmin(abs.(u2 .- m2))[2]]]
            p = Plots.plot(xs,mylog)
            p = Plots.plot(xss,mylog[end-2*th+1:end])
            p = Plots.scatter!(mx,[m1,m2])
            p = Plots.plot!(xss[end-2*th+1:end-th],[u1', (u1 .+ (st1 + min_possible_deviation))', (u1 .- (st1 + min_possible_deviation))'], legend=false)
            p = Plots.plot!(xss[end-th+1:end],[u2', (u2 .+ (st2 + min_possible_deviation))', (u2 .- (st2 + min_possible_deviation))'], legend=false)
            c1 = fill(m1-(st1 + min_possible_deviation),(2*th,1))
            c2 = fill(m2+(st2 + min_possible_deviation),(2*th,1))
            p = Plots.plot!(xss[end-2*th+1:end],c1,ls=:dash, legend=false)
            p = Plots.plot!(xss[end-2*th+1:end],c2,ls=:dash, legend=false)
        end

        # for batch processing: return the plot rather than displaying it
        if return_plot
            return p
        elseif plot_graph
            display(p)
        end

        # if abs(m1-m2) < st1 + st2
        # if m1 - st1 < m2 && m1 < m2 + st2
        if twosided
            criterion = abs(m1-m2)
        else
            criterion = m1-m2
        end
        if criterion <= min(st1, st2) + min_possible_deviation # for twosided, this corresponds to 
        # if m1 - st1 < m2 && m2 < m1 + st1 && m2 - st2 < m1 && m1 < m2 + st2
            return true
        else
            return false
        end
    end

end
