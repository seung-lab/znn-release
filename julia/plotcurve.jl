using EMIRT
using PyPlot

"""
usage:
julia plotcurve.jl tag1 fname1 tag2 fname2 tag3 fname3
"""

function parse_commandline()
    @assert length(ARGS) % 2 == 0
    argtbl = reshape(ARGS, 2,Int64( length(ARGS)/2))

    # key is tag, value is file name
    ret = Dict{ASCIIString, ASCIIString}()
    for c in 1:size(argtbl,2)
        ret[ argtbl[1,c] ] = argtbl[2,c]
    end
    return ret
end

function plotall()
    # dict of file of evaluation curves
    # key: tag, value: file name
    dtf = parse_commandline()

    # traverse the tags
    for (tag, fname) in dtf
        println("tag name: $tag")

        # read the evaluation curve file
        dec = ecread( fname )

        # plot
        c = rand(3)
        subplot(231)
        plot(dec["thds"], dec["ri"], color=c, "s-", label=tag, linewidth=2, alpha=0.5)
        xlabel("thresholds")
        ylabel("rand index")
        legend()
        println("maximum rand index: $(maximum(dec["ri"]))")

        subplot(234)
        plot(dec["rim"], dec["ris"], color=c, "s-", label=tag, linewidth=2, alpha=0.5)
        xlabel("rand index of merging")
        ylabel("rand index of splitting")


        subplot(232)
        plot(dec["thds"], dec["rf"], color=c, "s-", label=tag, linewidth=2, alpha=0.5)
        xlabel("thresholds")
        ylabel("rand F score")
        println("maximum rand F score: $(maximum(dec["rf"]))")

        subplot(235)
        plot(dec["rfm"], dec["rfs"], color=c, "s-", label=tag, linewidth=2, alpha=0.5)
        xlabel("rand F score of merging")
        ylabel("rand F score of splitting")

        subplot(233)
        plot(dec["thds"], dec["VIFS"], color=c, "s-", label=tag, linewidth=2, alpha=0.5)
        xlabel("thresholds")
        ylabel("Variation F score")
        println("maximum variation F score: $(maximum(dec["VIFS"]))")

        subplot(236)
        plot(dec["VIFSm"], dec["VIFSs"], color=c, "s-", label=tag, linewidth=2, alpha=0.5)
        xlabel("Variation F score of merging")
        ylabel("Variation F score of splitting")
    end
    show()
end

plotall()
