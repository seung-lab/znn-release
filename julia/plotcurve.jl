using ArgParse
using HDF5
using EMIRT
using PyPlot

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--fname"
        help = "error curve file name"
        arg_type = ASCIIString
        default = "/tmp/error_curve.h5"

        "--tag"
        help = "curve name"
        arg_type= ASCIIString
        default = "all"
    end
    return parse_args(s)
end

function plotall()
    fname = ""
    stag = ""
    for pa in parse_commandline()
        if pa[1] == "fname"
            fname = pa[2]
        elseif pa[1] == "tag"
            stag = pa[2]
        end
    end

    # traverse the tags
    f = h5open(fname, "r")
    for tag in names(f)
        if stag != "all" && tag != stag
            continue
        end

        println("tag name: $tag")
        # every error curve
        thds = read( f[tag]["thds"] )
        re   = read( f[tag]["re"]   )
        rem  = read( f[tag]["rem"]  )
        res  = read( f[tag]["res"]  )
        rf   = read( f[tag]["rf"]   )
        rfm  = read( f[tag]["rfm"]  )
        rfs  = read( f[tag]["rfs"]  )

        # plot
        c = rand(3)
        subplot(221)
        plot(thds, re, color=c, "s-", label=tag, linewidth=2, alpha=0.5)
        xlabel("thresholds")
        ylabel("rand error")
        legend()

        subplot(222)
        plot(rem, res, color=c, "s-", label=tag, linewidth=2, alpha=0.5)
        xlabel("rand error merger")
        ylabel("rand error splitter")


        subplot(223)
        plot(thds, rf, color=c, "s-", label=tag, linewidth=2, alpha=0.5)
        xlabel("thresholds")
        ylabel("rand f score")

        subplot(224)
        plot(rfm, rfs, color=c, "s-", label=tag, linewidth=2, alpha=0.5)
        xlabel("rand f score merger")
        ylabel("rand f score splitter")

    end
    show()
    close(f)
end

plotall()
