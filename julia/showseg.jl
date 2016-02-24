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

        "--threshold"
        help = "watershed threshold"
        arg_type = Float64
        default = 0.0
    end
    return parse_args(s)
end

function showseg()
    fname = ""
    thd = 0.0
    for pa in parse_commandline()
        if pa[1] == "fname"
            fname = pa[2]
        elseif pa[1] == "threshold"
            thd = pa[2]
        end
    end

    affs = h5read(fname, "/main")
    exchangeaffxz!(affs)
    seg = wsseg2d(affs, thd, 0.9, [(256, thd)], 100, thd)
    random_color_show(seg[:,:,1])
    show()
end

showseg()
