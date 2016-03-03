using ArgParse
using HDF5
using EMIRT
using PyPlot

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--tag"
        help = "curve type name"
        required = true
        arg_type = ASCIIString

        "--flbl"
        help = "label file name"
        required = true
        arg_type = ASCIIString

        "--faffs"
        help = "affinity file names, could be multiple files to compare"
        required = true
        arg_type = ASCIIString

        "--step"
        help = "error curve step"
        arg_type = Float64
        default = 0.1

        "--fcurve"
        help = "file to save the error curve"
        arg_type = ASCIIString
        default = "/tmp/error_curve.h5"

        "--seg_method"
        help = "segmentation method: connected_component; watershed"
        arg_type = ASCIIString
        default = "connected_component"

        "--isplot"
        help = "whether plot the curve or not"
        arg_type = Bool
        default = true
    end
    return parse_args(s)
end

function main()
    # read command line parameters
    faffs = ""
    flbl = ""
    tag = ""
    step = 0.1
    seg_method = ""
    fcurve = ""
    isplot = true
    for pa in parse_commandline()
        if pa[1] == "tag"
            tag = pa[2]
        elseif pa[1] == "flbl"
            flbl = pa[2]
        elseif pa[1] == "faffs"
            faffs = pa[2]
        elseif pa[1] == "step"
            step = pa[2]
        elseif pa[1] == "seg_method"
            seg_method = pa[2]
        elseif pa[1] == "fcurve"
            fcurve = pa[2]
        elseif pa[1] == "isplot"
            isplot = pa[2]
        end
    end

    # read data
    # read affinity data
    affs = h5read(faffs, "/main");
    # exchange X and Z channel
    exchangeaffxz!(affs)

    # read label ground truth
    lbl = h5read(flbl, "/main")

    # rand error and rand f score curve, both are foreground restricted
    thds, segs, rf, rfm, rfs, re, rem, res = affs_error_curve(affs, lbl, 2, 0.1, seg_method)

    # save the curve
    h5write(fcurve, "/$tag/segs", segs)
    h5write(fcurve, "/$tag/thds", thds)
    h5write(fcurve, "/$tag/rf",   rf )
    h5write(fcurve, "/$tag/rfm",  rfm )
    h5write(fcurve, "/$tag/rfs",  rfs )
    h5write(fcurve, "/$tag/re",   re )
    h5write(fcurve, "/$tag/rem",  rem )
    h5write(fcurve, "/$tag/res",  res )

    # plot
    subplot(121)
    plot(thds, re)
    subplot(122)
    plot(thds, rf)
    show()
end

main()
