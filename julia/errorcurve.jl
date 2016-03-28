using ArgParse
using HDF5
using EMIRT
using PyPlot

function main()
    pd = Dict('t'=>"tag",
              'a'=>"/tmp/affs.h5",
              'l'=>"/tmp/label.h5",
              'd'=>3,
              's'=>0.1,
              'g'=>"watershed",
              'p'=>false,
              'o'=>"/tmp/error_curve.h5"
              )
    argparser!(pd)

    @assert pd['d']==2 || pd['d']==3

    # save the parameter in output error curve file
    fcurve = pd['o']
    if isfile(fcurve)
        rm(fcurve)
    end
    # tag = pd['t']
    # for (k,v) in pd
    #     k = join(k)
    #     h5write(fcurve, "/$tag/$k", v)
    # end

    # read data
    # read affinity data
    affs = EMIRT.imread( pd['a'] );
    # exchange X and Z channel
    # exchangeaffxz!(affs)

    # read label ground truth
    lbl = EMIRT.imread( pd['l'] )
    lbl = Array{UInt32,3}(lbl)

    # rand error and rand f score curve, both are foreground restricted
    print("compute error curves of affinity map ......")
    thds, segs, rf, rfm, rfs, re, rem, res = affs_error_curve(affs, lbl, pd['d'], pd['s'], pd['g'], pd['p'])
    print("done :)")

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
    if is_plot
        subplot(121)
        plot(thds, re)
        subplot(122)
        plot(thds, rf)
        show()
    end
end

main()
