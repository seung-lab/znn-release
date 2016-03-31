using HDF5
using EMIRT
using PyPlot

function main()
    pd = Dict('t'=>"tag",
              'a'=>"/tmp/affs.h5",  # affinity file
              'l'=>"/tmp/label.h5", # label file
              'd'=>3,               # dimension
              's'=>0.1,             # threshold step
              'g'=>"watershed",     # segmentation method
              'p'=>false,           # path or not
              'o'=>""               # output file name
              )
    argparser!(pd)

    # assersions
    @assert isfile(pd['a'])
    @assert isfile(pd['l'])
    @assert pd['d']==2 || pd['d']==3

    # save the parameter in output error curve file
    fcurve = pd['o']
    if fcurve == ""
        fcurve = pd['a']
        fcurve = replace(fcurve, ".tif", ".h5")
    elseif isfile(fcurve)
        rm(fcurve)
    end

    tag = pd['t']
    for (k,v) in pd
        k = join(k)
        if v == true
            h5write(fcurve, "/processing/znn/forward/stage_2/sample_91/evaluate_params/$k", "true")
        elseif v==false
            h5write(fcurve, "/processing/znn/forward/stage_2/sample_91/evaluate_params/$k", "false")
        else
            h5write(fcurve, "/processing/znn/forward/stage_2/sample_91/evaluate_params/$k", v)
        end
    end

    # read data
    # read affinity data
    affs = EMIRT.imread( pd['a'] );

    # read label ground truth
    lbl = EMIRT.imread( pd['l'] )
    lbl = Array{UInt32,3}(lbl)

    # rand error and rand f score curve, both are foreground restricted
    print("compute error curves of affinity map ......")
    # dictionary of scores
    scd = affs_error_curve(affs, lbl, pd['d'], pd['s'], pd['g'], pd['p'])
    print("done :)")

    tag = pd['t']
    # save the curve
    for (k,v) in scd
        h5write(fcurve, "/processing/znn/forward/stage_2/sample_91/evaluate_curve/$k", v)
    end
end

main()
