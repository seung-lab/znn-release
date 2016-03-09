using ArgParse
using HDF5
using EMIRT

#export affs2omni
function segm2omprj(ombin, fimg, fsegm, fomprj="/tmp/tmp.omni", vs=[4,4,40])
    # prepare the cmd file for omnification

end

function affs2omni(affs, ws_low, ws_high, fsegm)

    # watershed
    println("watershed...")
    seg, rt = watershed(affs, ws_low, ws_high);

    # get dendrogram
    println("get mst for omnifycation...")
    N = length(rt)

    dendValues = zeros(Float32, N)
    dend = zeros(UInt32, N,2)

    for i in 1:N
        t = rt[i]
        dendValues[i] = t[1]
        dend[i,1] = t[2]
        dend[i,2] = t[3]
    end

    # save result
    println("save the segments and the mst...")
    h5write(fsegm, "/dend", dend)
    h5write(fsegm, "/dendValues", dendValues)
    h5write(fsegm, "/main", seg)
end


function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--faffs"
        help = "affinity file"
        required = true
        arg_type = ASCIIString

        "--ws_low"
        help = "low threshold of watershed"
        arg_type = Float64
        default = 0.3

        "--ws_high"
        help = "high threshold of watershed"
        arg_type = Float64
        default = 0.9

        "--fsegm"
        help = "segmentation file for omnification"
        arg_type = ASCIIString
        default = "/tmp/segm.h5"
    end
    return parse_args(s)
end

function main()
    faffs = ""
    ws_low = 0.3
    ws_high = 0.9
    fsegm = "/tmp/segm.h5"
    for pa in parse_commandline()
        if pa[1] == "faffs"
            faffs = pa[2]
        elseif pa[1] == "ws_low"
            ws_low = pa[2]
        elseif pa[1] == "ws_high"
            ws_high = pa[2]
        elseif pa[1] == "fsegm"
            fsegm = pa[2]
        end
    end

    # read affinity
    println("read affinity map...")
    affs = h5read(faffs, "/main");

    # make the affinity map value distribution uniform
    println("transfer to uniform distribution...")
    for z in 1:size(affs,3)
        println("section: $z")
        affs[:,:,z,:] = aff2uniform( affs[:,:,z,:] )
    end
    #affsuniform!(affs)
    # get segm file for omnification
    affs2omni(affs, ws_low, ws_high, fsegm)
end

main()
