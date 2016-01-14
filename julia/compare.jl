using ArgParse
using HDF5
using EMIRT

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--flbl"
        help = "label file name"
        required = true

        "--faffs"
        help = "affinity file names, could be multiple files to compare"
        required = true

        "--step"
        help = "error curve step"
        arg_type = Float64
        default = 0.1
    end
    return parse_args(s)
end

#function evaluate_aff_file(flbl, faffs)


# parameters
#faff = "/znn/experiments/malis21/N4_A_1000/out_sample91_output.h5"
#faff = "/znn/experiments/zfish/N4_A/out_sample91_output.h5"
#faff = "/znn/experiments/malis2/N4_A_100/out_sample91_output.h5"
faff = "/znn/experiments/malis2/srini2d_A_L10/out_sample91_output.h5"
flbl = "/znn/dataset/zfish/Merlin_lbl2.h5"

# read affinity data
affs = h5read(faff, "/main");

#seg = aff2seg(affs,2)
#using PyCall
#@pyimport emirt.show as emshow
#emshow.random_color_show(seg[:,:,1])

# read label ground truth
lbl = h5read(flbl, "/main")

# rand error curve
thds, rfs, res = affs_error_curve(affs, lbl, 2, 0.05)

# save curve
fcurve = "/znn/experiments/curves.h5"
h5write(fcurve, "/srini2d/malis2_2/thds", thds)
h5write(fcurve, "/srini2d/malis2_2/rfs",  rfs )
h5write(fcurve, "/srini2d/malis2_2/res",  res )

# plot
using PyPlot
subplot(121)
plot(thds, res)
subplot(122)
plot(thds, rfs)
show()
