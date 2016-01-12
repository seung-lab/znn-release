using HDF5
using EMIRT

# parameters
#faff = "/znn/experiments/malis21/N4_A_1000/out_sample91_output.h5"
#faff = "/znn/experiments/zfish/N4_A/out_sample91_output.h5"
faff = "/znn/experiments/malis2/N4_A_100/out_sample91_output.h5"
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
thds, res = affs_rand_error_curve(affs, lbl, 0.05, 2)

# save curve
h5write(faff, "/evaluate/thds", thds)
h5write(faff, "/evaluate/res",  res )

# plot
using PyPlot
plot(thds, res)
show()
