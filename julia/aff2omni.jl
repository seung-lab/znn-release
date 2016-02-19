using HDF5
using EMIRT

# parameters
# sample id
sid = 8

# watershed thresholds
ws_low = 0.2
ws_high = 0.9

# read affinity
println("read affinity map...")
affs = h5read("/znn/experiments/allen/W3/out_sample$sid" * "_output.h5", "/main");
exchangeaffxz!(affs)

# make the affinity map value distribution uniform
println("transform to uniform distribution")
affs = uniform_transformation( affs )

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
    dend[i,2] = t[2]
    dend[i,1] = t[3]
end

# save result
println("save the segments and the mst...")
fsegm = "sample$sid" * ".segm.h5"
h5write(fsegm, "/dend", dend)
h5write(fsegm, "/dendValues", dendValues)
h5write(fsegm, "/main", seg)
