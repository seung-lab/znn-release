using HDF5

# get the group and field name
function get_group_field(str::String)
    sl = split(str, '/')
    field = pop!(sl)
    group = pop!(sl)
    return group, field
end

function load_opts(fname)
    f = h5open(fname, "r")
    nodes = []
    edges = []
    for layer in f
        # dict of layer
        dly = Dict()
        for obj in layer
            group, field = get_group_field( name(obj) )
            dly[field] = read(obj)
        end

        if dly["group_type"] == "node"
            push!(nodes, dly)
        else
            push!(edges, dly)
        end
    end
    close(f)
    return nodes, edges
end



fname = "../../experiments/net_current.h5"
nodes, edges = load_opts(fname)

println("nodes and edges:")
println(nodes)
println(edges)
