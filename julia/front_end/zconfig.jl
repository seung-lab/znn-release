using ConfParser

function getboolean(str)
    if "yes" == str
        return true
    elseif "no" == str
        return false
    end
end

function getints(str)
    ret = {}
    s1 = split(str, ',')
    for s in s1
        lst = split(s,'-')
        for l in lst
            i = parseint(l)
            push!(ret, i)
        end
    end
    return ret
end


function configparse( fname )
    conf = ConfParse(fname)
    parse_conf!(conf)

    pars = Dict()
    pars["fnet_spec"]   = retrieve(conf, "parameters", "fnet_spec")
    pars["num_threads"] = parseint( retrieve(conf, "parameters", "num_threads") )
    pars["fdata_spec"]  = retrieve(conf, "parameters", "fdata_spec")
    pars["dtype"]       = retrieve(conf, "parameters", "dtype")
    pars["out_type"]    = retrieve(conf, "parameters", "out_type")

    pars["is_forward_optimize"] = retrieve(conf, "parameters", "is_forward_optimize")
    pars["force_fft"]   = getboolean( retrieve(conf, "parameters", "force_fft") )
    pars["is_bd_mirror"]= getboolean( retrieve(conf, "parameters", "is_bd_mirror") )
    pars["is_stdio"]    = getboolean( retrieve(conf, "parameters", "is_stdio") )
    pars["is_debug"]    = getboolean( retrieve(conf, "parameters", "is_debug") )

    pars["forward_range"] = getints( retrieve(conf, "parameters", "forward_range") )
    pars["forward_net"]   = retrieve(conf, "parameters", "forward_net")
    pars["forward_outsz"] = getints( retrieve(conf, "parameters", "forward_outsz") )
    pars["output_prefix"] = retrieve(conf, "parameters", "output_prefix")
    return pars
end

# print the pars
pars = configparse("../../python/config.cfg")
println("config parameters: ")
println(pars)
