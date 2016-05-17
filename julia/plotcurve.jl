using Gadfly
using Distributions
using HDF5
import Escher: Sampler

function get_learning_curve(fname::AbstractString)
    if contains(fname, "s3://")
        # local file name
        lcfname = "/tmp/net_current.h5"
        # download from  AWS S3
        run(`aws s3 cp $(fname) $(lcfname)`)
        # rename fname to local file name
        fname = lcfname
    end
    curve = Dict{ASCIIString, Dict{ASCIIString,Vector}}()
    if isfile(fname)
        curve["train"] = Dict{ASCIIString, Vector}()
        curve["test"]  = Dict{ASCIIString, Vector}()

        curve["train"]["it"]  = h5read(fname, "/processing/znn/train/statistics/train/it")
        curve["train"]["err"] = h5read(fname, "/processing/znn/train/statistics/train/err")
        curve["train"]["cls"] = h5read(fname, "/processing/znn/train/statistics/train/cls")
        curve["test"]["it"]   = h5read(fname, "/processing/znn/train/statistics/test/it")
        curve["test"]["err"]  = h5read(fname, "/processing/znn/train/statistics/test/err")
        curve["test"]["cls"]  = h5read(fname, "/processing/znn/train/statistics/test/cls")
    end
    return curve
end

function plotcurve(curve::Dict)
    if length( keys(curve) ) == 0
        return ""
    else
        return vbox(
                    md"## Learning Curve of Cost",
                    drawing(8Gadfly.inch, 4Gadfly.inch,
                            plot(layer(x=curve["train"]["it"]/1000, y=curve["train"]["err"], Geom.line, Theme(default_color=color("blue"))),
                                 layer(x=curve["test"]["it"] /1000, y=curve["test"]["err"],  Geom.line, Theme(default_color=color("red"))),
                                 Guide.xlabel("Iteration (K)"), Guide.ylabel("Cost"))),
                                 # Guide.title("Learning Curve of Cost"))),
        md"## Learning Curve of Pixel Error",
        drawing(8Gadfly.inch, 4Gadfly.inch,
                plot(layer(x=curve["train"]["it"]/1000, y=curve["train"]["cls"], Geom.line, Theme(default_color=color("blue"))),
                     layer(x=curve["test"]["it"] /1000, y=curve["test"]["cls"],  Geom.line, Theme(default_color=color("red"))),
                     Guide.xlabel("Iteration (K)"), Guide.ylabel("Pixel Error"))) #,
                #Guide.title("Learning Curve of Pixel Error"))),
        ) |> Escher.pad(2em)
    end
end

function plotcurve(fname::AbstractString)
    curve = get_learning_curve(fname)
    return plotcurve(curve)
end

"""
the form tile to provide learning curve plotting tile
"""
function tile_form(inp::Signal, s::Sampler)
    return vbox(
                h1("Choose your network file"),
                watch!(s, :fname, textinput("/tmp/net_current.h5", label="network file")),
                trigger!(s, :plot, button("Plot Learning Curve"))
                ) |> maxwidth(400px)
end

"""
get learning curve plotting tile
`Parameters`:
inp: input
s: sampler

`Return`:
ret: learning curve plotting tile
"""
function plotcurve()
    inp = Signal(Dict())
    s = Escher.sampler()
    form = tile_form(inp, s)
    ret = consume(inp) do dict
        vbox(
                intent(s, form) >>> inp,
                vskip(2em),
                plotcurve(get(dict, :fname, ""))
                ) |> Escher.pad(2em)
    end
    return ret
end
