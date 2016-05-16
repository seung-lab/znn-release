using Gadfly
using Distributions
using HDF5

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
                            plot(layer(x=curve["train"]["it"]/1000, y=curve["train"]["err"], color="r", Geom.line),
                                 layer(x=curve["test"]["it"] /1000, y=curve["test"]["err"],  Geom.line),
                                 Guide.xlabel("Iteration (K)"), Guide.ylabel("Cost"),
                                 Guide.title("Learning Curve of Cost"))),
        md"## Learning Curve of Pixel Error",
        drawing(8Gadfly.inch, 4Gadfly.inch,
                plot(layer(x=curve["train"]["it"]/1000, y=curve["train"]["cls"], Geom.line),
                     layer(x=curve["test"]["it"] /1000, y=curve["test"]["cls"],  color="r", Geom.line),
                     Guide.xlabel("Iteration (K)"), Guide.ylabel("Pixel Error"),
                     Guide.title("Learning Curve of Pixel Error"))),
        ) |> Escher.pad(2em)
    end
end

function plotcurve(fname::AbstractString)
    curve = get_learning_curve(fname)
    return plotcurve(curve)
end

main(window) = begin
    push!(window.assets, "widgets")

    inp = Signal(Dict())

    s = Escher.sampler()
    form = vbox(
                h1("Choose your network file"),
                watch!(s, :fname, textinput("./net_current.h5", label="network file")),
                trigger!(s, :plot, button("Plot"))
                ) |> maxwidth(400px)

    map(inp) do dict
        vbox(
             intent(s, form) >>> inp,
             vskip(2em),
             plotcurve(get(dict, :fname, "")),
             string( keys(dict) ),
             string( dict )
             ) |> Escher.pad(2em)
    end
end
