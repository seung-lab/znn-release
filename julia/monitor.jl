using Gadfly
using Distributions
using HDF5

function get_learning_curve(fname)
    # if "s3://" in fname
    #     # local file name
    #     lcfname = "./net_current.h5"
    #     # download from  AWS S3
    #     run(`aws s3 cp $(fname) $(lcfname)`)
    #     # rename fname to local file name
    #     fname = lcfname
    # end
    it  = h5read(fname, "/processing/znn/train/statistics/train/it")
    err = h5read(fname, "/processing/znn/train/statistics/train/err")
    cls = h5read(fname, "/processing/znn/train/statistics/train/cls")
    return it, err, cls
end

function plotcurve(fname)
    it, err, cls = get_learning_curve(fname)
    cvplot = vbox(md"## Learning Curve of Cost",
                  drawing(8Gadfly.inch, 4Gadfly.inch,
                          plot(x=it/1000, y=err, Geom.line,
                               Guide.xlabel("Iteration (K)"), Guide.ylabel("Cost"),
                               Guide.title("Learning Curve of Cost"))),
                  md"## Learning Curve of Pixel Error",
                  drawing(8Gadfly.inch, 4Gadfly.inch,
                          plot(x=it/1000, y=cls, Geom.line,
                               Guide.xlabel("Iteration (K)"), Guide.ylabel("Pixel Error"),
                               Guide.title("Learning Curve of Pixel Error")))
                  ) |> Escher.pad(2em)
    return cvplot
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
             plotcurve("./net_current.h5"),
             get(dict, :fname, ""),
             string( keys(dict) ),
             string( dict )
             ) |> Escher.pad(2em)
    end
end
