using Gadfly
using HDF5
import Escher: Sampler

"""
the form tile to provide learning curve plotting tile
"""
function tile_form_evaluate(inp::Signal, s::Sampler)
    return vbox(
                h2("Choose the segmentation file"),
                watch!(s, :fseg, textinput("/tmp/seg.h5", label="segmentation file")),
                vskip(1em),
                h2("Choose the label file"),
                watch!(s, :flbl, textinput("/tmp/lbl.h5", label="label file")),
                trigger!(s, :evaluate, button("Evaluate Segmenation"))
                ) |> maxwidth(400px)
end



"""
the tile of evaluate result
"""
function evaluate_result(fseg::AbstractString, flbl::AbstractString)
    if isfile(fseg) && isfile(flbl)
        return ""
    else
        return ""
    end
end

"""
the page of evaluate
"""
function evaluate()
    inp = Signal(Dict())
    s = Escher.sampler()

    form = tile_form_evaluate(inp, s)
    ret = consume(inp) do dict
        vbox(
             intent(s, form) >>> inp,
             vskip(2em),
             evaluate_result(get(dict, :fseg, ""), get(dict,:flbl, ""))
             ) |> Escher.pad(2em)
    end
end
