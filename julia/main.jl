import Escher: @api, render

include("plotcurve.jl")

function wiretabs(inp::Signal, s::Sampler)
    t, p = wire(tabs([hbox(icon("home"), hskip(1em), "LearningCurve"),
                      hbox(icon("info-outline"), hskip(1em), "Evaluation"),
                      hbox(icon("settings"), hskip(1em), "Training")]),
                pages([ plotcurve(inp, s), #map(inp) do dict plotcurve(inp, s, dict) end,
                       "Notification tab content",
                       "Settings tab content"]),
                :tabschannel, :selected)
    return t, p
end

main(window) = begin
    push!(window.assets, "layout2")
    push!(window.assets, "icons")
    push!(window.assets, "widgets")

    inp = Signal(Dict())
    s = Escher.sampler()

    t,p = wiretabs(inp, s)

    vbox(toolbar([iconbutton("face"), "Web ZNN", flex(), iconbutton("search")]),
         maxwidth(30em, t),
         Escher.pad(1em, p))
end
