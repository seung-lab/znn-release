import Escher: @api, render

include("plotcurve.jl")

function wiretabs()
    tabbar = tabs([hbox(icon("history"), hskip(1em), "Training"),
                   hbox(icon("trending-down"), hskip(1em), "LearningCurve"),
                   hbox(icon("arrow-forward"), hskip(1em), "Inference"),
                   hbox(icon("assessment"), hskip(1em), "Evaluation")] )
    tabcontents = pages([ "training",
                        plotcurve(),
                        "forward pass",
                        "evaluation"])
    t, p = wire( tabbar, tabcontents, :tabschannel, :selected)

    return vbox(toolbar([iconbutton("cloud"), "ZNN", flex(), iconbutton("search")]),
                maxwidth(30em, t),
                Escher.pad(1em, p))
end

main(window) = begin
    push!(window.assets, "layout2")
    push!(window.assets, "icons")
    push!(window.assets, "widgets")

    wiretabs()
end
