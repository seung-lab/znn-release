import Escher: @api, render

include("plotcurve.jl")
include("evaluate.jl")

function wiretabs()
    tabbar = tabs([hbox(icon("autorenew"), hskip(1em), "Train"),
                   hbox(icon("trending-down"), hskip(1em), "LearningCurve"),
                   hbox(icon("forward"), hskip(1em), "Inference"),
                   hbox(icon("dashboard"), hskip(1em), "Watershed"),
                   hbox(icon("assessment"), hskip(1em), "Evaluate"),
                   hbox(icon("polymer"), hskip(1em), "Pipeline"),
                   hbox(icon("help"), hskip(1em), "Help")] )
    tabcontents = pages([ "training", plotcurve(), "forward pass", "watershed", evaluate(), "Pipeline", "help"])

    t, p = wire( tabbar, tabcontents, :tabschannel, :selected)

    return vbox(toolbar(["ZNN @", iconbutton("cloud"), flex()]),
                maxwidth(65em, t), Escher.pad(1em, p))
end

main(window) = begin
    push!(window.assets, "layout2")
    push!(window.assets, "icons")
    push!(window.assets, "widgets")

    wiretabs()
end
