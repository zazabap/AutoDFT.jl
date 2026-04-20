# scripts/plot_results.jl — EDITABLE
#
# Renders docs/progress.png: horizontal bar chart of every trial in results.tsv,
# sorted by final_mse on a log scale. Baselines in blue, kept bases in green,
# dropped trials in red, with the acceptance threshold drawn as a vertical line.
#
# Regenerate after new trials with:
#   julia --project=. scripts/plot_results.jl

using CairoMakie
using Printf

const REPO_ROOT   = normpath(joinpath(@__DIR__, ".."))
const RESULTS_TSV = joinpath(REPO_ROOT, "results.tsv")
const BEST_TSV    = joinpath(REPO_ROOT, "best.tsv")
const OUT_PATH    = joinpath(REPO_ROOT, "docs", "progress.png")

# Read results.tsv into a vector of NamedTuples
function read_results(path::String)
    lines = readlines(path)
    isempty(lines) && return NamedTuple[]
    header = split(lines[1], '\t')
    rows = NamedTuple[]
    for line in lines[2:end]
        isempty(strip(line)) && continue
        cols = split(line, '\t')
        length(cols) < length(header) && continue
        push!(rows, (
            basis_name = cols[4],
            num_parameters = parse(Int, cols[6]),
            final_mse = parse(Float64, cols[7]),
            status = cols[13],
        ))
    end
    return rows
end

# Visible floor on log scale: DFTBasis hits ~3e-24 which would push the
# axis range to ~28 orders of magnitude. Floor display at 1e-12 so the
# bar is still clearly the shortest one but axis stays readable.
displayable(mse) = max(mse, 1e-12)

function main()
    rows = read_results(RESULTS_TSV)
    isempty(rows) && (@warn "results.tsv has no data rows — nothing to plot"; return)

    # Sort by final_mse ascending so the winner is at the top of the bar chart
    sort!(rows; by = r -> r.final_mse)

    # Cap axis at 2× max for headroom
    max_mse = maximum(r.final_mse for r in rows)

    colors = map(rows) do r
        r.status == "kept"     ? RGBf(0.20, 0.70, 0.30) :   # green — accepted
        r.status == "baseline" ? RGBf(0.20, 0.45, 0.80) :   # blue — baseline
                                 RGBf(0.80, 0.30, 0.25)     # red — dropped
    end

    fig = Figure(size = (1280, 480), fontsize = 14,
                 backgroundcolor = RGBf(0.98, 0.98, 0.98))

    ax = Axis(fig[1, 1];
        title = "AutoDFT.jl — compression MSE per basis (log scale)",
        titlealign = :left,
        xscale = log10,
        xlabel = "final_mse (lower is better)",
        ylabel = "",
        yticks = (1:length(rows), [r.basis_name for r in rows]),
        yreversed = true,                            # winner at top
        xgridstyle = :dash,
    )

    # Horizontal bars
    barplot!(ax,
        1:length(rows),
        [displayable(r.final_mse) for r in rows];
        direction = :x,
        color = colors,
        strokewidth = 0,
    )

    # Annotate each bar with the MSE value + param count. Use "%.2e" for
    # extreme values, "%.0f" for baseline-scale to keep labels compact.
    for (i, r) in enumerate(rows)
        label = if r.final_mse < 1e-6
            @sprintf("%.2e  (%d params)", r.final_mse, r.num_parameters)
        elseif r.final_mse >= 1e4
            @sprintf("%.2e  (%d params)", r.final_mse, r.num_parameters)
        else
            @sprintf("%.0f  (%d params)", r.final_mse, r.num_parameters)
        end
        text!(ax, label;
            position = (displayable(r.final_mse) * 1.3, i),
            align = (:left, :center),
            fontsize = 12,
            color = :black,
        )
    end

    # Vertical reference lines: baseline bar and acceptance threshold
    baseline_rows = filter(r -> r.status == "baseline", rows)
    if !isempty(baseline_rows)
        best_baseline = minimum(r -> r.final_mse, baseline_rows)
        accept_thresh = best_baseline * (1 - 0.01)

        vlines!(ax, [best_baseline];
                color = RGBf(0.20, 0.45, 0.80),
                linestyle = :dash, linewidth = 1.5)
        vlines!(ax, [accept_thresh];
                color = RGBf(0.80, 0.30, 0.25),
                linestyle = :dot, linewidth = 1.5)

        # Legend for the reference lines
        Legend(fig[1, 2],
            [LineElement(color = RGBf(0.20, 0.45, 0.80), linestyle = :dash),
             LineElement(color = RGBf(0.80, 0.30, 0.25), linestyle = :dot)],
            ["best baseline ($(round(best_baseline, digits=2)))",
             "acceptance bar (×0.99)"];
            framevisible = false,
            tellheight = false, tellwidth = false,
            halign = :right, valign = :top,
        )
    end

    # X-axis range: a bit below the smallest displayable value, enough above
    # the max to fit the text labels without truncation.
    min_disp = minimum(displayable(r.final_mse) for r in rows)
    xlims!(ax, min_disp / 3, max_mse * 300)

    mkpath(dirname(OUT_PATH))
    save(OUT_PATH, fig)
    @info "Wrote $OUT_PATH" n_rows = length(rows) min_mse = minimum(r -> r.final_mse, rows) max_mse
    return OUT_PATH
end

(abspath(PROGRAM_FILE) == @__FILE__) && main()
