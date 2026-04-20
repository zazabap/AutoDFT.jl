# scripts/plot_results.jl — EDITABLE
#
# Renders docs/progress.png: scatter of all trials in results.tsv in
# chronological order, with a running-minimum step line over "kept" + baseline
# trials. Mirrors the style of autoresearch-hubbard's progress.png.
#
# Regenerate after new trials with:
#   julia --project=. scripts/plot_results.jl

using CairoMakie
using Dates
using Printf

const REPO_ROOT   = normpath(joinpath(@__DIR__, ".."))
const RESULTS_TSV = joinpath(REPO_ROOT, "results.tsv")
const OUT_PATH    = joinpath(REPO_ROOT, "docs", "progress.png")

# Read results.tsv into a vector of NamedTuples sorted by timestamp.
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
            timestamp = DateTime(cols[1]),
            basis_name = cols[4],
            final_mse = parse(Float64, cols[7]),
            status = cols[13],
        ))
    end
    sort!(rows; by = r -> r.timestamp)
    return rows
end

# Visible floor on log scale: DFT hits ~3e-24. Floor display at 1e-12 so the
# point is still clearly the lowest but axis stays readable (~18 decades).
displayable(mse) = max(mse, 1e-12)

function main()
    rows = read_results(RESULTS_TSV)
    isempty(rows) && (@warn "results.tsv has no data rows — nothing to plot"; return)

    n = length(rows)
    xs = 1:n
    ys = [displayable(r.final_mse) for r in rows]

    # Split by status
    baseline_idx = findall(r -> r.status == "baseline", rows)
    kept_idx     = findall(r -> r.status == "kept",     rows)
    dropped_idx  = findall(r -> r.status == "dropped",  rows)

    # Running minimum over baseline + kept trials (dropped ones don't update the bar).
    rolling_min = Float64[]
    current = Inf
    for r in rows
        if r.status in ("baseline", "kept") && r.final_mse < current
            current = r.final_mse
        end
        push!(rolling_min, displayable(current))
    end

    fig = Figure(size = (1100, 520), fontsize = 14,
                 backgroundcolor = RGBf(1.0, 1.0, 1.0))

    kept_count = length(kept_idx) + length(baseline_idx)
    ax = Axis(fig[1, 1];
        title = @sprintf("Autoresearch progress: %d trials, %d kept", n, kept_count),
        titlealign = :left,
        xlabel = "Trial #",
        ylabel = "final_mse (lower is better)",
        yscale = log10,
        xticks = 1:n,
        xminorticksvisible = true,
        ygridstyle = :dash,
    )

    dropped_color = RGBf(0.70, 0.70, 0.70)
    kept_color    = RGBf(0.18, 0.65, 0.30)
    baseline_color = RGBf(0.20, 0.45, 0.80)

    # Running-best step line first so dots sit on top
    stairs!(ax, collect(xs), rolling_min;
        color = kept_color, linewidth = 2.5, step = :post,
        label = "Running best")

    # Dropped scatter (grey, small)
    isempty(dropped_idx) || scatter!(ax, collect(dropped_idx), ys[dropped_idx];
        color = dropped_color, markersize = 11, strokewidth = 0,
        label = "Dropped")

    # Baseline scatter (blue, medium)
    isempty(baseline_idx) || scatter!(ax, collect(baseline_idx), ys[baseline_idx];
        color = baseline_color, markersize = 14, strokewidth = 0,
        label = "Baseline")

    # Kept scatter (green, large, with black edge so it pops)
    isempty(kept_idx) || scatter!(ax, collect(kept_idx), ys[kept_idx];
        color = kept_color, markersize = 16,
        strokecolor = RGBf(0, 0, 0), strokewidth = 1,
        label = "Kept")

    # Annotate every baseline + kept point with basis name + final MSE.
    for i in vcat(baseline_idx, kept_idx)
        r = rows[i]
        mse_str = r.final_mse < 1e-6 ? @sprintf("%.2e", r.final_mse) :
                                        @sprintf("%.0f",  r.final_mse)
        text!(ax, "  $(r.basis_name)  ($(mse_str))";
            position = (i, displayable(r.final_mse)),
            align = (:left, :bottom),
            fontsize = 11,
            rotation = pi/8,
            color = RGBf(0.15, 0.15, 0.15),
        )
    end

    # Y-axis range: a bit below the running-min, a bit above the worst trial.
    min_y = minimum(ys)
    max_y = maximum(ys)
    ylims!(ax, min_y / 8, max_y * 8)
    xlims!(ax, 0.5, n + 0.5)

    axislegend(ax;
        position = :rc, framevisible = true,
        backgroundcolor = (:white, 0.9),
        labelsize = 11,
    )

    mkpath(dirname(OUT_PATH))
    save(OUT_PATH, fig)
    @info "Wrote $OUT_PATH" trials = n kept = kept_count min_mse = minimum(r -> r.final_mse, rows)
    return OUT_PATH
end

(abspath(PROGRAM_FILE) == @__FILE__) && main()
