# src/runners.jl — FROZEN
#
# User-facing entry points for `make baseline` and `make trial NAME=...`.
# Both produce rows in results.tsv; `run_trial` also compares against best.tsv
# and writes the acceptance decision.

using Dates
using Printf

const RESULTS_HEADER = "timestamp\tbranch\tcommit_sha\tbasis_name\tbasis_hash\tnum_parameters\tfinal_mse\tprobe_mse\ttrain_steps\ttrain_wallclock_ms\ttransform_time_ms\tdevice\tstatus\tnotes"

function _git_branch()
    try
        strip(read(`git rev-parse --abbrev-ref HEAD`, String))
    catch
        "unknown"
    end
end

function _git_sha()
    try
        strip(read(`git rev-parse --short HEAD`, String))
    catch
        "uncommitted"
    end
end

function _append_results_row(; timestamp, branch, commit_sha, basis_name, basis_hash_,
                              num_parameters, final_mse, probe_mse, train_steps,
                              train_wallclock_ms, transform_time_ms, device, status, notes)
    new_file = !isfile(RESULTS_PATH)
    open(RESULTS_PATH, "a") do io
        new_file && println(io, RESULTS_HEADER)
        @printf(io, "%s\t%s\t%s\t%s\t%s\t%d\t%.10e\t%.10e\t%d\t%.3f\t%.3f\t%s\t%s\t%s\n",
                timestamp, branch, commit_sha, basis_name, basis_hash_, num_parameters,
                final_mse, probe_mse, train_steps, train_wallclock_ms, transform_time_ms,
                device, status, notes)
    end
end

function _read_best_mse()
    isfile(BEST_PATH) || return Inf
    lines = readlines(BEST_PATH)
    length(lines) < 2 && return Inf
    cols = split(lines[2], '\t')
    # final_mse is column 7 (1-indexed)
    return parse(Float64, cols[7])
end

function _write_best_row(row_str::AbstractString)
    open(BEST_PATH, "w") do io
        println(io, RESULTS_HEADER)
        println(io, rstrip(row_str))
    end
end

"""
    run_baseline()

Evaluate each of the four ParametricDFT baselines with the frozen training config,
append their rows to results.tsv, and set best.tsv to the leader. Idempotent per-branch:
will NOT re-run if rows with `status=baseline` already exist on the current branch.
"""
function run_baseline()
    probe_mse = run_probe()
    branch = _git_branch()
    if isfile(RESULTS_PATH)
        for line in readlines(RESULTS_PATH)[2:end]
            cols = split(line, '\t')
            length(cols) >= 13 && cols[2] == branch && cols[13] == "baseline" &&
                (@info "Baselines already seeded on branch $branch — skipping"; return nothing)
        end
    end

    device = string(_default_device())
    best_mse = Inf
    best_row = ""
    for (name, T) in BASELINES
        @info "Evaluating baseline $name"
        basis, final_mse, wallclock = train_trial(T)
        ts = string(now())
        row = (
            timestamp = ts, branch = branch, commit_sha = _git_sha(),
            basis_name = name, basis_hash_ = basis_hash(basis),
            num_parameters = num_parameters(basis),
            final_mse = final_mse, probe_mse = probe_mse,
            train_steps = TRAIN_STEPS, train_wallclock_ms = wallclock,
            transform_time_ms = 0.0, device = device,
            status = "baseline", notes = "",
        )
        _append_results_row(; row...)
        if final_mse < best_mse
            best_mse = final_mse
            best_row = join((ts, branch, _git_sha(), name, basis_hash(basis),
                             string(num_parameters(basis)),
                             @sprintf("%.10e", final_mse), @sprintf("%.10e", probe_mse),
                             string(TRAIN_STEPS), @sprintf("%.3f", wallclock), "0.000",
                             device, "baseline", ""), '\t')
        end
    end
    _write_best_row(best_row)
    @info "Baseline seeding complete" best_mse
    return nothing
end

"""
    run_trial(name::String) -> Bool

Train+evaluate the basis registered under `name`. Append a row to results.tsv;
return true if accepted (final_mse < best * (1 - ACCEPTANCE_REL)), false otherwise.
The Makefile target translates the Bool into exit 0/1.
"""
function run_trial(name::String)
    T = get_basis_type(name)
    probe_mse = run_probe()
    best_mse = _read_best_mse()
    device = string(_default_device())
    @info "Running trial" name best_mse device
    basis, final_mse, wallclock = train_trial(T)
    accepted = final_mse < best_mse * (1 - ACCEPTANCE_REL)
    ts = string(now())
    branch = _git_branch()
    sha = _git_sha()
    bh = basis_hash(basis)
    row_fields = (ts, branch, sha, name, bh,
                  string(num_parameters(basis)),
                  @sprintf("%.10e", final_mse), @sprintf("%.10e", probe_mse),
                  string(TRAIN_STEPS), @sprintf("%.3f", wallclock), "0.000",
                  device, accepted ? "kept" : "dropped", "")
    _append_results_row(; timestamp=ts, branch=branch, commit_sha=sha, basis_name=name,
                        basis_hash_=bh, num_parameters=num_parameters(basis),
                        final_mse=final_mse, probe_mse=probe_mse,
                        train_steps=TRAIN_STEPS, train_wallclock_ms=wallclock,
                        transform_time_ms=0.0, device=device,
                        status=(accepted ? "kept" : "dropped"), notes="")
    if accepted
        _write_best_row(join(row_fields, '\t'))
        @info "Accepted — new best" final_mse baseline=best_mse
    else
        @info "Rejected" final_mse baseline=best_mse
    end
    return accepted
end
