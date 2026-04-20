# scripts/compute_manifest.jl — FROZEN (itself hashed in the manifest it creates)
#
# Regenerates frozen-manifest.toml:
#   [files]   — SHA256 of every frozen file
#   [probe]   — qft_identity_mse = run_probe()
#   [secret]  — sha256(BASIS_FREEZE_SALT + concat(sorted_file_shas)), if env var set
#
# Run after any legitimate harness edit. Developers without the salt simply
# regenerate [files] + [probe]; [secret] is injected in CI by the rehash workflow.

using SHA
using TOML

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))

"List of FROZEN file paths (relative to repo root). Keep in sync with the spec."
const FROZEN_FILES = [
    ".claude/CLAUDE.md",
    ".claude/rules/julia-conventions.md",
    ".github/workflows/CI.yml",
    ".github/workflows/basis-freeze.yml",
    "Project.toml",
    "Manifest.toml",
    "Makefile",
    "prepare.jl",
    "program.md",
    "README.md",
    "LICENSE",
    "src/AutoDFT.jl",
    "src/runners.jl",
    "src/harness/fixture.jl",
    "src/harness/evaluate.jl",
    "src/harness/probe.jl",
    "src/harness/train.jl",
    "scripts/generate_fixture.jl",
    "scripts/generate_fixture2.jl",
    "scripts/compute_manifest.jl",
    "data/fixture_512.bin",
    "data/fixture2_512.bin",
    "test/runtests.jl",
    "test/harness_tests.jl",
]

function file_sha(path)
    bytes2hex(open(sha256, path))
end

function main()
    file_hashes = Dict{String, String}()
    for rel in FROZEN_FILES
        abs = joinpath(REPO_ROOT, rel)
        if !isfile(abs)
            @warn "Frozen file missing — skipping" rel
            continue
        end
        file_hashes[rel] = file_sha(abs)
    end

    # Compute probe value (requires ParametricDFT + fixture + src/ all present)
    probe_mse = try
        @info "Running probe to pin qft_identity_mse"
        push!(LOAD_PATH, REPO_ROOT)
        Core.eval(Main, :(using AutoDFT; AutoDFT.run_probe()))
    catch e
        @warn "Could not compute probe — leaving sentinel -1.0. Run prepare.jl manually when environment is ready." exception=e
        -1.0
    end

    # Optional CI secret: salt from env
    secret_entry = Dict{String, Any}()
    salt = get(ENV, "BASIS_FREEZE_SALT", "")
    if !isempty(salt)
        sorted_shas = [file_hashes[k] for k in sort(collect(keys(file_hashes)))]
        combined = salt * join(sorted_shas, "")
        secret_entry["manifest_sha"] = bytes2hex(sha256(combined))
    end

    manifest = Dict{String, Any}(
        "files" => file_hashes,
        "probe" => Dict("qft_identity_mse" => probe_mse),
    )
    !isempty(secret_entry) && (manifest["secret"] = secret_entry)

    out_path = joinpath(REPO_ROOT, "frozen-manifest.toml")
    open(out_path, "w") do io
        TOML.print(io, manifest; sorted=true)
    end
    @info "Wrote manifest" out_path files=length(file_hashes) probe_mse
    return nothing
end

(abspath(PROGRAM_FILE) == @__FILE__) && main()
