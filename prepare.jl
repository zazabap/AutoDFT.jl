# prepare.jl — FROZEN (itself hashed)
#
# Runs at the start of every `make trial` and `make test`. Verifies:
#   1. Every file in [files] has matching SHA256.
#   2. Probe value matches [probe].qft_identity_mse (atol 1e-10).
#   3. If BASIS_FREEZE_SALT is in env, recompute [secret].manifest_sha and compare.
#
# Exits with status 0 ("frozen surface OK") or 1 (prints the failing check).

using SHA
using TOML

const REPO_ROOT = @__DIR__

function die(msg)
    println(stderr, "FROZEN SURFACE VIOLATION: $msg")
    exit(1)
end

function verify_files(manifest)
    files = get(manifest, "files", Dict{String,String}())
    for (rel, expected_sha) in files
        abs = joinpath(REPO_ROOT, rel)
        isfile(abs) || die("missing frozen file: $rel")
        actual = bytes2hex(open(sha256, abs))
        actual == expected_sha || die("SHA mismatch for $rel\n  expected: $expected_sha\n  actual:   $actual")
    end
end

function verify_probe(manifest)
    expected = get(get(manifest, "probe", Dict()), "qft_identity_mse", nothing)
    expected === nothing && die("[probe].qft_identity_mse missing from frozen-manifest.toml")
    if expected == -1.0
        @warn "Probe not yet pinned (sentinel -1.0). Run scripts/compute_manifest.jl on a host with ParametricDFT installed."
        return
    end
    push!(LOAD_PATH, REPO_ROOT)
    actual = Core.eval(Main, :(using AutoDFT; AutoDFT.run_probe()))
    isapprox(actual, expected; atol=1e-10) ||
        die("probe mismatch\n  expected: $expected\n  actual:   $actual\n  diff:     $(actual - expected)")
end

function verify_secret(manifest)
    salt = get(ENV, "BASIS_FREEZE_SALT", "")
    isempty(salt) && return
    expected = get(get(manifest, "secret", Dict()), "manifest_sha", nothing)
    expected === nothing && die("[secret].manifest_sha missing but BASIS_FREEZE_SALT is set")
    files = get(manifest, "files", Dict())
    sorted_shas = [files[k] for k in sort(collect(keys(files)))]
    actual = bytes2hex(sha256(salt * join(sorted_shas, "")))
    actual == expected || die("manifest secret SHA mismatch (file + manifest edits don't align)")
end

function main()
    manifest_path = joinpath(REPO_ROOT, "frozen-manifest.toml")
    isfile(manifest_path) || die("frozen-manifest.toml not found")
    manifest = TOML.parsefile(manifest_path)
    verify_files(manifest)
    verify_secret(manifest)
    verify_probe(manifest)
    println("frozen surface OK")
end

main()
