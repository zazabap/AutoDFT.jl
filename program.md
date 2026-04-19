# Autoresearch Runbook — AutoDFT.jl

> **⚠ First-run note:** `results.tsv` and `best.tsv` are empty. Run `make baseline`
> before your first trial to seed the leaderboard (~10-60 min). If `make init`
> fails because the host Julia version differs from the committed `Manifest.toml`,
> run `make init-fresh` to re-resolve the lockfile first.

You are an autonomous Claude Code session running on a checkout of
`github.com/zazabap/AutoDFT.jl`. Your goal: find a new `AbstractSparseBasis`
implementation that achieves lower reconstruction MSE than the current
best (see `best.tsv`) on the frozen 512×512 fixture at `k = 26_214`.

## Rules

1. **Do not edit frozen files.** Files marked `[F]` in the design spec
   (`docs/superpowers/specs/2026-04-19-autoresearch-parametricdft-basis-design.md`)
   are SHA-hashed in `frozen-manifest.toml`. Any edit trips `make verify`
   and CI will reject the PR.
2. **Only edit `src/bases/*.jl`, `test/bases_tests.jl`, `results.tsv`, `best.tsv`.**
3. **Every trial must run through `make trial NAME=<Name>`.** Never
   hand-edit `results.tsv` or `best.tsv` — they are updated by the runner.
4. **If a trial is rejected (`make trial` exits 1):** immediately
   `git reset --hard HEAD~1` to undo the basis file. Then commit a
   dropped-trial note on a SEPARATE commit:
   `git commit --allow-empty -m "dropped: <Name>Basis — <why it failed>"`.
   This preserves the attempt log even though the code was reverted.
5. **The user is the kill switch.** Ctrl+C ends the session.

## Startup

```bash
make init            # Or: make init-fresh  if Project.toml has drifted.
make verify          # Prints "frozen surface OK" on success.
make test            # Full test suite — must pass.
```

If `results.tsv` is empty (no rows under current branch with status=baseline):

```bash
make baseline        # ~4 × 500-step trainings. Takes 10-60 min on GPU, much
                     # longer on CPU. Seeds results.tsv and best.tsv.
```

## Iteration loop

1. **Read** `best.tsv` — that's the bar to beat.
2. **Form a hypothesis.** Examples:
   - "MERA with 3 layers underperforms vs. QFT because its unitaries are
     over-constrained at 512×512. A 4-layer MERA with periodic boundary may help."
   - "Block-diagonal QFT⊕TEBD could capture both low- and high-frequency
     structure without increasing parameter count."
   Write the hypothesis as a commit message.
3. **Copy** `src/bases/_example_basis.jl` to `src/bases/<slug>.jl`.
   Rename `IdentityBasis` → `<Name>Basis`. Implement the contract described in
   the template. Add `register_basis!("<Name>", <Name>Basis)` at the bottom.
4. **Add a conformance test** to `test/bases_tests.jl`:
   ```julia
   @testset "<Name>Basis interface" begin
       b = <Name>Basis(9, 9)
       @test image_size(b) == (512, 512)
       @test num_parameters(b) > 0
       x = randn(ComplexF64, 512, 512)
       y = forward_transform(b, x)
       x̂ = inverse_transform(b, y)
       @test x̂ ≈ x atol=1e-8
       @test basis_hash(b) == basis_hash(b)
   end
   ```
5. **Commit:**
   ```bash
   git add src/bases/<slug>.jl test/bases_tests.jl
   git commit -m "trial: <Name>Basis — <one-line hypothesis>"
   ```
6. **Run the trial:**
   ```bash
   make trial NAME=<Name>
   ```
7. **On acceptance (exit 0):** `best.tsv` updated automatically. Commit it:
   ```bash
   git add best.tsv results.tsv
   git commit -m "accept: <Name>Basis lowers MSE to <x>"
   ```
   Then go back to step 1.

8. **On rejection (exit 1):**
   ```bash
   git reset --hard HEAD~1                  # remove the basis file
   git commit --allow-empty -m "dropped: <Name>Basis — <reason>"
   git add results.tsv
   git commit --amend --no-edit             # fold the results row into the note
   ```
   Then go back to step 1.

## Adding dependencies

If your new basis needs an additional Julia package, DO NOT edit `Project.toml`
or `Manifest.toml` — both are frozen. Instead, raise a human PR:
"harness update: add <Pkg> for <reason>." A harness-update PR bundles:
- edits to frozen files
- rerun of `scripts/compute_manifest.jl` (updates `frozen-manifest.toml`)
- a CI run with `BASIS_FREEZE_SALT` that updates `[secret].manifest_sha`

## Performance budget

`make trial` should complete in under ~10 minutes per trial on GPU, ~60 minutes
on CPU. If yours routinely exceeds this, log it in the commit note and skip to
the next idea — a slow trial starves exploration.
