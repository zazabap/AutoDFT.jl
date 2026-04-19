# Makefile — FROZEN
# Shell targets wrapping the AutoDFT runners and the freeze-verification flow.

JULIA   ?= julia
PROJECT := --project=.

.PHONY: help init init-fresh test verify baseline trial rehash fixture

help:
	@echo "Targets:"
	@echo "  init         Pkg.instantiate() from committed Manifest.toml"
	@echo "  init-fresh   Pkg.resolve() + instantiate (after Project.toml change)"
	@echo "  fixture      Regenerate data/fixture_512.bin"
	@echo "  verify       Run prepare.jl (frozen-surface check)"
	@echo "  test         verify + Pkg.test()"
	@echo "  baseline     Seed results.tsv with 4 ParametricDFT baselines"
	@echo "  trial NAME=X Train+eval basis X; exit 0 if accepted, 1 if dropped"
	@echo "  rehash       Regenerate frozen-manifest.toml (harness-edit flow only)"

init:
	$(JULIA) $(PROJECT) -e 'using Pkg; Pkg.instantiate()'

init-fresh:
	$(JULIA) $(PROJECT) -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'

fixture:
	$(JULIA) $(PROJECT) scripts/generate_fixture.jl

verify:
	$(JULIA) $(PROJECT) prepare.jl

test: verify
	$(JULIA) $(PROJECT) -e 'using Pkg; Pkg.test()'

baseline:
	$(JULIA) $(PROJECT) -e 'using AutoDFT; AutoDFT.run_baseline()'

trial:
	@test -n "$(NAME)" || { echo "Usage: make trial NAME=<BasisName>"; exit 2; }
	$(JULIA) $(PROJECT) -e 'using AutoDFT; exit(AutoDFT.run_trial("$(NAME)") ? 0 : 1)'

rehash:
	$(JULIA) $(PROJECT) scripts/compute_manifest.jl
