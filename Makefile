# dedrift — repository Makefile.
#
# The benchmark target regenerates the null-calibration study end to end:
# every measured number on https://dedrift.ai/benchmark/ and in the paper's
# validity-scale section comes from these commands, from the committed seed
# list, at the pinned versions in benchmark/requirements-benchmark.txt.

PY ?= python

.PHONY: benchmark benchmark-quick benchmark-tables

# Full study: 500 seeded runs per method per scale. Several hours on a
# 14-core machine; the two dedrift legs dominate.
benchmark:
	$(PY) -m benchmark.run --all
	$(PY) -m benchmark.tables

# Smoke run for development: 20 seeds, same code path.
benchmark-quick:
	$(PY) -m benchmark.run --all --quick
	$(PY) -m benchmark.tables

# Re-render the table fragments (web page + paper) from existing results
# without re-running the measurements.
benchmark-tables:
	$(PY) -m benchmark.tables
