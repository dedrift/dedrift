"""Null-calibration benchmark: measured false-alarm rates at canary scale.

A reproducible validity-scale study. Every method under test is run over the
SAME seeded stable-agent histories (no drift, no config change, ever), so
differences in measured false-alarm rates are attributable to the method,
not the data. See README.md for the protocol and METHODS_CONSIDERED.md for
tools that were attempted and excluded.
"""
