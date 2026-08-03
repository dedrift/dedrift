## What this changes

<!-- One or two sentences. Link the issue if there is one. -->

## Evidence

<!--
If this touches a detector, threshold, p-value, or anything affecting
when an alert fires: what measurement backs it? Calibration test, power
test, or the reasoning if neither applies. If it deviates from SPEC.md,
say so and why.
-->

## Checklist

- [ ] I have read the [CLA](../CLA.md) and I agree to it for this and all my future contributions to this project
- [ ] Commits are signed off (`git commit -s`, per the [DCO](https://developercertificate.org/))
- [ ] `ruff format . && ruff check src tests` and `mypy src` are clean
- [ ] `pytest` passes; calibration/power suites run if statistics changed
- [ ] Docs updated if behaviour or a documented claim changed
