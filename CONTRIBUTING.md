# Contributing to dedrift

Thanks for considering it. dedrift's whole premise is that statistical
claims should be measured rather than asserted, so contributions are
held to that standard — but the bar is about evidence, not ceremony.

## Licensing: read this first

Two separate things, often confused:

- **Outbound (what users get):** dedrift is distributed under
  **AGPL-3.0-only**. That does not change.
- **Inbound (what you grant when contributing):** contributions are
  accepted under the [Contributor License Agreement](CLA.md), which
  grants the project owner the right to relicense contributed code —
  including under commercial terms.

Why: dedrift is open-core. The open project stays open, and a
commercially licensed edition funds the work. That model only functions
if one party can license the whole codebase, which means every
contribution needs an explicit relicensing grant. You keep the copyright
in your work.

**Accepting is one line** — tick the box in the pull-request template.
You only do it once, and it covers your future contributions too.

Each commit must also carry a `Signed-off-by` line certifying the
[Developer Certificate of Origin](https://developercertificate.org/):

```bash
git commit -s -m "fix: ..."
```

## Before you write code

Open an issue first for anything beyond a small fix. Two reasons: the
scope of this repository is deliberately narrow, and some advanced
statistical methods are out of scope here regardless of quality — a
maintainer will tell you upfront rather than after you've done the work.

## The statistical bar

If a change touches a detector, a threshold, a p-value, or anything that
affects when an alert fires, it needs evidence in the same form the
project already uses:

- **New or changed detector:** a null-calibration test measuring its
  false-alarm rate against a documented acceptance band, and a power
  test against an injected shift of stated size. See
  `tests/test_calibration.py` for the pattern.
- **Changed default:** the reasoning, and a test that fails if the new
  value stops meaning what the docs say it means (see the
  `ks_distance` binding test for an example of a config-aware test).
- **New claim in the docs:** the measurement that backs it, and the
  limitation it implies. Publishing what the tool *can't* do is a
  feature here, not an embarrassment.

Deviating from `SPEC.md` may well be right, but say so explicitly in the
pull request and explain why.

## Practical checks

```bash
pip install -e ".[dev]"
ruff format . && ruff check src tests
mypy src
pytest                       # fast suite
pytest -m "calibration or power"   # the slow simulation suites
```

Conventional commits (`feat:`, `fix:`, `docs:`, `test:`, `chore:`).
Keep pull requests small and in a working state.

## Reporting a false alarm or a missed detection

These are the most valuable reports the project can receive, and they
need enough to reproduce: the `dedrift` version, your config, the number
of canaries and repetitions, and — ideally — the seed and a minimal log
extract. A report that lets a maintainer reproduce a bad alert is worth
more than a patch.
