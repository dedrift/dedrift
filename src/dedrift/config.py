"""Project configuration: .dedrift/config.toml loading with documented defaults."""

from __future__ import annotations

import sys

if sys.version_info >= (3, 11):  # noqa: UP036 - fallback kept for 3.10 dev sandboxes
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Materiality:
    """Effect-size (materiality) gates per signature channel (principle 2).

    An alert requires BOTH statistical significance after FDR AND an effect
    exceeding these thresholds. All overridable in ``.dedrift/config.toml``
    under ``[materiality]``.

    Attributes:
        refusal_rate_pp: Minimum refusal-rate shift, percentage points.
        format_validity_pp: Minimum format-validity shift, percentage points.
        rate_default_pp: Minimum shift for other rate signatures, pp.
        scalar_cohen_d: Minimum |Cohen's d| for location tests (Welch).
        ks_distance: Minimum KS statistic D (sup-norm CDF distance, in
            [0, 1]) for KS alerts. KS detects any distributional change, so
            gating it on Cohen's d would wrongly discard shape changes with
            equal means; D is the honest effect measure for this channel.
            Binding scale (stated, not hidden): the raw-alpha=0.05 critical
            value is D_crit ~ 1.36*sqrt((n+m)/(n*m)), so for equal arms the
            0.15 default lies below D_crit until n >~ 165 per arm — and BH
            adjustment only raises the bar. At typical per-family scale
            (tens of samples per window) significance is the stricter
            filter and this gate cannot be the one that fires; it exists to
            stop trivially significant D at large n from alerting.
        dispersion_ratio: Minimum robust dispersion ratio (or its inverse)
            for dispersion alerts; 1.5 means the mean absolute deviation
            from the median must grow or shrink by 50%. Gated on the same
            robust scale Brown--Forsythe tests on, not on the sample
            variance -- the variance ratio is unstable under heavy tails
            and gating on it would undo the reason for choosing a robust
            test. (Renamed from ``variance_ratio`` in v0.3.0, when the
            quantity it names actually changed; the old name would now be
            a lie about what is measured.)
        p95_relative: Minimum relative P95 shift.
        embedding_mmd2_floor: MMD^2 materiality floor. Negative (default)
            means auto-calibrate from reference-cycle pairs at check time
            (95th percentile of the known-same-distribution MMD^2 null);
            zero disables the floor; positive values are used as-is.
    """

    refusal_rate_pp: float = 2.0
    format_validity_pp: float = 1.0
    rate_default_pp: float = 2.0
    scalar_cohen_d: float = 0.5
    ks_distance: float = 0.15
    dispersion_ratio: float = 1.5
    p95_relative: float = 0.10
    embedding_mmd2_floor: float = -1.0

    def rate_threshold(self, signature: str) -> float:
        """Return the pp threshold for a rate signature (in [0,1] units)."""
        if signature == "refusal":
            return self.refusal_rate_pp / 100
        if signature == "format_valid":
            return self.format_validity_pp / 100
        return self.rate_default_pp / 100


@dataclass(frozen=True)
class AnytimeConfig:
    """Anytime-valid inference settings (``[anytime]`` in config.toml).

    The budget decomposes as ``alpha = alpha_prime + gamma_total``, where
    ``alpha_prime`` is the e-BH level and ``gamma_total`` pays for the
    nuisance-parameter coverage of *all* processes together — it is divided
    by the live process count at check time, because e-BH needs every input
    to be a valid e-value and coverage failures union-bound across the
    battery. Getting that split wrong inflates the real guarantee without
    changing anything visible in a report, so it is computed, never typed.

    Attributes:
        alpha: Lifetime, battery-wide false-alert budget. The claim is
            "P(ever alerting on a stable agent) <= alpha", per epoch.
        gamma_total: Portion of ``alpha`` spent on nuisance coverage;
            ``alpha_prime = alpha - gamma_total`` is the e-BH level. The
            default 0.02 was chosen from a sweep over allocations whose
            battery-wide claim is actually 0.05 (per-process split
            included). Within that valid region the trade is one-sided —
            detection at +10pp rises 24% -> 90% as gamma_total goes
            0.005 -> 0.03 with the measured null rate 0 throughout — so the
            choice is made on power alone, since validity does not
            discriminate. We stop at 0.02 rather than 0.03 because
            ``alpha_prime`` *is* the battery's FDR level, and spending most
            of the lifetime budget insuring against a coverage failure that
            has never been observed buys little.
        tilts: Base odds-ratio tilts; symmetrised to ``{psi, 1/psi}`` so
            drift in either direction is covered.
        epoch_allocation: ``"per_epoch"`` (default, honest: a reset means
            the hypothesis changed) or ``"geometric"`` (spends
            ``alpha * 2**-e`` on epoch ``e``, giving a genuine
            unbounded-epoch lifetime bound at a power cost).
    """

    alpha: float = 0.05
    gamma_total: float = 0.02
    tilts: tuple[float, ...] = (1.5, 2.0, 3.0)
    epoch_allocation: str = "per_epoch"

    @property
    def alpha_prime(self) -> float:
        """e-BH level after paying for nuisance coverage."""
        return self.alpha - self.gamma_total


@dataclass(frozen=True)
class ProjectConfig:
    """Parsed project configuration with defaults matching config.toml.

    Attributes:
        name: Project name.
        canary_repetitions: N repeated runs per canary per cycle.
        rolling_window_cycles: K cycles in the rolling reference.
        fdr_q: Benjamini-Hochberg FDR level.
        permutations: Number of label permutations for permutation tests.
        seed: Global seed recorded in every report.
        ph_lambda: Page-Hinkley alarm threshold (standardized units).
        ph_delta: Page-Hinkley dead-zone (standardized units).
        materiality: Effect-size gates.
        embedder: Pinned embedding model identifier ("" = embeddings unused).
        anytime: Anytime-valid inference settings.
        inference: Default inference mode, ``"fixed"`` or ``"anytime"``.
            Stays ``"fixed"`` until the anytime path has been reviewed on a
            project's own data; both paths run on identical logs so the
            comparison is reproducible.
    """

    name: str = "default"
    canary_repetitions: int = 7
    rolling_window_cycles: int = 5
    fdr_q: float = 0.05
    permutations: int = 500
    seed: int = 1729
    ph_lambda: float = 12.0
    ph_delta: float = 0.3
    materiality: Materiality = field(default_factory=Materiality)
    embedder: str = ""
    anytime: AnytimeConfig = field(default_factory=AnytimeConfig)
    inference: str = "fixed"

    @classmethod
    def load(cls, project_dir: Path) -> ProjectConfig:
        """Load configuration from ``<project_dir>/config.toml``.

        Missing keys fall back to the documented defaults; unknown keys are
        ignored.

        Args:
            project_dir: The ``.dedrift`` directory.

        Returns:
            The parsed configuration.
        """
        path = project_dir / "config.toml"
        if not path.exists():
            return cls()
        data: dict[str, Any] = tomllib.loads(path.read_text(encoding="utf-8"))
        project = data.get("project", {})
        detection = data.get("detection", {})
        materiality_raw = data.get("materiality", {})
        embeddings = data.get("embeddings", {})
        anytime_raw = data.get("anytime", {})
        anytime = AnytimeConfig(
            alpha=float(anytime_raw.get("alpha", 0.05)),
            gamma_total=float(anytime_raw.get("gamma_total", 0.02)),
            tilts=tuple(float(x) for x in anytime_raw.get("tilts", (1.5, 2.0, 3.0))),
            epoch_allocation=str(anytime_raw.get("epoch_allocation", "per_epoch")),
        )
        materiality = Materiality(
            refusal_rate_pp=float(materiality_raw.get("refusal_rate_pp", 2.0)),
            format_validity_pp=float(materiality_raw.get("format_validity_pp", 1.0)),
            rate_default_pp=float(materiality_raw.get("rate_default_pp", 2.0)),
            scalar_cohen_d=float(materiality_raw.get("scalar_cohen_d", 0.5)),
            ks_distance=float(materiality_raw.get("ks_distance", 0.15)),
            dispersion_ratio=float(
                materiality_raw.get("dispersion_ratio", materiality_raw.get("variance_ratio", 1.5))
            ),
            p95_relative=float(materiality_raw.get("p95_relative", 0.10)),
            embedding_mmd2_floor=float(materiality_raw.get("embedding_mmd2_floor", -1.0)),
        )
        return cls(
            name=str(project.get("name", "default")),
            canary_repetitions=int(project.get("canary_repetitions", 7)),
            rolling_window_cycles=int(project.get("rolling_window_cycles", 5)),
            fdr_q=float(detection.get("fdr_q", 0.05)),
            permutations=int(detection.get("permutations", 500)),
            seed=int(detection.get("seed", 1729)),
            ph_lambda=float(detection.get("ph_lambda", 12.0)),
            ph_delta=float(detection.get("ph_delta", 0.3)),
            materiality=materiality,
            embedder=str(embeddings.get("model", "")),
            anytime=anytime,
            inference=str(detection.get("inference", "fixed")),
        )
