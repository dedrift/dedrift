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
        variance_ratio: Minimum variance ratio (or its inverse) for
            dispersion alerts; 1.5 means var must grow or shrink by 50%.
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
    variance_ratio: float = 1.5
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
        materiality = Materiality(
            refusal_rate_pp=float(materiality_raw.get("refusal_rate_pp", 2.0)),
            format_validity_pp=float(materiality_raw.get("format_validity_pp", 1.0)),
            rate_default_pp=float(materiality_raw.get("rate_default_pp", 2.0)),
            scalar_cohen_d=float(materiality_raw.get("scalar_cohen_d", 0.5)),
            ks_distance=float(materiality_raw.get("ks_distance", 0.15)),
            variance_ratio=float(materiality_raw.get("variance_ratio", 1.5)),
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
        )
