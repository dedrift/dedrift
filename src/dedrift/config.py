"""Project configuration: .dedrift/config.toml loading with documented defaults."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):  # noqa: UP036 - fallback kept for 3.10 dev sandboxes
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib


_TOP_LEVEL_KEYS = frozenset({"project", "detection", "materiality", "embeddings", "anytime"})
_TABLE_KEYS = {
    "project": frozenset({"name", "canary_repetitions", "rolling_window_cycles"}),
    "detection": frozenset({"fdr_q", "permutations", "seed", "ph_lambda", "ph_delta", "inference"}),
    "materiality": frozenset(
        {
            "refusal_rate_pp",
            "format_validity_pp",
            "rate_default_pp",
            "scalar_cohen_d",
            "ks_distance",
            "dispersion_ratio",
            "variance_ratio",  # v0.2 compatibility alias
            "p95_relative",
            "embedding_mmd2_floor",
        }
    ),
    "embeddings": frozenset({"model"}),
    "anytime": frozenset({"alpha", "gamma_total", "tilts", "epoch_allocation"}),
}


def _require_finite(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")


def _require_range(
    name: str,
    value: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
    maximum_inclusive: bool = True,
) -> None:
    _require_finite(name, value)
    below = minimum is not None and (value < minimum if minimum_inclusive else value <= minimum)
    above = maximum is not None and (value > maximum if maximum_inclusive else value >= maximum)
    if below or above:
        left = "[" if minimum_inclusive else "("
        right = "]" if maximum_inclusive else ")"
        low = "-inf" if minimum is None else str(minimum)
        high = "inf" if maximum is None else str(maximum)
        raise ValueError(f"{name} must be in {left}{low}, {high}{right}, got {value!r}")


def _require_int(name: str, value: int, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")


def _table(data: dict[str, Any], name: str) -> dict[str, Any]:
    raw = data.get(name, {})
    if not isinstance(raw, dict):
        raise ValueError(f"[{name}] must be a TOML table")
    unknown = sorted(set(raw) - _TABLE_KEYS[name])
    if unknown:
        rendered = ", ".join(f"{name}.{key}" for key in unknown)
        raise ValueError(f"unknown config field(s): {rendered}")
    return raw


def _toml_string(table: dict[str, Any], key: str, default: str, *, section: str) -> str:
    value = table.get(key, default)
    if not isinstance(value, str):
        raise ValueError(f"{section}.{key} must be a string, got {value!r}")
    return value


def _toml_int(table: dict[str, Any], key: str, default: int, *, section: str) -> int:
    value = table.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{section}.{key} must be an integer, got {value!r}")
    return value


def _toml_float(table: dict[str, Any], key: str, default: float, *, section: str) -> float:
    value = table.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{section}.{key} must be a number, got {value!r}")
    return float(value)


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
        scalar_cohen_d: Reserved legacy setting. Welch is corroboration-only
            in v0.3.1, so this value does not gate alerts.
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

    def __post_init__(self) -> None:
        """Reject impossible or non-finite effect-size gates."""
        for name in ("refusal_rate_pp", "format_validity_pp", "rate_default_pp"):
            _require_range(f"materiality.{name}", getattr(self, name), minimum=0.0, maximum=100.0)
        _require_range("materiality.scalar_cohen_d", self.scalar_cohen_d, minimum=0.0)
        _require_range("materiality.ks_distance", self.ks_distance, minimum=0.0, maximum=1.0)
        _require_range("materiality.dispersion_ratio", self.dispersion_ratio, minimum=1.0)
        _require_range("materiality.p95_relative", self.p95_relative, minimum=0.0)
        _require_range("materiality.embedding_mmd2_floor", self.embedding_mmd2_floor, minimum=-1.0)

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

    def __post_init__(self) -> None:
        """Validate the lifetime error budget and betting-grid domain."""
        _require_range(
            "anytime.alpha",
            self.alpha,
            minimum=0.0,
            maximum=1.0,
            minimum_inclusive=False,
            maximum_inclusive=False,
        )
        _require_range(
            "anytime.gamma_total",
            self.gamma_total,
            minimum=0.0,
            maximum=self.alpha,
            minimum_inclusive=False,
            maximum_inclusive=False,
        )
        if not isinstance(self.tilts, tuple) or not self.tilts:
            raise ValueError("anytime.tilts must be a non-empty array")
        for index, tilt in enumerate(self.tilts):
            _require_range(f"anytime.tilts[{index}]", tilt, minimum=1.0, minimum_inclusive=False)
        if len(set(self.tilts)) != len(self.tilts):
            raise ValueError("anytime.tilts must not contain duplicates")
        if not isinstance(self.epoch_allocation, str) or self.epoch_allocation not in {
            "per_epoch",
            "geometric",
        }:
            raise ValueError(
                "anytime.epoch_allocation must be 'per_epoch' or 'geometric', "
                f"got {self.epoch_allocation!r}"
            )

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

    def __post_init__(self) -> None:
        """Validate project, detector, and inference settings."""
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("project.name must be a non-empty string")
        _require_int("project.canary_repetitions", self.canary_repetitions, minimum=2)
        _require_int("project.rolling_window_cycles", self.rolling_window_cycles, minimum=1)
        _require_range(
            "detection.fdr_q",
            self.fdr_q,
            minimum=0.0,
            maximum=1.0,
            minimum_inclusive=False,
            maximum_inclusive=False,
        )
        _require_int("detection.permutations", self.permutations, minimum=100)
        _require_int("detection.seed", self.seed, minimum=0)
        _require_range("detection.ph_lambda", self.ph_lambda, minimum=0.0, minimum_inclusive=False)
        _require_range("detection.ph_delta", self.ph_delta, minimum=0.0)
        if not isinstance(self.materiality, Materiality):
            raise ValueError("materiality must be a Materiality configuration")
        if not isinstance(self.embedder, str):
            raise ValueError(f"embeddings.model must be a string, got {self.embedder!r}")
        if not isinstance(self.anytime, AnytimeConfig):
            raise ValueError("anytime must be an AnytimeConfig configuration")
        if not isinstance(self.inference, str) or self.inference not in {"fixed", "anytime"}:
            raise ValueError(
                f"detection.inference must be 'fixed' or 'anytime', got {self.inference!r}"
            )

    @classmethod
    def load(cls, project_dir: Path) -> ProjectConfig:
        """Load configuration from ``<project_dir>/config.toml``.

        Missing keys fall back to the documented defaults. Unknown sections,
        unknown fields, and values outside their documented domains are
        rejected so a typo cannot silently weaken monitoring.

        Args:
            project_dir: The ``.dedrift`` directory.

        Returns:
            The parsed configuration.
        """
        path = project_dir / "config.toml"
        if not path.exists():
            return cls()
        data: dict[str, Any] = tomllib.loads(path.read_text(encoding="utf-8"))
        unknown_sections = sorted(set(data) - _TOP_LEVEL_KEYS)
        if unknown_sections:
            raise ValueError(f"unknown config section(s): {', '.join(unknown_sections)}")
        project = _table(data, "project")
        detection = _table(data, "detection")
        materiality_raw = _table(data, "materiality")
        embeddings = _table(data, "embeddings")
        anytime_raw = _table(data, "anytime")
        if "dispersion_ratio" in materiality_raw and "variance_ratio" in materiality_raw:
            raise ValueError(
                "materiality.dispersion_ratio and legacy materiality.variance_ratio "
                "cannot both be set"
            )
        tilts_raw = anytime_raw.get("tilts", [1.5, 2.0, 3.0])
        if not isinstance(tilts_raw, list) or not tilts_raw:
            raise ValueError("anytime.tilts must be a non-empty array")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float)) for value in tilts_raw
        ):
            raise ValueError("anytime.tilts must contain only numbers")
        anytime = AnytimeConfig(
            alpha=_toml_float(anytime_raw, "alpha", 0.05, section="anytime"),
            gamma_total=_toml_float(anytime_raw, "gamma_total", 0.02, section="anytime"),
            tilts=tuple(float(x) for x in tilts_raw),
            epoch_allocation=_toml_string(
                anytime_raw, "epoch_allocation", "per_epoch", section="anytime"
            ),
        )
        materiality = Materiality(
            refusal_rate_pp=_toml_float(
                materiality_raw, "refusal_rate_pp", 2.0, section="materiality"
            ),
            format_validity_pp=_toml_float(
                materiality_raw, "format_validity_pp", 1.0, section="materiality"
            ),
            rate_default_pp=_toml_float(
                materiality_raw, "rate_default_pp", 2.0, section="materiality"
            ),
            scalar_cohen_d=_toml_float(
                materiality_raw, "scalar_cohen_d", 0.5, section="materiality"
            ),
            ks_distance=_toml_float(materiality_raw, "ks_distance", 0.15, section="materiality"),
            dispersion_ratio=_toml_float(
                materiality_raw,
                ("dispersion_ratio" if "dispersion_ratio" in materiality_raw else "variance_ratio"),
                1.5,
                section="materiality",
            ),
            p95_relative=_toml_float(materiality_raw, "p95_relative", 0.10, section="materiality"),
            embedding_mmd2_floor=_toml_float(
                materiality_raw, "embedding_mmd2_floor", -1.0, section="materiality"
            ),
        )
        return cls(
            name=_toml_string(project, "name", "default", section="project"),
            canary_repetitions=_toml_int(project, "canary_repetitions", 7, section="project"),
            rolling_window_cycles=_toml_int(project, "rolling_window_cycles", 5, section="project"),
            fdr_q=_toml_float(detection, "fdr_q", 0.05, section="detection"),
            permutations=_toml_int(detection, "permutations", 500, section="detection"),
            seed=_toml_int(detection, "seed", 1729, section="detection"),
            ph_lambda=_toml_float(detection, "ph_lambda", 12.0, section="detection"),
            ph_delta=_toml_float(detection, "ph_delta", 0.3, section="detection"),
            materiality=materiality,
            embedder=_toml_string(embeddings, "model", "", section="embeddings"),
            anytime=anytime,
            inference=_toml_string(detection, "inference", "fixed", section="detection"),
        )
