"""Strict project-configuration parsing and domain validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from dedrift.config import AnytimeConfig, Materiality, ProjectConfig


def write_config(tmp_path: Path, content: str) -> Path:
    project_dir = tmp_path / ".dedrift"
    project_dir.mkdir()
    (project_dir / "config.toml").write_text(content, encoding="utf-8")
    return project_dir


def test_loads_complete_valid_config(tmp_path: Path) -> None:
    project_dir = write_config(
        tmp_path,
        """
[project]
name = "support-agent"
canary_repetitions = 9
rolling_window_cycles = 4

[detection]
fdr_q = 0.1
permutations = 750
seed = 4
ph_lambda = 8.0
ph_delta = 0.2
inference = "anytime"

[materiality]
refusal_rate_pp = 3.0
format_validity_pp = 2.0
rate_default_pp = 4.0
scalar_cohen_d = 0.6
ks_distance = 0.2
dispersion_ratio = 1.8
p95_relative = 0.15
embedding_mmd2_floor = 0.01

[embeddings]
model = "hash"

[anytime]
alpha = 0.08
gamma_total = 0.03
tilts = [1.25, 2.5]
epoch_allocation = "geometric"
""",
    )
    config = ProjectConfig.load(project_dir)
    assert config.name == "support-agent"
    assert config.canary_repetitions == 9
    assert config.permutations == 750
    assert config.materiality.dispersion_ratio == 1.8
    assert config.anytime.alpha_prime == pytest.approx(0.05)
    assert config.inference == "anytime"


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("[unknown]\nvalue = 1\n", "unknown config section"),
        ("[project]\ncanary_repetitons = 7\n", "project.canary_repetitons"),
        ('[detection]\nfdr_q = "0.05"\n', "detection.fdr_q must be a number"),
        ("[project]\ncanary_repetitions = 7.5\n", "must be an integer"),
        ('[anytime]\ntilts = [1.5, "2"]\n', "must contain only numbers"),
    ],
)
def test_rejects_unknown_or_mistyped_toml(tmp_path: Path, content: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ProjectConfig.load(write_config(tmp_path, content))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"canary_repetitions": 1},
        {"rolling_window_cycles": 0},
        {"fdr_q": 0.0},
        {"fdr_q": 1.0},
        {"fdr_q": float("nan")},
        {"permutations": 99},
        {"seed": -1},
        {"ph_lambda": 0.0},
        {"ph_delta": -0.1},
        {"inference": "sometimes"},
    ],
)
def test_rejects_invalid_project_domains(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ProjectConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"refusal_rate_pp": -0.1},
        {"format_validity_pp": 100.1},
        {"scalar_cohen_d": -0.1},
        {"ks_distance": 1.01},
        {"dispersion_ratio": 0.99},
        {"p95_relative": float("inf")},
        {"embedding_mmd2_floor": -1.01},
    ],
)
def test_rejects_invalid_materiality_domains(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        Materiality(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha": 0.0},
        {"alpha": 1.0},
        {"gamma_total": 0.0},
        {"gamma_total": 0.05},
        {"tilts": ()},
        {"tilts": (1.0,)},
        {"tilts": (1.5, 1.5)},
        {"epoch_allocation": "reset_whenever"},
    ],
)
def test_rejects_invalid_anytime_domains(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        AnytimeConfig(**kwargs)  # type: ignore[arg-type]


def test_loads_legacy_variance_ratio_alias(tmp_path: Path) -> None:
    config = ProjectConfig.load(write_config(tmp_path, "[materiality]\nvariance_ratio = 2.25\n"))
    assert config.materiality.dispersion_ratio == 2.25


def test_rejects_new_and_legacy_dispersion_keys_together(tmp_path: Path) -> None:
    project_dir = write_config(
        tmp_path,
        "[materiality]\ndispersion_ratio = 1.5\nvariance_ratio = 2.0\n",
    )
    with pytest.raises(ValueError, match="cannot both be set"):
        ProjectConfig.load(project_dir)
