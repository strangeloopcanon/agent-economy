from __future__ import annotations

import pytest

from semverlite import Version


def test_semver_precedence_examples() -> None:
    # SemVer 2.0.0 example ordering:
    # alpha < alpha.1 < alpha.beta < beta < beta.2 < beta.11 < rc.1 < (no prerelease)
    ordered = [
        "1.0.0-alpha",
        "1.0.0-alpha.1",
        "1.0.0-alpha.beta",
        "1.0.0-beta",
        "1.0.0-beta.2",
        "1.0.0-beta.11",
        "1.0.0-rc.1",
        "1.0.0",
    ]
    parsed = [Version.parse(v) for v in ordered]
    assert parsed == sorted(parsed)


def test_numeric_vs_non_numeric_prerelease() -> None:
    # Numeric identifiers have lower precedence than non-numeric.
    assert Version.parse("1.0.0-1") < Version.parse("1.0.0-alpha")


def test_invalid_prerelease_identifiers_rejected() -> None:
    with pytest.raises(ValueError):
        Version.parse("1.0.0-01")
    with pytest.raises(ValueError):
        Version.parse("1.0.0-")
    with pytest.raises(ValueError):
        Version.parse("1.0.0-alpha..1")
