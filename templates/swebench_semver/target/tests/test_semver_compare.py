from __future__ import annotations

import pytest

from semverlite import Version


def test_basic_core_ordering() -> None:
    assert Version.parse("1.2.3") < Version.parse("2.0.0")
    assert Version.parse("1.2.3") < Version.parse("1.3.0")
    assert Version.parse("1.2.3") < Version.parse("1.2.4")


def test_build_is_ignored_for_precedence() -> None:
    assert Version.parse("1.0.0+abc") == Version.parse("1.0.0+def")
    assert not (Version.parse("1.0.0+abc") < Version.parse("1.0.0+def"))
    assert not (Version.parse("1.0.0+abc") > Version.parse("1.0.0+def"))


def test_prerelease_lower_than_release() -> None:
    assert Version.parse("1.0.0-alpha") < Version.parse("1.0.0")


def test_prerelease_numeric_identifiers_are_numeric() -> None:
    # This is currently broken in the template implementation.
    assert Version.parse("1.0.0-alpha.10") > Version.parse("1.0.0-alpha.2")


def test_invalid_core_versions_rejected() -> None:
    with pytest.raises(ValueError):
        Version.parse("")
    with pytest.raises(ValueError):
        Version.parse("1")
    with pytest.raises(ValueError):
        Version.parse("1.2")
    with pytest.raises(ValueError):
        Version.parse("1.2.3.4")
    with pytest.raises(ValueError):
        Version.parse("01.2.3")
