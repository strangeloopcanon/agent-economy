from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True, order=False)
class Version:
    major: int
    minor: int
    patch: int
    prerelease: tuple[str, ...] = ()
    build: tuple[str, ...] = ()

    @classmethod
    def parse(cls, raw: str) -> "Version":
        return parse_version(raw)

    def __str__(self) -> str:
        core = f"{self.major}.{self.minor}.{self.patch}"
        pre = "" if not self.prerelease else "-" + ".".join(self.prerelease)
        build = "" if not self.build else "+" + ".".join(self.build)
        return core + pre + build

    def _cmp_key(self) -> tuple[int, int, int, tuple[str, ...], bool]:
        # NOTE: build must not affect precedence.
        # BUG (intentional for arena): prerelease identifiers are compared lexicographically only,
        # so numeric identifiers like "10" sort before "2".
        return (self.major, self.minor, self.patch, self.prerelease, bool(self.prerelease))

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Version):
            return NotImplemented
        a = (self.major, self.minor, self.patch)
        b = (other.major, other.minor, other.patch)
        if a != b:
            return a < b

        # Final releases have higher precedence than pre-releases.
        if not self.prerelease and other.prerelease:
            return False
        if self.prerelease and not other.prerelease:
            return True

        # Pre-release compare (BUG: numeric identifiers should compare numerically).
        return self.prerelease < other.prerelease

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Version):
            return NotImplemented
        return (
            self.major,
            self.minor,
            self.patch,
            self.prerelease,
        ) == (
            other.major,
            other.minor,
            other.patch,
            other.prerelease,
        )


def parse_version(raw: str) -> Version:
    raw = str(raw)
    if not raw.strip():
        raise ValueError("empty version")

    core_and_more = raw.strip().split("+", 1)
    core_pre = core_and_more[0]
    build: tuple[str, ...] = ()
    if len(core_and_more) == 2:
        build_s = core_and_more[1]
        if not build_s:
            raise ValueError("empty build metadata")
        build = tuple(build_s.split("."))

    core_pre_split = core_pre.split("-", 1)
    core = core_pre_split[0]
    prerelease: tuple[str, ...] = ()
    if len(core_pre_split) == 2:
        pre_s = core_pre_split[1]
        if not pre_s:
            raise ValueError("empty prerelease")
        prerelease = tuple(pre_s.split("."))

    parts = core.split(".")
    if len(parts) != 3:
        raise ValueError("expected MAJOR.MINOR.PATCH")
    major_s, minor_s, patch_s = parts
    for s in (major_s, minor_s, patch_s):
        if not s.isdigit():
            raise ValueError("core version must be numeric")
        if len(s) > 1 and s.startswith("0"):
            raise ValueError("core version must not contain leading zeros")

    major = int(major_s)
    minor = int(minor_s)
    patch = int(patch_s)

    # Lightweight validation for identifiers.
    for ident in prerelease:
        if not ident:
            raise ValueError("empty prerelease identifier")
        if ident.isdigit() and len(ident) > 1 and ident.startswith("0"):
            raise ValueError("numeric prerelease identifiers must not have leading zeros")
        if not all(ch.isalnum() or ch == "-" for ch in ident):
            raise ValueError("invalid prerelease identifier")

    for ident in build:
        if not ident:
            raise ValueError("empty build identifier")
        if not all(ch.isalnum() or ch == "-" for ch in ident):
            raise ValueError("invalid build identifier")

    return Version(major=major, minor=minor, patch=patch, prerelease=prerelease, build=build)
