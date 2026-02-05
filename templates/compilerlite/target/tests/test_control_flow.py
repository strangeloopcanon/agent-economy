from __future__ import annotations

from compilerlite.api import run_source


def test_while_and_if_else():
    src = """
let x = 0;
while (x < 3) {
  print x;
  let x = x + 1;
}
if (x == 3) { print 99; } else { print 0; }
""".strip()
    out = run_source(src=src)
    assert out == ["0", "1", "2", "99"]
