from __future__ import annotations

from compilerlite.api import run_source


def test_nested_blocks():
    src = """
let x = 0;
while (x < 2) {
  if (x == 0) { print 10; }
  else { print 20; }
  let x = x + 1;
}
""".strip()
    assert run_source(src=src) == ["10", "20"]
