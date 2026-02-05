from __future__ import annotations

import sys
from pathlib import Path

# Ensure `target/` is importable as a package root (so `import nanogptlite` works)
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
