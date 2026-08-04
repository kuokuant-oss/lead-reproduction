"""Import support for testable research runners.

Historical runners import sibling modules by filename because they are normally
executed as scripts.  Add this directory to the import path when a runner is
loaded as ``scripts.<name>`` by unit tests so both invocation styles resolve the
same modules.
"""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
