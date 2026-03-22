"""
Compatibility wrapper for the integrated hard-positive pipeline.

This script delegates to Select_Hard_Positive.py in revision-only mode so we
keep a single source of truth for the actual selection logic.
"""

from __future__ import annotations

import sys

from Select_Hard_Positive import main


if __name__ == "__main__":
    if "--revise-existing-only" not in sys.argv:
        sys.argv.append("--revise-existing-only")
    main()
