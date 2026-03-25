"""
Compatibility wrapper for the integrated hard-positive pipeline.

This script delegates to Select_Hard_Positive.py in revision-only mode so we
keep a single source of truth for the actual selection logic.

읽는 순서:
- 이 파일 자체에는 로직이 거의 없습니다.
- "revise only" 작업을 짧은 진입점으로 실행하기 위한 래퍼입니다.
"""

from __future__ import annotations

import sys

from Select_Hard_Positive import main


if __name__ == "__main__":
    if "--revise-existing-only" not in sys.argv:
        sys.argv.append("--revise-existing-only")
    main()
