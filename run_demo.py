"""
项目根目录一键体验入口。
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tests.demo import main as demo_main


if __name__ == "__main__":
    asyncio.run(demo_main())
