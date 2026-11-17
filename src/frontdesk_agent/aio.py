"""Event loop setup that also works on Windows.

psycopg's async mode cannot run on the ProactorEventLoop, which is the
default on Windows (the dev machine), so every entry point that starts a loop
goes through here. On Linux (CI, production) this is plain ``asyncio.run``.
"""

import asyncio
import sys
from collections.abc import Coroutine
from typing import Any


def run[T](main: Coroutine[Any, Any, T]) -> T:
    """``asyncio.run`` on a loop psycopg can use."""
    if sys.platform == "win32":
        return asyncio.run(main, loop_factory=asyncio.SelectorEventLoop)
    return asyncio.run(main)


def use_selector_loop_policy() -> None:
    """Make loops created by others (pytest-asyncio, anyio) selector loops on Windows."""
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
