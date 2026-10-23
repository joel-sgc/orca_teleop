"""Manual step-through gating for teleop, driven by raw terminal keypresses.

A plain terminal has no "key released" event, so there's no direct way to
tell a tap from a hold. The trick used here: treat every keypress
*character* received as one "advance" request. A tap sends one character,
so it advances exactly one step. A held key relies on the OS/terminal's own
key-repeat — which sends a fast stream of the same character for as long as
it's held — so it advances continuously for free, at whatever rate the
terminal's repeat setting delivers. No GUI/X11 dependency, so this works
over SSH.
"""

from __future__ import annotations

import logging
import select
import sys
import threading

logger = logging.getLogger(__name__)

_POLL_TIMEOUT_S = 0.05


class KeyGate:
    """Tracks whether ``key`` has been pressed since the last :meth:`consume`.

    Falls back to always-open (never actually gating) when stdin isn't a
    terminal, so a non-interactive run (piped input, a background process)
    streams freely instead of hanging forever waiting for a keypress that
    can never arrive.
    """

    def __init__(self, key: str = " ") -> None:
        self._key = key
        self._stop = threading.Event()
        self._advance = threading.Event()
        self._thread: threading.Thread | None = None
        self._always_open = False

    def start(self) -> None:
        if not sys.stdin.isatty():
            logger.warning(
                "stdin is not a terminal; step gating disabled (streaming freely)."
            )
            self._always_open = True
            return
        self._thread = threading.Thread(target=self._read_loop, name="key-gate", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def consume(self) -> bool:
        """Return True, and clear the pending flag, if `key` was pressed
        since the last call to :meth:`consume`."""
        if self._always_open:
            return True
        if self._advance.is_set():
            self._advance.clear()
            return True
        return False

    def _read_loop(self) -> None:
        import termios
        import tty

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)  # no line buffering, no echo; Ctrl+C still raises SIGINT
            while not self._stop.is_set():
                ready, _, _ = select.select([sys.stdin], [], [], _POLL_TIMEOUT_S)
                if ready and sys.stdin.read(1) == self._key:
                    self._advance.set()
        except Exception:
            logger.exception("Key gate reader failed; streaming freely instead.")
            self._always_open = True
        finally:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            except Exception:
                pass
