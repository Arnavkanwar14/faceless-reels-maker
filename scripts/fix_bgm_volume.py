"""Re-mix an already-rendered video's background music at a different volume.

Command-line front end for app.services.remix_bgm - see that module for why
this beats regenerating the whole video. The picture is stream-copied, so it
finishes in seconds with no quality loss.

    python scripts/fix_bgm_volume.py <task_id> 0.08
    python scripts/fix_bgm_volume.py <task_id> 0.08 --bgm "path/to/song.mp3"

Replaces the video file(s) in place - run it again with a different volume
to re-adjust; there is no separate copy to clean up.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services import remix_bgm  # noqa: E402


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 1

    task_id, volume = argv[0], float(argv[1])
    bgm_override = argv[argv.index("--bgm") + 1] if "--bgm" in argv else None

    written = remix_bgm.remix_task_bgm(task_id, volume, bgm_override)
    if not written:
        print("nothing was updated - see the log above for the reason")
        return 1

    for path in written:
        print(f"updated {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
