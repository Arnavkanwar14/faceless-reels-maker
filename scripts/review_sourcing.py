"""Contact sheet for whatever the sourcing pipeline just picked.

Every gate in real_images (sharpness floor, border crop, watermark/generic
checks, official-channel preference) is tuned by threshold, and thresholds
picked against one topic's handful of samples quietly over- or under-filter
on the next topic. Logs don't help here - "kept 2/11" reads fine whether the
two are perfect or useless. The only real check is looking at them.

This runs sourcing for a few topics and writes one PNG per topic showing the
frames that were actually selected, so reviewing a threshold change takes a
few seconds of looking instead of a full video render.

    python scripts/review_sourcing.py                  # default topics
    python scripts/review_sourcing.py "topic a" "topic b"

Writes to storage/sourcing_review/.
"""

from __future__ import annotations

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.schema import VideoAspect  # noqa: E402
from app.services import quality_gate, real_images, review_sheet  # noqa: E402
from app.utils import utils  # noqa: E402

DEFAULT_TOPICS = [
    "GTA 6 new gameplay mechanics",
    "Avengers Doomsday all heroes confirmed",
    "James Webb telescope new discoveries",
]

def main(topics: list[str]) -> int:
    out_dir = utils.storage_dir("sourcing_review", create=True)
    work_dir = utils.storage_dir("cache_videos", create=True)

    for topic in topics:
        print(f"\n=== {topic} ===")
        clips = real_images.download_real_image_clips(
            task_id="review",
            search_terms=[topic],
            video_subject=topic,
            video_aspect=VideoAspect.portrait,
            audio_duration=20.0,
            max_clip_duration=5,
            material_directory=work_dir,
        )
        distinct = quality_gate.count_distinct_shots(clips) if clips else 0
        print(f"selected {len(clips)} clips ({distinct} visually distinct)")

        sheet_path = review_sheet.build_contact_sheet(
            clips, f"{topic}  -  {distinct} distinct shot(s)",
            os.path.join(out_dir, f"{utils.md5(topic)}.png"),
        )
        print(f"contact sheet: {sheet_path}")

    print(f"\nreview the sheets in: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or DEFAULT_TOPICS))
