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

from PIL import Image, ImageDraw  # noqa: E402

from app.models.schema import VideoAspect  # noqa: E402
from app.services import real_images  # noqa: E402
from app.utils import utils  # noqa: E402

DEFAULT_TOPICS = [
    "GTA 6 new gameplay mechanics",
    "Avengers Doomsday all heroes confirmed",
    "James Webb telescope new discoveries",
]

THUMB_W, THUMB_H = 270, 480
PADDING = 8


def _first_frame(clip_path: str, out_path: str) -> bool:
    cmd = [
        utils.get_ffmpeg_binary(), "-y",
        "-ss", "1", "-i", clip_path,
        "-frames:v", "1", out_path,
    ]
    try:
        subprocess.run(cmd, capture_output=True, timeout=30)
        return os.path.exists(out_path)
    except Exception:
        return False


def _contact_sheet(frames: list[str], title: str, out_path: str) -> None:
    if not frames:
        return
    cols = len(frames)
    sheet = Image.new(
        "RGB",
        (cols * THUMB_W + (cols + 1) * PADDING, THUMB_H + 2 * PADDING + 24),
        (18, 18, 18),
    )
    for i, frame in enumerate(frames):
        try:
            with Image.open(frame) as img:
                img = img.convert("RGB").resize((THUMB_W, THUMB_H), Image.LANCZOS)
                sheet.paste(img, (PADDING + i * (THUMB_W + PADDING), PADDING + 24))
        except Exception:
            continue
    ImageDraw.Draw(sheet).text((PADDING, 6), title, fill=(230, 230, 230))
    sheet.save(out_path)


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
        print(f"selected {len(clips)} clips")

        frames = []
        for i, clip in enumerate(clips):
            frame = os.path.join(work_dir, f"review-{utils.md5(topic)}-{i}.jpg")
            if _first_frame(clip, frame):
                frames.append(frame)

        sheet_path = os.path.join(out_dir, f"{utils.md5(topic)}.png")
        _contact_sheet(frames, topic, sheet_path)
        print(f"contact sheet: {sheet_path}")

    print(f"\nreview the sheets in: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or DEFAULT_TOPICS))
