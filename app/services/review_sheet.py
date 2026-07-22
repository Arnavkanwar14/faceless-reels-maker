"""Contact sheet of the clips a run actually selected.

Logs cannot answer "do these look right". A line reading "kept 2/29" is
identical whether the two survivors are perfect or useless - during
development that exact line hid four near-duplicate posters, branded key
art, and a shot repeated four times. One glance at the frames catches all of
it, so the sheet is written automatically next to every generated video
rather than being a command someone has to remember.

Never raises: a failed sheet must not fail a video that rendered fine.
"""

from __future__ import annotations

import os
import subprocess
from typing import List

from loguru import logger
from PIL import Image, ImageDraw

from app.utils import utils

_THUMB_W, _THUMB_H = 270, 480
_PADDING = 8
_TITLE_HEIGHT = 24
_MAX_TILES = 10


def _first_frame(clip_path: str, out_path: str, timestamp: float = 1.0) -> bool:
    cmd = [
        utils.get_ffmpeg_binary(), "-y",
        "-ss", f"{timestamp:.2f}",
        "-i", clip_path,
        "-frames:v", "1",
        out_path,
    ]
    try:
        subprocess.run(cmd, capture_output=True, timeout=30)
        return os.path.exists(out_path) and os.path.getsize(out_path) > 0
    except Exception:
        return False


def build_contact_sheet(clip_paths: List[str], title: str, output_path: str) -> str | None:
    """把每段素材的一帧拼成一张总览图，返回写出的路径（失败返回 None）。"""
    if not clip_paths:
        return None

    clips = clip_paths[:_MAX_TILES]
    work_dir = os.path.dirname(output_path) or "."
    os.makedirs(work_dir, exist_ok=True)

    frames = []
    for i, clip in enumerate(clips):
        frame_path = os.path.join(work_dir, f".sheet-frame-{utils.md5(clip)}-{i}.jpg")
        if _first_frame(clip, frame_path):
            frames.append(frame_path)

    if not frames:
        return None

    try:
        cols = len(frames)
        sheet = Image.new(
            "RGB",
            (
                cols * _THUMB_W + (cols + 1) * _PADDING,
                _THUMB_H + 2 * _PADDING + _TITLE_HEIGHT,
            ),
            (18, 18, 18),
        )
        for i, frame in enumerate(frames):
            try:
                with Image.open(frame) as img:
                    thumb = img.convert("RGB").resize((_THUMB_W, _THUMB_H), Image.LANCZOS)
                    sheet.paste(
                        thumb,
                        (_PADDING + i * (_THUMB_W + _PADDING), _PADDING + _TITLE_HEIGHT),
                    )
            except Exception:
                continue
        ImageDraw.Draw(sheet).text((_PADDING, 6), title[:120], fill=(230, 230, 230))
        sheet.save(output_path)
        return output_path
    except Exception as e:
        logger.debug(f"review sheet: failed to build for '{title}': {e}")
        return None
    finally:
        for frame in frames:
            try:
                os.remove(frame)
            except OSError:
                pass


def write_for_task(clip_paths: List[str], subject: str, output_dir: str) -> None:
    """在成片目录里放一张素材总览图。任何失败都只记日志，不影响生成流程。"""
    try:
        path = build_contact_sheet(
            clip_paths, subject, os.path.join(output_dir, "materials-review.png")
        )
        if path:
            logger.info(f"materials contact sheet written: {path}")
    except Exception as e:
        logger.debug(f"review sheet: skipped ({e})")
