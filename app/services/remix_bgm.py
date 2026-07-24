"""Re-mix a finished video's background music at a different volume.

Music sitting too loud is an audio-only problem, but the obvious fix -
regenerate the video - throws away several minutes of sourcing and encoding to
change one number. The narration and the music are still on disk as separate
files next to the output, so the mix can be rebuilt and swapped in instead:
the picture (with its burned-in subtitles) is stream-copied untouched, so
there is no quality loss and it finishes in seconds.

Reuses the pipeline's own ducking + two-pass loudness normalisation, so the
result matches what the audio would have been had it rendered at the lower
volume in the first place.
"""

from __future__ import annotations

import glob
import os
import subprocess
from typing import List

from loguru import logger

from app.services import video as video_service
from app.utils import utils

# 输出文件的后缀。也用来把本模块自己的产物排除在输入之外，否则重复调用会对着
# 上一次的结果再混一遍，堆出 -quietbgm-quietbgm 这种层层叠加的文件。
OUTPUT_SUFFIX = "-quietbgm"


def _narration_duration(path: str) -> float:
    try:
        from moviepy.audio.io.AudioFileClip import AudioFileClip

        with AudioFileClip(path) as clip:
            return float(clip.duration)
    except Exception:
        return 0.0


def find_source_videos(task_dir: str) -> List[str]:
    """任务目录里可以重新混音的成片（排除本模块自己的输出）。"""
    return sorted(
        p
        for p in glob.glob(os.path.join(task_dir, "final-*.mp4"))
        if OUTPUT_SUFFIX not in os.path.basename(p)
    )


def latest_bgm_file() -> str | None:
    songs = sorted(
        glob.glob(os.path.join(utils.song_dir(), "*.mp3")),
        key=os.path.getmtime,
        reverse=True,
    )
    return songs[0] if songs else None


def remix_task_bgm(
    task_id: str, bgm_volume: float, bgm_path: str | None = None
) -> List[str]:
    """按新的背景音乐音量重混任务成片，返回新生成的文件路径列表。

    找不到旁白音频、成片或背景音乐时返回空列表，并记录原因——这是一个事后
    修补操作，失败不该抛异常打断界面。
    """
    task_dir = utils.task_dir(task_id)
    narration = os.path.join(task_dir, "audio.mp3")
    if not os.path.exists(narration):
        logger.warning(f"remix_bgm: no narration audio in {task_dir}")
        return []

    videos = find_source_videos(task_dir)
    if not videos:
        logger.warning(f"remix_bgm: no final-*.mp4 in {task_dir}")
        return []

    bgm = bgm_path or latest_bgm_file()
    if not bgm or not os.path.exists(bgm):
        logger.warning("remix_bgm: no background music file available")
        return []

    duration = _narration_duration(narration)
    mixed = os.path.join(task_dir, "temp-remix.mp3")
    if not video_service._mix_narration_with_ducked_bgm(
        narration, bgm, bgm_volume, duration, mixed
    ):
        logger.warning("remix_bgm: re-mix failed (ducking/loudnorm unavailable)")
        return []

    ffmpeg = utils.get_ffmpeg_binary()
    written: List[str] = []
    for src in videos:
        out = src.replace(".mp4", f"{OUTPUT_SUFFIX}.mp4")
        cmd = [
            ffmpeg, "-y",
            "-i", src,
            "-i", mixed,
            "-map", "0:v:0", "-map", "1:a:0",
            # 画面（含已烧录字幕）直接流拷贝，不重新编码：又快又不掉画质。
            "-c:v", "copy",
            # 混音链出来的是单声道 48k，和原成片的 44.1k 立体声对不上；显式
            # 指定，免得替换完音轨反而把声道数/采样率改掉。
            "-c:a", "aac", "-b:a", "192k", "-ac", "2", "-ar", "44100",
            "-shortest",
            out,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=300)
            if result.returncode == 0 and os.path.exists(out):
                written.append(out)
                logger.info(f"remix_bgm: wrote {out}")
            else:
                logger.warning(
                    f"remix_bgm: ffmpeg failed for {src}: "
                    f"{(result.stderr or b'')[-300:]}"
                )
        except Exception as e:
            logger.warning(f"remix_bgm: failed for {src}: {e}")

    if os.path.exists(mixed):
        try:
            os.remove(mixed)
        except OSError:
            pass

    return written
