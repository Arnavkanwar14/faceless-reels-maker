"""AI-generated image clips for subjects with no real-world footage.

Fictional characters, game/anime IP, brand mascots, etc. don't exist in any
stock or YouTube library. For these subjects the pipeline generates one image
per search term via Pollinations (free, keyless FLUX) and turns it into a
short video clip with Ken Burns pan/zoom, so the rest of the pipeline
(video.combine_videos, subtitles, audio mixing) can treat it exactly like any
other downloaded material - no changes needed downstream.
"""

from __future__ import annotations

import os
import subprocess
import urllib.parse
from typing import List

import requests
from loguru import logger

from app.models.schema import VideoAspect
from app.utils import utils

_POLLINATIONS_TIMEOUT = 60


def _generate_image(prompt: str, width: int, height: int, output_path: str) -> bool:
    encoded_prompt = urllib.parse.quote(prompt)
    params = {
        "width": str(width),
        "height": str(height),
        "model": "flux",
        "nologo": "true",
        "safe": "false",
    }
    url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?{urllib.parse.urlencode(params)}"

    try:
        response = requests.get(url, timeout=_POLLINATIONS_TIMEOUT)
        response.raise_for_status()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(response.content)
        return os.path.exists(output_path) and os.path.getsize(output_path) > 0
    except Exception as e:
        logger.error(f"pollinations image generation failed for '{prompt}': {e}")
        return False


def _image_to_ken_burns_clip(
    image_path: str, output_path: str, duration: int, width: int, height: int
) -> bool:
    """用 ffmpeg zoompan 把静态图片转成带缓慢推进效果的短视频片段。"""
    ffmpeg_binary = utils.get_ffmpeg_binary()
    fps = 30
    total_frames = duration * fps
    # 缩放到目标分辨率的 1.15 倍再推进，避免 zoompan 边缘出现黑边。
    scale_w, scale_h = int(width * 1.15), int(height * 1.15)

    filter_complex = (
        f"scale={scale_w}:{scale_h},"
        f"zoompan=z='min(zoom+0.0008,1.1)':d={total_frames}:s={width}x{height}:fps={fps},"
        "format=yuv420p"
    )

    cmd = [
        ffmpeg_binary,
        "-y",
        "-loop", "1",
        "-i", image_path,
        "-t", str(duration),
        "-vf", filter_complex,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        output_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            logger.error(f"ffmpeg Ken Burns render failed: {result.stderr[-500:]}")
            return False
        return os.path.exists(output_path) and os.path.getsize(output_path) > 0
    except Exception as e:
        logger.error(f"ffmpeg Ken Burns render failed: {e}")
        return False


def generate_ai_visual_clips(
    task_id: str,
    search_terms: List[str],
    video_subject: str,
    video_aspect: VideoAspect,
    max_clip_duration: int = 5,
    material_directory: str = "",
) -> List[str]:
    """为每个搜索词生成一张 AI 图片，再转成 Ken Burns 短片段。"""
    width, height = video_aspect.to_resolution()
    # 生成阶段用较小分辨率换取速度，Ken Burns 渲染时再放大到目标分辨率。
    gen_width = min(width, 768)
    gen_height = min(height, 1344)

    output_dir = material_directory or utils.storage_dir("cache_videos")
    os.makedirs(output_dir, exist_ok=True)

    video_paths = []
    for index, term in enumerate(search_terms):
        prompt = (
            f"{term}, related to {video_subject}, hyperrealistic 3D render, "
            "cinematic lighting, highly detailed, vertical composition"
        )
        image_path = os.path.join(output_dir, f"aivis-{task_id}-{index}.png")
        clip_path = os.path.join(output_dir, f"aivis-{task_id}-{index}.mp4")

        if os.path.exists(clip_path) and os.path.getsize(clip_path) > 0:
            video_paths.append(clip_path)
            continue

        if not _generate_image(prompt, gen_width, gen_height, image_path):
            continue

        if _image_to_ken_burns_clip(
            image_path, clip_path, max_clip_duration, width, height
        ):
            video_paths.append(clip_path)
            logger.info(f"ai visuals: generated clip for '{term}'")

    logger.success(f"generated {len(video_paths)} ai visual clips")
    return video_paths
