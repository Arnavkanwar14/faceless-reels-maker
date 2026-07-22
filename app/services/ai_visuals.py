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
import random
import subprocess
import urllib.parse
from typing import List

import requests
from loguru import logger

from app.models.schema import VideoAspect
from app.utils import utils

_POLLINATIONS_TIMEOUT = 60

# 每种运镜方式给出独立的 zoompan 表达式：缩放边界固定裁到目标画幅
# （见 image_to_ken_burns_clip 的 scale+crop 步骤），这里只负责起止焦距
# 和推进方向，让多张静态图连续出现时不会每张都是同一个运镜观感。
_KEN_BURNS_STYLES = (
    # zoom in, centered
    "zoompan=z='min(zoom+0.0008,1.1)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s={w}x{h}:fps={fps}",
    # zoom out, centered
    "zoompan=z='if(eq(on,0),1.1,max(zoom-0.0008,1.0))':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s={w}x{h}:fps={fps}",
    # slow pan left -> right while holding zoom steady
    "zoompan=z='1.08':x='if(eq(on,0),0,x+1.2)':y='ih/2-(ih/zoom/2)':d={frames}:s={w}x{h}:fps={fps}",
    # slow pan right -> left while holding zoom steady
    "zoompan=z='1.08':x='if(eq(on,0),iw-iw/zoom,x-1.2)':y='ih/2-(ih/zoom/2)':d={frames}:s={w}x{h}:fps={fps}",
)


def _generate_image(
    prompt: str, width: int, height: int, output_path: str, enhance: bool = True
) -> bool:
    encoded_prompt = urllib.parse.quote(prompt)
    params = {
        "width": str(width),
        "height": str(height),
        "model": "flux",
        "nologo": "true",
        "safe": "false",
        "enhance": "true" if enhance else "false",
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


def image_to_ken_burns_clip(
    image_path: str, output_path: str, duration: int, width: int, height: int
) -> bool:
    """用 ffmpeg zoompan 把静态图片转成带缓慢推进效果的短视频片段。

    源图片尺寸/长宽比可能和目标画幅完全不同（尤其是网上下载的真实图片，
    不像 AI 生成图那样能控制尺寸），所以必须先用
    force_original_aspect_ratio=increase + crop 把图片"填满再裁切"到目标
    画幅，再进入 zoompan。任何一步只用 scale 而不裁切，都会在长宽比不匹配
    时把图片拉伸变形。
    """
    ffmpeg_binary = utils.get_ffmpeg_binary()
    fps = 30
    total_frames = duration * fps
    # 先裁到目标画幅，再放大到 1.15 倍留出推进余量，避免 zoompan 边缘黑边。
    zoom_w, zoom_h = int(width * 1.15), int(height * 1.15)

    zoompan_expr = random.choice(_KEN_BURNS_STYLES).format(
        frames=total_frames, w=width, h=height, fps=fps
    )

    filter_complex = (
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},"
        f"scale={zoom_w}:{zoom_h},"
        f"{zoompan_expr},"
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


def _generation_size(width: int, height: int, max_side: int = 1344) -> tuple[int, int]:
    """按目标画幅的真实长宽比算生成尺寸，长边不超过 max_side。

    之前固定用 min(width, 768) x min(height, 1344)：横屏 1920x1080 会算出
    768x1080（约 0.71:1），和目标 16:9（1.78:1）完全对不上，生成阶段就已经
    是错的长宽比，后续任何 scale 都只能在“裁掉大半画面”和“继续拉伸”之间
    选一个。这里先按比例收缩到 max_side，让生成图和目标画幅同一个长宽比，
    Ken Burns 阶段的 crop 就只需要裁掉一点点边缘，而不是硬拉伸。
    """
    scale = max_side / max(width, height)
    gen_w = max(64, int(width * scale) // 8 * 8)
    gen_h = max(64, int(height * scale) // 8 * 8)
    return gen_w, gen_h


def generate_ai_visual_clips(
    task_id: str,
    search_terms: List[str],
    video_subject: str,
    video_aspect: VideoAspect,
    max_clip_duration: int = 5,
    material_directory: str = "",
    is_named_person: bool | None = None,
) -> List[str]:
    """为每个搜索词生成一张 AI 图片，再转成 Ken Burns 短片段。"""
    width, height = video_aspect.to_resolution()
    gen_width, gen_height = _generation_size(width, height)

    if width >= height:
        composition = "wide cinematic composition"
    else:
        composition = "vertical composition"

    # "hyperrealistic 3D render" 是导致人脸发蜡、发假的主要提示词——那正是
    # 引导模型走向 3D/CG 渲染风格；改成摄影语言能明显改善真实感。非真人
    # 主题额外引导成远景/环境镜头：免费 FLUX 生成的特写人脸最容易穿帮，
    # 环境/物体镜头则往往足够以假乱真。
    style = "professional photograph, natural lighting, sharp focus, shallow depth of field"
    if not is_named_person:
        style += ", wide environmental shot, no close-up faces"

    output_dir = material_directory or utils.storage_dir("cache_videos")
    os.makedirs(output_dir, exist_ok=True)

    video_paths = []
    for index, term in enumerate(search_terms):
        prompt = f"{term}, related to {video_subject}, {style}, highly detailed, {composition}"
        image_path = os.path.join(output_dir, f"aivis-{task_id}-{index}.png")
        clip_path = os.path.join(output_dir, f"aivis-{task_id}-{index}.mp4")

        if os.path.exists(clip_path) and os.path.getsize(clip_path) > 0:
            video_paths.append(clip_path)
            continue

        if not _generate_image(prompt, gen_width, gen_height, image_path):
            continue

        if image_to_ken_burns_clip(
            image_path, clip_path, max_clip_duration, width, height
        ):
            video_paths.append(clip_path)
            logger.info(f"ai visuals: generated clip for '{term}'")

    logger.success(f"generated {len(video_paths)} ai visual clips")
    return video_paths
