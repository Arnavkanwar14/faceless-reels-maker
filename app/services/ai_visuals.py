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
#
# 关键：推进速度按总帧数算，而不是写死每帧增量。写死增量的话，运镜会在
# 固定帧数内走完然后卡住不动——5s 的片段刚好，但素材不够、需要把每张图
# 拉长到 15s 铺满时间线时，就变成"动 4 秒、静止 11 秒"。用 {zstep}/{xstep}
# 让整个推进正好铺满整个片段，多长的片段都是一路匀速在动。
_KEN_BURNS_TOTAL_ZOOM = 0.12  # 全程缩放幅度：1.0 -> 1.12
_KEN_BURNS_PAN_ZOOM = 1.08    # 平移时保持的固定焦距

_KEN_BURNS_STYLES = (
    # zoom in, centered
    "zoompan=z='min(zoom+{zstep},1.12)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s={w}x{h}:fps={fps}",
    # zoom out, centered
    "zoompan=z='if(eq(on,0),1.12,max(zoom-{zstep},1.0))':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s={w}x{h}:fps={fps}",
    # slow pan left -> right while holding zoom steady
    "zoompan=z='1.08':x='if(eq(on,0),0,x+{xstep})':y='ih/2-(ih/zoom/2)':d={frames}:s={w}x{h}:fps={fps}",
    # slow pan right -> left while holding zoom steady
    "zoompan=z='1.08':x='if(eq(on,0),iw-iw/zoom,x-{xstep})':y='ih/2-(ih/zoom/2)':d={frames}:s={w}x{h}:fps={fps}",
)

# 只缩不移的运镜。主体偏离画面中心时用这一组：平移会把主体推出画外，纯缩放
# 始终围绕画面中心推进，主体一直留在框内。
_KEN_BURNS_ZOOM_STYLES = (_KEN_BURNS_STYLES[0], _KEN_BURNS_STYLES[1])


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


def _aspect_mismatch_ratio(image_path: str, width: int, height: int) -> float:
    """源图长宽比和目标画幅差多少倍。1.0 表示完全一致。"""
    try:
        from PIL import Image

        with Image.open(image_path) as img:
            src_w, src_h = img.size
    except Exception:
        return 1.0
    if not src_w or not src_h or not width or not height:
        return 1.0
    src_aspect = src_w / src_h
    target_aspect = width / height
    return max(src_aspect / target_aspect, target_aspect / src_aspect)


# 长宽比差异超过这个倍数时，改用"模糊铺底 + 完整画面"而不是裁切填满。
#
# 把 16:9 的横图塞进 9:16 竖屏，填满式裁切只能保留大约 31% 的画面宽度，而且
# 是闭着眼睛从正中间裁——主体稍微偏一点就被切没了，剩下的部分还要再放大
# 1.78 倍，本来清晰的图也会被放糊。这正是"选出来的图看着啥也没有、还很糊"
# 的直接原因。视频素材那条路早就改成模糊铺底了（见 video.py 的
# _make_blurred_fill_background），静态图这里一直还在做破坏性裁切。
_MAX_CROP_ASPECT_MISMATCH = 1.35


def image_to_ken_burns_clip(
    image_path: str, output_path: str, duration: int, width: int, height: int
) -> bool:
    """用 ffmpeg zoompan 把静态图片转成带缓慢推进效果的短视频片段。

    长宽比接近目标画幅时，走"填满再裁切"，裁掉的只是一点边缘；差得多时
    （典型的横图进竖屏）改成"完整画面居中 + 自身模糊放大铺底"，保证画面
    内容一张不少地呈现出来，而不是只剩中间三分之一。
    """
    ffmpeg_binary = utils.get_ffmpeg_binary()
    fps = 30
    total_frames = max(1, duration * fps)
    # 先裁到目标画幅，再放大到 1.15 倍留出推进余量，避免 zoompan 边缘黑边。
    zoom_w, zoom_h = int(width * 1.15), int(height * 1.15)

    # 把整段推进按总帧数摊开：缩放在整段里从 1.0 匀速走到 1.12；平移的每帧
    # 步长用 ffmpeg 变量表达（iw、zoom 在渲染时才知道具体值），让平移正好在
    # 片段结束时走到画面另一端。这样 5s 和 15s 的片段都是一路匀速在动，不会
    # 出现"前几秒动、后面卡住"。
    zstep = _KEN_BURNS_TOTAL_ZOOM / total_frames
    xstep = f"(iw-iw/{_KEN_BURNS_PAN_ZOOM})/{total_frames}"

    # 估计主体焦点，用来决定裁切窗口的位置和运镜方式。读图失败时退回正中。
    fx, fy = 0.5, 0.5
    try:
        from PIL import Image

        from app.services import quality_gate

        with Image.open(image_path) as _img:
            fx, fy = quality_gate.salient_focus(_img)
    except Exception:
        pass

    # 主体明显偏离画面中心时，平移运镜有可能在推进过程中把主体推出画外；
    # 这种情况改用居中缩放（zoom in/out），只缩不移，主体始终在框内。主体
    # 大致居中时才允许用平移，增加镜头语言的多样性。
    off_center = abs(fx - 0.5) > 0.18 or abs(fy - 0.5) > 0.18
    style_pool = _KEN_BURNS_ZOOM_STYLES if off_center else _KEN_BURNS_STYLES
    zoompan_expr = random.choice(style_pool).format(
        frames=total_frames, w=width, h=height, fps=fps,
        zstep=f"{zstep:.6f}", xstep=xstep,
    )

    if _aspect_mismatch_ratio(image_path, width, height) <= _MAX_CROP_ASPECT_MISMATCH:
        # 填满式裁切：把裁切窗口对准主体焦点，而不是死板取正中。in_w/in_h 是
        # 缩放后（crop 的输入）尺寸，(in_w-W) 是水平方向多出来、要被裁掉的量，
        # 乘 fx 决定往哪边裁——fx=0.5 等于居中，fx 偏小则保留左侧/顶部，主体
        # （尤其是头部）不会被切掉。
        filter_complex = (
            f"scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height}:(in_w-{width})*{fx:.3f}:(in_h-{height})*{fy:.3f},"
            f"scale={zoom_w}:{zoom_h},"
            f"{zoompan_expr},"
            "format=yuv420p"
        )
    else:
        # 前景：完整画面按比例缩放进画幅内（decrease，不裁切）。
        # 背景：同一张图放大铺满后高斯模糊，垫在前景后面填掉上下空白。
        # 两者叠加后再统一走 zoompan，运镜作用在合成结果上而不是原图，
        # 前景不会在推进过程中被推出画外。
        filter_complex = (
            f"[0:v]scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},gblur=sigma=20[bg];"
            f"[0:v]scale={width}:{height}:force_original_aspect_ratio=decrease[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2,"
            f"scale={zoom_w}:{zoom_h},"
            f"{zoompan_expr},"
            "format=yuv420p"
        )

    # 模糊铺底那条分支用到了具名流（[bg]/[fg]），必须走 -filter_complex；
    # 简单裁切分支是单链滤镜，-vf 就够了。
    filter_flag = "-filter_complex" if "[bg]" in filter_complex else "-vf"

    cmd = [
        ffmpeg_binary,
        "-y",
        "-loop", "1",
        "-i", image_path,
        "-t", str(duration),
        filter_flag, filter_complex,
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
