"""Technical quality gate for downloaded/generated video clips.

Catches the artifacts that make a generated video look obviously broken
regardless of whether the content is "relevant" (that's visual_gate.py's
job): a clip that's mostly black (failed download, slate/intro frame,
fade-to-black segment), a clip that's out-of-focus/heavily compressed
mush, or the same stock clip appearing twice in one video because two
different search terms happened to match it.

Uses only Pillow + numpy (already dependencies via moviepy) rather than
adding OpenCV - a Laplacian-style edge-variance measure via PIL's edge
filter is a well-known lightweight blur proxy that doesn't need a new
~60MB dependency.

Fails open: any clip that can't be read/scored passes through unchanged,
and if every clip in a batch gets rejected (almost certainly a threshold
problem, not genuinely all-bad material), the original unfiltered list is
returned. This is a polish pass, not a hard gate.
"""

from __future__ import annotations

import os
import subprocess
from typing import List

import numpy as np
from loguru import logger
from PIL import Image, ImageFilter

from app.utils import utils

_BLACK_LUMINANCE_THRESHOLD = 12.0  # 0-255 scale; below this frame is near-black
# 用高对比度测试图实测过：清晰帧边缘方差 ~1065，heavy gblur(sigma=20) 之后
# 降到 ~214。真实素材通常没有测试图那么高对比度，所以阈值刻意定得比较低——
# 这一层是加分项而不是硬门槛，漏判比误杀正常但内容本身偏柔和的素材更安全。
_BLUR_VARIANCE_THRESHOLD = 40.0  # edge-variance below this is treated as blurry
_DUPLICATE_HAMMING_THRESHOLD = 4  # aHash bits differing; lower = more similar
_HASH_SIZE = 8


def _extract_frame_image(video_path: str, timestamp: float = 1.0) -> Image.Image | None:
    ffmpeg_binary = utils.get_ffmpeg_binary()
    frame_path = f"{video_path}.qgate-frame.jpg"
    cmd = [
        ffmpeg_binary,
        "-y",
        "-ss", f"{timestamp:.2f}",
        "-i", video_path,
        "-frames:v", "1",
        "-q:v", "2",
        frame_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=15)
        if result.returncode != 0 or not os.path.exists(frame_path):
            return None
        with Image.open(frame_path) as img:
            img.load()
            return img.convert("RGB")
    except Exception as e:
        logger.debug(f"quality_gate: failed to extract frame from {video_path}: {e}")
        return None
    finally:
        if os.path.exists(frame_path):
            try:
                os.remove(frame_path)
            except OSError:
                pass


def _luminance(image: Image.Image) -> float:
    grayscale = image.convert("L")
    return float(np.asarray(grayscale, dtype=np.float32).mean())


def _blur_variance(image: Image.Image) -> float:
    edges = image.convert("L").filter(ImageFilter.FIND_EDGES)
    return float(np.asarray(edges, dtype=np.float32).var())


def _average_hash(image: Image.Image, hash_size: int = _HASH_SIZE) -> str:
    small = image.convert("L").resize((hash_size, hash_size))
    pixels = np.asarray(small, dtype=np.float32)
    avg = pixels.mean()
    bits = (pixels > avg).flatten()
    return "".join("1" if b else "0" for b in bits)


def _hamming_distance(hash_a: str, hash_b: str) -> int:
    return sum(a != b for a, b in zip(hash_a, hash_b))


def filter_low_quality_clips(video_paths: List[str]) -> List[str]:
    """剔除近黑屏、明显模糊、以及和已通过素材重复的片段。

    单个素材抽帧/评分失败时按"放行"处理，避免检测本身的问题连带影响
    正常素材；如果这一批全部被拒绝，大概率是阈值或抽帧环节出了问题，
    此时返回原始未过滤列表，而不是让视频生成失败。
    """
    accepted: List[str] = []
    accepted_hashes: List[str] = []
    rejected_black = 0
    rejected_blur = 0
    rejected_duplicate = 0

    for path in video_paths:
        image = _extract_frame_image(path)
        if image is None:
            accepted.append(path)
            continue

        try:
            luminance = _luminance(image)
            if luminance < _BLACK_LUMINANCE_THRESHOLD:
                rejected_black += 1
                continue

            blur_score = _blur_variance(image)
            if blur_score < _BLUR_VARIANCE_THRESHOLD:
                rejected_blur += 1
                continue

            frame_hash = _average_hash(image)
            is_duplicate = any(
                _hamming_distance(frame_hash, existing) <= _DUPLICATE_HAMMING_THRESHOLD
                for existing in accepted_hashes
            )
            if is_duplicate:
                rejected_duplicate += 1
                continue

            accepted.append(path)
            accepted_hashes.append(frame_hash)
        except Exception as e:
            logger.debug(f"quality_gate: scoring failed for {path}, allowing through: {e}")
            accepted.append(path)

    if not accepted:
        logger.warning(
            "quality_gate: all clips were rejected, which is more likely a "
            "gate malfunction than genuinely all-bad material - keeping the "
            "original unfiltered clips"
        )
        return video_paths

    rejected_total = rejected_black + rejected_blur + rejected_duplicate
    if rejected_total:
        logger.info(
            f"quality_gate: kept {len(accepted)}/{len(video_paths)} clips "
            f"(rejected {rejected_black} near-black, {rejected_blur} blurry, "
            f"{rejected_duplicate} duplicate)"
        )

    return accepted


def _load_image(image_path: str) -> Image.Image | None:
    try:
        with Image.open(image_path) as img:
            img.load()
            return img.convert("RGB")
    except Exception as e:
        logger.debug(f"quality_gate: failed to load image {image_path}: {e}")
        return None


def filter_low_quality_images(image_paths: List[str]) -> List[str]:
    """和 filter_low_quality_clips 同样的黑场/模糊/重复判定，但直接对静态
    图片文件评分，不需要先用 ffmpeg 从视频里抽帧。"""
    accepted: List[str] = []
    accepted_hashes: List[str] = []
    rejected_black = 0
    rejected_blur = 0
    rejected_duplicate = 0

    for path in image_paths:
        image = _load_image(path)
        if image is None:
            accepted.append(path)
            continue

        try:
            luminance = _luminance(image)
            if luminance < _BLACK_LUMINANCE_THRESHOLD:
                rejected_black += 1
                continue

            blur_score = _blur_variance(image)
            if blur_score < _BLUR_VARIANCE_THRESHOLD:
                rejected_blur += 1
                continue

            frame_hash = _average_hash(image)
            is_duplicate = any(
                _hamming_distance(frame_hash, existing) <= _DUPLICATE_HAMMING_THRESHOLD
                for existing in accepted_hashes
            )
            if is_duplicate:
                rejected_duplicate += 1
                continue

            accepted.append(path)
            accepted_hashes.append(frame_hash)
        except Exception as e:
            logger.debug(f"quality_gate: scoring failed for {path}, allowing through: {e}")
            accepted.append(path)

    if not accepted:
        logger.warning(
            "quality_gate: all images were rejected, which is more likely a "
            "gate malfunction than genuinely all-bad material - keeping the "
            "original unfiltered images"
        )
        return image_paths

    rejected_total = rejected_black + rejected_blur + rejected_duplicate
    if rejected_total:
        logger.info(
            f"quality_gate: kept {len(accepted)}/{len(image_paths)} images "
            f"(rejected {rejected_black} near-black, {rejected_blur} blurry, "
            f"{rejected_duplicate} duplicate)"
        )

    return accepted


def rank_by_visual_interest(video_paths: List[str]) -> List[str]:
    """把边缘方差（细节/对比度的粗略代理指标）最高的一段素材挪到最前面。

    第一个镜头就是观众决定要不要划走的那一下——用视觉上最"抓眼"的素材
    开场，比不管三七二十一使用搜索结果的第一条更能留住观众。评分拿不到
    的素材保持原有相对顺序，不因为抽帧失败被强制排到后面或前面。
    """
    if len(video_paths) <= 1:
        return video_paths

    scored = []
    unscored = []
    for path in video_paths:
        image = _extract_frame_image(path)
        if image is None:
            unscored.append(path)
            continue
        try:
            score = _blur_variance(image)
            scored.append((score, path))
        except Exception as e:
            logger.debug(f"quality_gate: scoring failed for {path}: {e}")
            unscored.append(path)

    if not scored:
        return video_paths

    scored.sort(key=lambda item: item[0], reverse=True)
    best_path = scored[0][1]
    rest = [path for path in video_paths if path != best_path]
    return [best_path] + rest
