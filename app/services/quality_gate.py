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


def salient_focus(image: Image.Image) -> tuple[float, float]:
    """估计画面主体的焦点位置，返回归一化坐标 (fx, fy)，范围都是 [0,1]。

    用途：裁切/运镜时把窗口对准主体，而不是闭着眼睛裁正中间——正中间裁法
    最常见的翻车就是把人物的头裁掉（头一般在竖图偏上的位置）。

    原理：人脸和主体细节（五官、边缘）的边缘响应远高于天空、纯色背景这类
    区域，所以边缘能量的重心通常就落在主体上。纯 Pillow+numpy，无需人脸
    检测模型或额外依赖。

    再叠加一点"上偏"先验：同等条件下主体（尤其是头部）更可能在画面上半部，
    给上半部一点额外权重，让焦点更稳地压在头/脸而不是躯干。
    """
    try:
        # 缩到小图算，既快又能抹掉高频噪声对重心的干扰。
        small = image.convert("L").resize((64, 64))
        edges = np.asarray(small.filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    except Exception:
        return 0.5, 0.5

    total = float(edges.sum())
    if total <= 0:
        return 0.5, 0.5

    ys, xs = np.mgrid[0:edges.shape[0], 0:edges.shape[1]]
    # 上偏先验：从上到下权重 1.15 -> 0.85，轻微把重心往上拉。
    row_bias = np.linspace(1.15, 0.85, edges.shape[0]).reshape(-1, 1)
    weighted = edges * row_bias
    wtotal = float(weighted.sum()) or total

    fx = float((weighted * xs).sum() / wtotal) / (edges.shape[1] - 1)
    fy = float((weighted * ys).sum() / wtotal) / (edges.shape[0] - 1)
    return min(max(fx, 0.0), 1.0), min(max(fy, 0.0), 1.0)


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


# 归一化尺寸下的清晰度下限。
#
# 直接用像素尺寸判断素材好坏是不够的：网上很多"1600x900"的图其实是把一张
# 480p 的截图拉大再存一次的产物，尺寸达标但完全没有细节，放进竖屏成片里
# 就是一团糊，观众根本看不出画面里是什么。
#
# _blur_variance 的绝对值会随图片尺寸变化，所以先统一缩放到同一个长边再测，
# 这样不同来源、不同尺寸的图之间才可比。实测值：
#   448x252 的糊图              ~790
#   被判定"看不出是什么"的软图   ~1580
#   正片抽出的原生 1080p 帧      ~2790
#   路牌清晰可读的游戏截图       ~3600
#   官方宣传图                   ~4070
# 阈值取在 1580 和 2790 之间。
_MIN_NORMALIZED_SHARPNESS = 2200.0
_SHARPNESS_NORMALIZE_SIDE = 720


def normalized_sharpness(image_path: str) -> float | None:
    """把图片统一缩放到固定长边后测边缘方差，作为跨尺寸可比的清晰度指标。

    读取失败返回 None，由调用方决定怎么处理（默认按放行处理，不因为测量
    本身出问题就丢掉素材）。
    """
    try:
        with Image.open(image_path) as img:
            image = img.convert("RGB")
            scale = _SHARPNESS_NORMALIZE_SIDE / max(image.size)
            if scale < 1:
                image = image.resize(
                    (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                    Image.LANCZOS,
                )
            return _blur_variance(image)
    except Exception as e:
        logger.debug(f"quality_gate: sharpness measure failed for {image_path}: {e}")
        return None


def is_sharp_enough(
    image_path: str, minimum: float = _MIN_NORMALIZED_SHARPNESS
) -> bool:
    """判断图片是否有足够的真实细节，而不只是尺寸达标。"""
    score = normalized_sharpness(image_path)
    if score is None:
        return True
    return score >= minimum


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


def dedupe_clips(video_paths: List[str]) -> List[str]:
    """只做去重：剔除画面和前面某一段几乎一样的片段，不做明暗/模糊判断。

    filter_low_quality_clips 不适合这个场景：从预告片里切出来的镜头本来就
    可能整体很暗（夜戏、暗色调），会被近黑判定整批拒掉，然后触发"全被拒就
    全部放行"的兜底，结果一条都没去掉。这里只比较感知哈希，暗不暗不影响
    判断。

    抽帧失败的片段一律保留，不因为检测本身的问题丢素材。
    """
    kept: List[str] = []
    kept_hashes: List[str] = []

    for path in video_paths:
        image = _extract_frame_image(path)
        if image is None:
            kept.append(path)
            continue
        try:
            frame_hash = _average_hash(image)
        except Exception:
            kept.append(path)
            continue
        if any(
            _hamming_distance(frame_hash, existing) <= _DUPLICATE_HAMMING_THRESHOLD
            for existing in kept_hashes
        ):
            continue
        kept_hashes.append(frame_hash)
        kept.append(path)

    return kept


def count_distinct_shots(video_paths: List[str]) -> int:
    """这批素材里实际有几个互不相同的画面。

    "有几段素材"和"观众能看出几个不同画面"是两回事：同一个长镜头切出来的
    四段、或者同一张海报的四个转载版本，数量上是 4，观感上是 1。判断素材
    够不够时，该看的是后者。

    抽帧失败的按各自独立计数，避免因为检测问题低估可用素材。
    """
    hashes: List[str] = []
    unreadable = 0

    for path in video_paths:
        image = _extract_frame_image(path)
        if image is None:
            unreadable += 1
            continue
        try:
            frame_hash = _average_hash(image)
        except Exception:
            unreadable += 1
            continue
        if not any(
            _hamming_distance(frame_hash, existing) <= _DUPLICATE_HAMMING_THRESHOLD
            for existing in hashes
        ):
            hashes.append(frame_hash)

    return len(hashes) + unreadable


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
