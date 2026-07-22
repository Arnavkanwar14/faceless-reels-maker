"""Lightweight CPU image upscaling (Lanczos resample + unsharp mask).

Applied only to still images below a usable resolution for the target
video frame - mainly real web-sourced images (real_images.py), which
unlike AI-generated images don't have a controlled output size.

Previously this ran Real-ESRGAN. That was removed: on this machine a
single 640x360 frame needed 57-70s of CPU inference, which meant it blew
through its time budget on *every* image and fell back to the unscaled
original anyway - i.e. it cost ~60s per image and delivered nothing.
Lanczos + a mild unsharp pass takes ~130ms (~450x faster) and, unlike a
neural upscaler, invents no detail that wasn't in the source.

The real fix for softness is upstream, not here: material.py now allows
enough time to fetch genuine 1080p footage, so most frames arrive above
the threshold and skip this path entirely. This is the fallback for when
that download fails and only a low-res source is available.

Fails open: any read/write/resize error returns the original image path
unchanged. Upscaling is a quality bonus, never a requirement for the
pipeline to produce a video.
"""

from __future__ import annotations

from loguru import logger
from PIL import Image, ImageFilter

_MIN_SHORT_SIDE = 900
_TARGET_SHORT_SIDE = 1080
# 放大倍数上限。下游 real_images 的分辨率门槛假设放大不会凭空把一张极小的
# 缩略图变成"合格"素材（Real-ESRGAN 时期天然受限于模型固定的 4 倍），这里
# 保留同样的上限，避免一张 120px 的垃圾图被拉到 1080px 混过尺寸检查。
_MAX_SCALE = 4.0
_UNSHARP_RADIUS = 2
_UNSHARP_PERCENT = 110
_UNSHARP_THRESHOLD = 3


def maybe_upscale_image(image_path: str, min_short_side: int = _MIN_SHORT_SIDE) -> str:
    """图片短边低于阈值时用 Lanczos 放大并轻度锐化，就地覆盖原文件。

    已经够大的图片直接原样返回，不做任何处理。任何失败都返回原始路径，
    不影响整条流水线继续运行。
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            if min(width, height) >= min_short_side:
                return image_path

            scale = min(_TARGET_SHORT_SIDE / min(width, height), _MAX_SCALE)
            new_size = (max(1, round(width * scale)), max(1, round(height * scale)))

            upscaled = img.convert("RGB").resize(new_size, Image.LANCZOS)
            # Lanczos 放大后画面会偏软，一道轻度 unsharp 能把边缘对比拉回来，
            # 代价只有几十毫秒。力度刻意压得比较保守，避免在压缩痕迹明显的
            # 视频截图上把块状噪点也一并锐化出来。
            upscaled = upscaled.filter(
                ImageFilter.UnsharpMask(
                    radius=_UNSHARP_RADIUS,
                    percent=_UNSHARP_PERCENT,
                    threshold=_UNSHARP_THRESHOLD,
                )
            )

        upscaled.save(image_path)
        logger.info(
            f"upscale: {image_path.rsplit(chr(92), 1)[-1]} "
            f"{width}x{height} -> {new_size[0]}x{new_size[1]}"
        )
    except Exception as e:
        logger.warning(f"upscale: failed for {image_path}, keeping original: {e}")

    return image_path
