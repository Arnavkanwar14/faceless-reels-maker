"""Real (non-AI) images tailored to the topic, for subjects where actual
footage/photos exist but a good short video clip doesn't (or a video clip
alone doesn't cover the needed duration).

Two real (non-AI) sources, tried per search term:
1. Screenshots pulled directly from a YouTube video that material.py's
   title-relevance search already verified as being about the subject (see
   material.find_best_youtube_video / extract_screenshot_frames). Preferred
   when available - the frame is guaranteed to be real footage of the
   subject, not a loosely keyword-matched result.
2. DuckDuckGo image search (via `ddgs`) as a supplementary/fallback pool for
   terms with no single clearly-matching video (e.g. broad topics covered by
   many videos rather than one canonical source).

Deliberately NOT using YouTube thumbnails as a source: a thumbnail is
click-bait art (title cards, reaction faces, edited overlays), not
necessarily a real frame of the video's actual content, unlike a screenshot
extracted from inside the video itself.

Every candidate image is quality-gated (real resolution, sane aspect ratio,
not blurry/black/a near-duplicate of one already used) before being handed
to ai_visuals' image_to_ken_burns_clip. The final produced clips also pass
through task.py's visual_gate (Groq vision relevance check), which is the
main defense against a DDG result that technically matched the search
keywords but doesn't actually show the subject.
"""

from __future__ import annotations

import os
from typing import List

import requests
from loguru import logger
from PIL import Image

from app.models.schema import VideoAspect
from app.services import ai_visuals, material, quality_gate, upscale, visual_gate
from app.utils import utils

_MIN_SHORT_SIDE = 640
_MIN_LONG_SIDE = 1000
# 原始素材的分辨率下限，在裁边和放大之前判断。
#
# 放大用的是 Lanczos，它只做插值、不会凭空补出细节，所以"先收一张 360p 的图
# 再放大到 1080p"得到的只是一张放大的糊图。再叠加裁边（四周各 15%）和竖屏
# Ken Burns 的二次裁切，最后呈现的是一块被放大好几倍的马赛克。
#
# 门槛按"裁完还够用"来定：600 的短边裁掉 30% 之后还有 420，放大到竖屏画幅
# 仍然能看。低于这个值的素材直接不要——宁可回退去用网络图片搜索（那边的
# 新闻配图普遍在 1200px 以上），也不要硬塞一张糊图进成片。
_MIN_SOURCE_SHORT_SIDE = 600
_MAX_ASPECT_RATIO = 2.5  # long_side / short_side above this gets rejected
_IMAGES_PER_TERM = 2
_FRAMES_TO_SAMPLE_PER_TERM = _IMAGES_PER_TERM * 3  # extra headroom for quality gate rejects
_DOWNLOAD_TIMEOUT = 20
_MAX_IMAGE_BYTES = 15 * 1024 * 1024

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

# 这些来源的图片按其性质就是"带水印或带他人频道标识"的，不做逐张判断，直接
# 整域排除——比拿视觉模型一张张去认便宜得多，也不受 API 额度影响。
#
# youtube/ytimg：搜索结果里最大的一类其实是视频封面图。封面本身就是为了点击率
#   做的二次创作——频道台标、主播人脸、"LEAKED!" 之类的大字，全部烤进了图里，
#   属于典型的"别的频道的东西"，不是原始画面。
# 图库站：预览图默认打满水印，商用还要授权，两头都不合适。
# pinterest 等聚合站：本身不是原始来源，转载的往往已经是别人打过标的版本。
_BLOCKED_IMAGE_SOURCES = (
    "youtube.com", "ytimg.com", "youtu.be",
    "shutterstock", "alamy", "gettyimages", "istockphoto", "dreamstime",
    "depositphotos", "123rf", "bigstockphoto", "canstockphoto", "agefotostock",
    "imago-images", "profimedia", "zumapress", "newscom", "stock.adobe",
    "pinterest", "pinimg",
)

# 图片标题/页面里出现这些词，基本可以认定是二次创作的封面图或带订阅引导的图，
# 而不是原始画面。
_BLOCKED_TITLE_MARKERS = (
    "subscribe", "reaction", "react ", "watermark", "stock photo",
    "royalty free", "getty images",
)


def _is_blocked_source(*fields: str) -> bool:
    """判断图片来源是否属于"必然带水印/带他人频道标识"的黑名单。"""
    haystack = " ".join(f.lower() for f in fields if f)
    return any(blocked in haystack for blocked in _BLOCKED_IMAGE_SOURCES)


# 台标、频道 logo、电视台角标这类东西几乎总是贴在画面四个角或紧贴边缘的位置——
# 那本来就是"不挡住正片内容"的地方。所以直接把四周裁掉一圈，就能不依赖任何
# 外部服务地干掉绝大部分角标。
#
# 裁多少是个取舍：裁太少去不干净，裁太多会把主体本身切掉。实测 12% 还会在
# 边角留下一条台标残边，15% 能完整裁掉常见角标，同时主体构图仍然完好——顺带
# 还能把搬运视频常见的信箱式黑边一起裁掉。
#
# 注意这一手只对"贴边"的水印有效。压在画面正中的大标题字幕（新闻/博客做的
# 文章头图常见）裁不掉，那种只能靠 visual_gate 看图判断。
_BORDER_CROP_RATIO = 0.15


def crop_borders(image_path: str, ratio: float = _BORDER_CROP_RATIO) -> str:
    """把图片四周各裁掉 ratio 比例的一圈，就地覆盖原文件。

    裁完仍然保持原始宽高比，所以不会影响后续 Ken Burns 的构图逻辑。
    任何失败都返回原路径，不影响流水线继续。
    """
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            width, height = img.size
            dx, dy = int(width * ratio), int(height * ratio)
            if width - 2 * dx < 100 or height - 2 * dy < 100:
                return image_path
            cropped = img.crop((dx, dy, width - dx, height - dy))
        cropped.save(image_path)
    except Exception as e:
        logger.debug(f"real_images: border crop failed for '{image_path}': {e}")
    return image_path


def _passes_dimension_gate(width: int, height: int) -> bool:
    if width <= 0 or height <= 0:
        return False
    short_side, long_side = sorted((width, height))
    if short_side < _MIN_SHORT_SIDE or long_side < _MIN_LONG_SIDE:
        return False
    if long_side / short_side > _MAX_ASPECT_RATIO:
        return False
    return True


def _ddgs_image_candidates(search_term: str, video_subject: str) -> List[str]:
    try:
        from ddgs import DDGS
    except ImportError:
        logger.warning("real_images: ddgs package not installed, skipping web image search")
        return []

    query = f"{search_term} {video_subject}".strip()
    try:
        with DDGS() as ddgs:
            # 多要一些结果——下面按来源和标题过滤掉的比例很高，要不然过完
            # 一轮就没剩几张可用的了。
            results = list(ddgs.images(query, max_results=25, size="Large"))
    except Exception as e:
        logger.debug(f"real_images: ddgs search failed for '{query}': {e}")
        return []

    subject_keywords = material._extract_subject_keywords(video_subject)

    urls = []
    blocked_source = 0
    blocked_title = 0
    off_topic = 0
    for r in results:
        image_url = r.get("image")
        if not image_url:
            continue

        source = str(r.get("source") or "")
        page_url = str(r.get("url") or "")
        title = str(r.get("title") or "")

        if _is_blocked_source(source, page_url, image_url):
            blocked_source += 1
            continue

        lowered_title = title.lower()
        if any(marker in lowered_title for marker in _BLOCKED_TITLE_MARKERS):
            blocked_title += 1
            continue

        # 标题必须真的提到主体本身。DDG 是模糊匹配，只对上搜索词里某个
        # 泛用词（"gameplay"、"new"）就返回的结果，画面基本和主题无关。
        if subject_keywords and not material._text_matches_subject(
            f"{title} {page_url}", subject_keywords
        ):
            off_topic += 1
            continue

        try:
            width, height = int(r.get("width") or 0), int(r.get("height") or 0)
        except (TypeError, ValueError):
            width, height = 0, 0
        if width and height and not _passes_dimension_gate(width, height):
            continue
        urls.append(image_url)

    if blocked_source or blocked_title or off_topic:
        logger.info(
            f"real_images: ddg '{query}' -> {len(urls)} usable "
            f"(dropped {blocked_source} watermarked/branded sources, "
            f"{blocked_title} by title, {off_topic} off-topic)"
        )
    return urls


def _download_ddg_image(url: str, output_path: str) -> bool:
    try:
        response = requests.get(url, headers=_HEADERS, timeout=_DOWNLOAD_TIMEOUT, stream=True)
        response.raise_for_status()

        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > _MAX_IMAGE_BYTES:
            return False

        data = response.content
        if len(data) == 0 or len(data) > _MAX_IMAGE_BYTES:
            return False

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(data)

        with Image.open(output_path) as img:
            img.verify()
        with Image.open(output_path) as img:
            if img.format not in ("JPEG", "PNG", "WEBP"):
                os.remove(output_path)
                return False

        return True
    except Exception as e:
        logger.debug(f"real_images: failed to download/validate '{url}': {e}")
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass
        return False


def _has_usable_source_resolution(image_path: str) -> bool:
    """判断原图是否够格进入后面的裁边 + 放大 + 竖屏裁切。

    同时看两件事：像素尺寸够不够，以及画面里是不是真的有细节。只看尺寸会
    被"低清图拉大再存一次"的素材骗过去——文件写着 1600x900，实际是一张
    480p 的糊图，裁切放大之后观众根本认不出画面里是什么。
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size
    except Exception:
        return False

    if min(width, height) < _MIN_SOURCE_SHORT_SIDE:
        return False

    if not quality_gate.is_sharp_enough(image_path):
        logger.debug(
            f"real_images: '{os.path.basename(image_path)}' is {width}x{height} but "
            "lacks real detail (upscaled/soft source), skipping"
        )
        return False

    return True


def _collect_video_screenshot_candidates(
    term: str, video_subject: str, output_dir: str
) -> tuple[List[str], str]:
    """从与 term/video_subject 最相关的那条 YouTube 视频里截取候选截图。
    返回 (通过质量门的图片路径列表, 来源视频标题) - 标题仅用于日志。"""
    video = material.find_best_youtube_video(term, video_subject, minimum_duration=10)
    if not video:
        return [], ""

    frame_paths = material.extract_screenshot_frames(
        video_id=video["video_id"],
        video_url=video["url"],
        duration=video["duration"],
        output_dir=output_dir,
        max_frames=_FRAMES_TO_SAMPLE_PER_TERM,
    )

    # 1080p 直链拉取失败时会回退到渐进式流，那条路有时候只有 360p 甚至更低。
    # 这种帧放大之后就是一团糊，宁可整批丢掉让流程回退到网络图片搜索。
    usable = [p for p in frame_paths if _has_usable_source_resolution(p)]
    if frame_paths and not usable:
        logger.info(
            f"real_images: screenshots for '{term}' came back below "
            f"{_MIN_SOURCE_SHORT_SIDE}px (1080p fetch likely fell back to a "
            "low-res stream) - skipping them in favour of web image search"
        )

    return quality_gate.filter_low_quality_images(usable), video["title"]


def _collect_ddg_candidates(term: str, video_subject: str, output_dir: str) -> List[str]:
    """DDG 图片搜索作为补充来源：某些宽泛主题不存在单条能覆盖全部素材的
    权威视频，这时候需要更多样的图片池。下载并本地校验后同样过质量门。"""
    image_paths = []
    for url in _ddgs_image_candidates(term, video_subject):
        url_hash = utils.md5(url)
        image_path = os.path.join(output_dir, f"ddgimg-{url_hash}.jpg")
        if os.path.exists(image_path) and os.path.getsize(image_path) > 0:
            image_paths.append(image_path)
            continue
        if _download_ddg_image(url, image_path) and _has_usable_source_resolution(
            image_path
        ):
            image_paths.append(image_path)

    image_paths = quality_gate.filter_low_quality_images(image_paths)
    # 域名黑名单挡不住的那一类：新闻/博客站自己做的文章头图，同样会带站点
    # logo 和大标题——形式上和视频封面图没有区别。这类只能看图本身才认得
    # 出来，所以在这里过一遍视觉检查，在还只是一张图的时候就把它丢掉。
    # strict=True：网络搜到的图是最容易带台标/大标题的来源，一旦没法核实
    # 就直接跳过。真正相关的画面还可以从正片抽帧那条路补上，不值得为了凑
    # 数把无法核实的图放进成片。
    return visual_gate.filter_clean_images(image_paths, video_subject, strict=True)


def download_real_image_clips(
    task_id: str,
    search_terms: List[str],
    video_subject: str,
    video_aspect: VideoAspect,
    audio_duration: float = 0.0,
    max_clip_duration: int = 5,
    material_directory: str = "",
) -> List[str]:
    """为每个搜索词找真实图片素材（优先：主体相关视频截图；补充：DDG 图片
    搜索），转成 Ken Burns 短片段，直到覆盖音频时长或用完关键词。"""
    width, height = video_aspect.to_resolution()
    output_dir = material_directory or utils.storage_dir("cache_videos")
    os.makedirs(output_dir, exist_ok=True)

    video_paths = []
    seen_hashes = set()
    total_duration = 0.0

    for term in search_terms:
        if total_duration >= audio_duration and video_paths:
            logger.info(
                f"real_images: total duration {total_duration:.1f}s covers audio, "
                "skip remaining terms"
            )
            break

        screenshot_candidates, source_title = _collect_video_screenshot_candidates(
            term, video_subject, output_dir
        )
        candidates = [(path, source_title) for path in screenshot_candidates]

        if len(candidates) < _IMAGES_PER_TERM:
            ddg_candidates = _collect_ddg_candidates(term, video_subject, output_dir)
            candidates += [(path, "DuckDuckGo image search") for path in ddg_candidates]

        accepted_for_term = 0
        for image_path, source in candidates:
            if accepted_for_term >= _IMAGES_PER_TERM:
                break

            frame_hash = utils.md5(image_path)
            if frame_hash in seen_hashes:
                continue

            clip_path = os.path.join(output_dir, f"realimg-{task_id}-{frame_hash}.mp4")
            if os.path.exists(clip_path) and os.path.getsize(clip_path) > 0:
                video_paths.append(clip_path)
                seen_hashes.add(frame_hash)
                accepted_for_term += 1
                total_duration += max_clip_duration
                continue

            # 先裁边再放大：台标/角标/信箱黑边都贴着画面边缘，趁放大之前裁掉，
            # 既不浪费算力去放大注定要丢的像素，也避免把水印一起放大。
            crop_borders(image_path)

            # 真实图片（视频截图或网络搜索结果）分辨率不像 AI 生成图那样可控，
            # 放大到目标画幅前先补一次分辨率；图片本身已经够大时函数会直接
            # 跳过。分辨率门槛放在放大之后检查，而不是放大之前——否则一张
            # 内容完全相关但来源画质偏低的图会被直接拒绝，明明放大后完全够用。
            upscaled_path = upscale.maybe_upscale_image(image_path)
            try:
                with Image.open(upscaled_path) as upscaled_img:
                    upscaled_width, upscaled_height = upscaled_img.size
            except Exception as e:
                logger.debug(f"real_images: failed to read '{upscaled_path}': {e}")
                continue
            if not _passes_dimension_gate(upscaled_width, upscaled_height):
                continue

            if ai_visuals.image_to_ken_burns_clip(
                upscaled_path, clip_path, max_clip_duration, width, height
            ):
                video_paths.append(clip_path)
                seen_hashes.add(frame_hash)
                accepted_for_term += 1
                total_duration += max_clip_duration
                logger.info(f"real_images: got clip for '{term}' from {source}")

    logger.success(f"real_images: generated {len(video_paths)} clips from real images")
    return video_paths
