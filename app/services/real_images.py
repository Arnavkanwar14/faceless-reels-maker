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

import math
import os
import subprocess
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
# 候选池要比实际需要的张数多几倍，择优才有意义：过完清晰度、水印、拼图
# 几道闸之后能留下的通常只是少数，池子太浅就退化回"有什么用什么"。
_POOL_OVERSAMPLE = 4
# 素材不足需要拉长单张时长时的上限。设成 15s 是为了让常见的"薄素材"情况
# 正好一遍铺满、完全不循环——比如 3 张图铺 45 秒，每张 15 秒刚好覆盖，不会
# 出现"同样三张每 15 秒重来一次"。再长会让单张静态图停留过久显得拖沓，所以
# 到此为止；真的只有一两张时，剩下的时间仍需少量循环，但每次停留够长，
# 观感远好于短片段快速循环。
_MAX_STRETCHED_CLIP_DURATION = 15
# 覆盖率底线：最终不同镜头（按基础片段时长算）至少要覆盖这么多比例的旁白，
# 否则从 COLLAGE、再到 GENERIC 逐级往回捞，直到够线或候选用尽。
_COVERAGE_FLOOR = 0.7
# 感知哈希汉明距离阈值，超过这个距离才算不同的画面。海报类素材被各站转载后
# 往往只有轻微的压缩/裁切差异，阈值太小会漏判成"不同的图"。
_DUPLICATE_HAMMING_THRESHOLD = 6
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
# 无论主体在哪，每条边至少裁掉这么多——保证常见的角标/信箱黑边始终被削掉，
# 即便裁切窗口为了保住主体而贴到了某条边。
_BORDER_CROP_MIN_MARGIN = 0.04


def crop_borders(image_path: str, ratio: float = _BORDER_CROP_RATIO) -> str:
    """裁掉四周一圈以去除角标/黑边，但裁切窗口对准主体而不是死板地取正中。

    原来是四周各裁固定 15%，正中不动——竖图里人物的头通常就在顶部那 15% 里，
    于是"去水印"顺手把头也裁没了。现在先用边缘能量估出主体焦点，再把（尺寸
    不变的）裁切窗口滑动到对准焦点，同时保证每条边至少还裁掉
    _BORDER_CROP_MIN_MARGIN，这样既保住主体，又不至于漏掉贴边的角标。

    任何失败都返回原路径，不影响流水线继续。
    """
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            width, height = img.size

            win_w = int(width * (1 - 2 * ratio))
            win_h = int(height * (1 - 2 * ratio))
            if win_w < 100 or win_h < 100:
                return image_path

            fx, fy = quality_gate.salient_focus(img)

            # 理想位置：让焦点落在裁切窗口正中。
            left = int(fx * width - win_w / 2)
            top = int(fy * height - win_h / 2)

            # 但每条边至少保留 min_margin 的裁切量，所以窗口不能一路贴到图像边缘。
            min_mx = int(width * _BORDER_CROP_MIN_MARGIN)
            min_my = int(height * _BORDER_CROP_MIN_MARGIN)
            left = min(max(left, min_mx), width - win_w - min_mx)
            top = min(max(top, min_my), height - win_h - min_my)

            cropped = img.crop((left, top, left + win_w, top + win_h))
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

    # 同一个主题用几种问法各搜一轮。
    #
    # 原因有两个：一是过滤掉的比例很高（域名黑名单、清晰度、海报/拼图判定
    # 层层筛完，一轮搜索经常只剩下个位数候选，池子太浅就退化成"有什么用
    # 什么"）；二是不加限定词的搜索，返回的大多是海报和宣传图——加上
    # "screenshot"、"gameplay" 这类词，才更容易搜到真实画面。
    base_query = f"{search_term} {video_subject}".strip()
    queries = [
        base_query,
        f"{search_term} screenshot",
        f"{video_subject} gameplay screenshot"
        if "game" in video_subject.lower()
        else f"{video_subject} scene still",
        # 多几个角度是为了保证有足够多"看得出区别"的候选进池子——层层过滤
        # （海报/拼图/水印/清晰度）会筛掉一大半，问法太少就容易只剩三五张、
        # 最后在成片里循环。这些角度偏向真实画面而不是宣传图。
        f"{video_subject} still frame",
        f"{video_subject} close up",
        f"{video_subject} action scene",
    ]

    results = []
    seen_urls = set()
    for query in queries:
        try:
            with DDGS() as ddgs:
                for r in ddgs.images(query, max_results=20, size="Large"):
                    url = r.get("image")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        results.append(r)
        except Exception as e:
            logger.debug(f"real_images: ddgs search failed for '{query}': {e}")

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
            f"real_images: ddg '{base_query}' ({len(queries)} query variants) -> "
            f"{len(urls)} usable (dropped {blocked_source} watermarked/branded "
            f"sources, {blocked_title} by title, {off_topic} off-topic)"
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


def _clip_short_side(clip_path: str) -> int:
    """读出视频片段的短边像素数，用来判断这段素材值不值得用。"""
    try:
        result = subprocess.run(
            [
                utils.get_ffmpeg_binary().replace("ffmpeg", "ffprobe"),
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=p=0",
                clip_path,
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        width, height = (int(v) for v in result.stdout.strip().split(",")[:2])
        return min(width, height)
    except Exception:
        # ffprobe 可能不在 imageio-ffmpeg 的分发包里，退回用 moviepy 读。
        try:
            from moviepy.video.io.VideoFileClip import VideoFileClip

            with VideoFileClip(clip_path) as clip:
                return min(clip.size)
        except Exception as e:
            logger.debug(f"real_images: could not read size of {clip_path}: {e}")
            return 0


def _collect_motion_clips(
    search_terms: List[str],
    video_subject: str,
    output_dir: str,
    needed: int,
    clip_duration: int,
) -> List[str]:
    """从主体最相关的那条视频里切出真实动态片段。

    分辨率不达标时整批放弃——低清动态片段和低清截图一样不能用，而且这种
    情况下回退到网页图片搜索通常还能捞到能用的图。
    """
    # 先用"找原始素材"的问法去搜，再退回原始关键词。
    #
    # 关键词是按脚本内容生成的（"所有英雄确认出演"这种），而官方发布的视频
    # 标题里根本不会这么写——标题相关性一过滤，剩下的全是解说号和搬运号，
    # 拿到的画面自然又糊又带台标。实测同一个主题：用 "…all heroes confirmed"
    # 搜到的是粉丝剪辑（360p，被判定不可用），换成 "…trailer" 直接搜到
    # Marvel Entertainment 官方频道的 1080p 预告片。
    footage_terms = [
        f"{video_subject} official trailer",
        f"{video_subject} trailer",
    ] + list(search_terms)

    clips: List[str] = []
    tried_videos = set()
    for term in footage_terms:
        if len(clips) >= needed:
            break

        video = material.find_best_youtube_video(term, video_subject, minimum_duration=10)
        if video and video["video_id"] in tried_videos:
            continue
        if video:
            tried_videos.add(video["video_id"])
        if not video:
            continue

        fetched = material.fetch_topic_segment(
            video["video_id"], video["url"], video["duration"], output_dir
        )
        if not fetched:
            continue
        local_path, seg_duration = fetched

        short_side = _clip_short_side(local_path)
        if short_side and short_side < _MIN_SOURCE_SHORT_SIDE:
            logger.info(
                f"real_images: '{video['title'][:50]}' segment came back at "
                f"{short_side}px short side - too low to use as motion footage"
            )
            continue

        fresh = material.extract_motion_clips(
            local_path,
            seg_duration,
            output_dir,
            video["video_id"],
            count=needed - len(clips),
            clip_duration=clip_duration,
        )
        if not fresh:
            continue

        # 每拿到一批就立刻和已接受的素材一起去重，而不是等全部收完再统一
        # 处理。差别在于名额怎么算：收完再去重的话，两段其实一样的素材在
        # 循环里各占一个名额，凑够数就不再往下找了，去重之后才发现只剩一段
        # ——这时候已经没有机会再去别的视频里补了。边收边去重，重复的当场
        # 不占名额，循环会自动继续找下一条视频，直到真的凑够不同的画面。
        before = len(clips)
        clips = quality_gate.dedupe_clips(clips + fresh)
        rejected = len(fresh) - (len(clips) - before)
        if rejected > 0:
            logger.info(
                f"real_images: {rejected} clip(s) from '{video['title'][:40]}' "
                "duplicate a shot we already have - looking at another video "
                "instead of repeating it"
            )

    if clips:
        # 动态片段同样要过一遍水印/相关性判定：搬运号的画面一样会压台标。
        clips = visual_gate.filter_relevant_clips(clips, video_subject)

    distinct = quality_gate.count_distinct_shots(clips) if clips else 0
    if clips and distinct < len(clips):
        logger.info(
            f"real_images: {len(clips)} motion clip(s) but only {distinct} "
            "visually distinct shot(s)"
        )
    return clips


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

    # 视觉检查放到调用方统一批量做——两个来源的候选先汇成一个池子，再一次性
    # 判定，配额按批次消耗而不是按张。
    return quality_gate.filter_low_quality_images(image_paths)


def download_real_image_clips(
    task_id: str,
    search_terms: List[str],
    video_subject: str,
    video_aspect: VideoAspect,
    audio_duration: float = 0.0,
    max_clip_duration: int = 5,
    material_directory: str = "",
) -> List[str]:
    """先把两个来源的候选图汇成一个池子，统一打分排序，再挑最好的几张转成
    Ken Burns 片段。

    之前是"逐个关键词、按搜索引擎给的顺序、取前两张能过闸的"——搜索结果
    的排序是按引擎的相关性算的，和画面好不好看没有关系，于是排在第 8 位的
    好图永远没机会出场，因为名额早被第 4、第 6 位填满了。改成先攒池子再
    择优，才能真正"挑出最合适的"，而不是"碰到的头两张"。
    """
    width, height = video_aspect.to_resolution()
    output_dir = material_directory or utils.storage_dir("cache_videos")
    os.makedirs(output_dir, exist_ok=True)

    needed = max(1, int(audio_duration / max_clip_duration + 0.999)) if audio_duration else 1

    # ---- 0. 真实动态片段优先 ----
    #
    # 静态图配 Ken Burns 推镜说到底是在模拟运动，真实素材本身的运动永远更好
    # 看。而且对还没上映的电影/还没发售的游戏这类主题，网上能搜到的图基本
    # 全是海报和同人图——真正的画面只存在于预告片里，网页图片搜索这条路
    # 天然拿不到。下载片段本来就是抽帧要做的事，顺手切成动态片段几乎零成本。
    motion_clips = _collect_motion_clips(
        search_terms, video_subject, output_dir, needed, max_clip_duration
    )
    if len(motion_clips) >= needed:
        logger.success(
            f"real_images: using {len(motion_clips)} real motion clips "
            "(better than stills, no Ken Burns needed)"
        )
        return motion_clips[:needed]

    still_slots_needed = needed - len(motion_clips)
    if motion_clips:
        logger.info(
            f"real_images: got {len(motion_clips)} motion clip(s), filling the "
            f"remaining {still_slots_needed} slot(s) with stills"
        )

    # ---- 1. 汇集候选：两个来源都收，不再是"截图不够才去搜图" ----
    pool: List[tuple[str, str]] = []
    seen_files = set()
    for term in search_terms:
        for path in _collect_video_screenshot_candidates(term, video_subject, output_dir)[0]:
            if path not in seen_files:
                seen_files.add(path)
                pool.append((path, f"video screenshot ({term})"))
        for path in _collect_ddg_candidates(term, video_subject, output_dir):
            if path not in seen_files:
                seen_files.add(path)
                pool.append((path, f"web image search ({term})"))
        # 池子够深就不用把每个关键词都跑一遍——再往下收益递减，只是徒增
        # 下载和判定的时间。
        if len(pool) >= needed * _POOL_OVERSAMPLE:
            break

    if not pool:
        logger.warning("real_images: no usable candidates found from any source")
        return []

    # ---- 2. 本地打分（免费）：清晰度排序 + 跨来源感知哈希去重 ----
    scored = []
    for path, source in pool:
        sharpness = quality_gate.normalized_sharpness(path) or 0.0
        scored.append((sharpness, path, source))
    scored.sort(key=lambda item: item[0], reverse=True)

    # 同一个主题在不同来源、不同关键词下经常搜到同一张（或几乎一样的）图，
    # 尤其是官方海报这种被各家站点反复转载的素材。不跨来源去重的话，最后
    # 挑出来的几张可能是同一张图的不同副本，成片看着就是同一个画面反复出现。
    # 按清晰度从高到低遍历，保留每组近似图里最清晰的那张。
    deduped = []
    kept_hashes = []
    for sharpness, path, source in scored:
        image = quality_gate._load_image(path)
        if image is None:
            deduped.append((sharpness, path, source))
            continue
        try:
            image_hash = quality_gate._average_hash(image)
        except Exception:
            deduped.append((sharpness, path, source))
            continue
        if any(
            quality_gate._hamming_distance(image_hash, kept) <= _DUPLICATE_HAMMING_THRESHOLD
            for kept in kept_hashes
        ):
            logger.debug(
                f"real_images: skipping near-duplicate {os.path.basename(path)}"
            )
            continue
        kept_hashes.append(image_hash)
        deduped.append((sharpness, path, source))

    if len(deduped) < len(scored):
        logger.info(
            f"real_images: dropped {len(scored) - len(deduped)} near-duplicate "
            "candidate(s) across sources"
        )
    scored = deduped

    # ---- 3. 视觉判定：整池一次性批量判，配额按批次算 ----
    ranked_paths = [path for _, path, _ in scored]
    verdicts = visual_gate.classify_images_batch(ranked_paths, video_subject)
    source_by_path = {path: source for _, path, source in scored}

    # ---- 覆盖率兜底：分级放行，先干净后切题，绝不越过水印线 ----
    #
    # ranked_paths 已经按清晰度从高到低排过序，所以各档按此顺序取就是"每档
    # 里最清晰的优先"。
    #
    # 目标：至少凑够能覆盖 70% 旁白的不同镜头数（按基础片段时长算）。达不到
    # 就分级往回捞：
    #   1) OK      —— 干净、切题、能认出主体。能填多少填多少，直到 needed。
    #   2) COLLAGE —— 多格漫画页：切题、是真内容，只是构图杂。用来把镜头数
    #                 补到 70% 底线（漫画类主题网上大多是这种，总比循环强）。
    #   3) GENERIC —— 切题但画面空洞的兜底。只有前两档还凑不够底线时才用。
    # POSTER（海报/宣传图）、UNRELATED（跑题）、WATERMARK（带标）、UNKNOWN
    # （没核实成）一律不进——保持"稍微严格"，也守住绝不用带水印素材这条线。
    def _tier(*wanted):
        return [p for p in ranked_paths if verdicts.get(p) in wanted]

    target_distinct = max(1, math.ceil(_COVERAGE_FLOOR * needed))

    usable = _tier(visual_gate.VERDICT_OK)[:needed]

    if len(usable) < target_distinct:
        for p in _tier(visual_gate.VERDICT_COLLAGE):
            if len(usable) >= target_distinct:
                break
            if p not in usable:
                usable.append(p)
        collage_added = len(usable) - len(_tier(visual_gate.VERDICT_OK)[:needed])
        if collage_added > 0:
            logger.info(
                f"real_images: only "
                f"{len(_tier(visual_gate.VERDICT_OK))} clean clip(s) - added "
                f"{collage_added} on-topic collage/comic-page shot(s) to reach "
                f"the {int(_COVERAGE_FLOOR * 100)}% coverage floor"
            )

    if len(usable) < target_distinct:
        before = len(usable)
        for p in _tier(visual_gate.VERDICT_GENERIC):
            if len(usable) >= target_distinct:
                break
            if p not in usable:
                usable.append(p)
        if len(usable) > before:
            logger.info(
                f"real_images: still short - added {len(usable) - before} "
                "generic on-topic shot(s) as last resort for coverage"
            )

    coverage = (len(usable) * max_clip_duration / audio_duration) if audio_duration else 1.0
    logger.info(
        f"real_images: pooled {len(pool)} candidates, {len(usable)} usable "
        f"(~{coverage:.0%} coverage, floor {int(_COVERAGE_FLOOR * 100)}%, "
        f"need {needed})"
    )

    # ---- 4. 择优渲染 ----
    #
    # 图片没凑够铺满时间线的张数时，把每张的时长拉长，让它们一遍就铺满整段
    # 旁白，而不是回落到"同样几张短片段来回循环"。三张图 45 秒，与其切成
    # 3×5s 再循环三遍（观众看到的就是每 15 秒同样三张重来一次），不如每张
    # 15 秒放一遍——Ken Burns 现在按时长匀速推进，长片段也一路在动。时长
    # 设了上限，避免真的只有一两张时把单张停留拉到夸张的地步。
    per_clip_duration = max_clip_duration
    if usable and audio_duration > 0 and len(usable) < needed:
        spread = -(-int(audio_duration) // len(usable))  # ceil
        per_clip_duration = min(_MAX_STRETCHED_CLIP_DURATION, max(max_clip_duration, spread))
        if per_clip_duration > max_clip_duration:
            logger.info(
                f"real_images: only {len(usable)} distinct image(s) for a "
                f"{audio_duration:.0f}s video - stretching each to "
                f"{per_clip_duration}s so they fill the timeline once instead "
                "of looping"
            )

    video_paths = []
    for image_path in usable:
        if len(video_paths) >= needed:
            break

        frame_hash = utils.md5(image_path)
        clip_path = os.path.join(
            output_dir, f"realimg-{task_id}-{frame_hash}-{per_clip_duration}.mp4"
        )
        if os.path.exists(clip_path) and os.path.getsize(clip_path) > 0:
            video_paths.append(clip_path)
            continue

        # 先裁边再放大：台标/角标/信箱黑边都贴着画面边缘，趁放大之前裁掉，
        # 既不浪费算力去放大注定要丢的像素，也避免把水印一起放大。
        crop_borders(image_path)

        # 真实图片（视频截图或网络搜索结果）分辨率不像 AI 生成图那样可控，
        # 放大到目标画幅前先补一次分辨率；图片本身已经够大时函数会直接跳过。
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
            upscaled_path, clip_path, per_clip_duration, width, height
        ):
            video_paths.append(clip_path)
            logger.info(
                f"real_images: selected {os.path.basename(image_path)} from "
                f"{source_by_path.get(image_path, 'unknown source')}"
            )

    logger.success(f"real_images: generated {len(video_paths)} clips from real images")
    return video_paths
