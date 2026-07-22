import os
import random
import re
import subprocess
import threading
from typing import List
from urllib.parse import urlencode

import requests
from loguru import logger
from moviepy.video.io.VideoFileClip import VideoFileClip

from app.config import config
from app.models.schema import MaterialInfo, VideoAspect, VideoConcatMode
from app.utils import utils

# Thread-safe counter for API key rotation
_api_key_counter = 0
_api_key_lock = threading.Lock()

# -----------------------------------------------------------------------------
# Relevance guard / 素材相关性兜底
#
# Stock-footage search engines (Pexels/Pixabay) do loose, per-word fuzzy
# matching, not literal AND matching. A query like "octopus intelligence" can
# return robot/chess videos because "intelligence" alone matched, even though
# "octopus" was in the query. Putting the subject word in the search query is
# not enough - the *results* must be checked too. Pexels doesn't expose tags
# for videos, but its page URL (e.g. ".../octopus-gliding-over-ocean-floor-123/")
# is an auto-generated descriptive slug we can check. Pixabay exposes real tags.
# -----------------------------------------------------------------------------

_SUBJECT_STOPWORDS = frozenset(
    {
        "a", "an", "the", "of", "in", "on", "at", "to", "for", "and", "or", "with",
        "about", "facts", "fact", "surprising", "amazing", "top", "best", "things",
        "thing", "how", "why", "what", "is", "are", "video", "short", "reel",
    }
)


def _extract_subject_keywords(video_subject: str) -> List[str]:
    # Prefer proper-noun words (capitalized in the original subject) when present.
    # These are usually the actual named person/entity/brand the topic is about,
    # and are a far more reliable relevance anchor than generic topic words like
    # "controversy" or "reaction" - a clip of two random people arguing will
    # match "controversy" just fine without having anything to do with the
    # actual subject. Falling back to those generic words as the *only* anchor
    # lets clearly wrong footage pass the filter.
    proper_nouns = re.findall(r"\b[A-Z][a-zA-Z]+\b", video_subject or "")
    proper_nouns = [
        w.lower() for w in proper_nouns
        if w.lower() not in _SUBJECT_STOPWORDS and len(w) > 2
    ]
    if proper_nouns:
        return proper_nouns

    words = re.findall(r"[A-Za-z]+", (video_subject or "").lower())
    keywords = [
        w for w in words if w not in _SUBJECT_STOPWORDS and not w.isdigit() and len(w) > 2
    ]
    return keywords or words


def _singular_forms(word: str) -> List[str]:
    forms = [word]
    if word.endswith("es") and len(word) > 4:
        forms.append(word[:-2])
    if word.endswith("s") and len(word) > 3:
        forms.append(word[:-1])
    return forms


def _text_matches_subject(text: str, subject_keywords: List[str]) -> bool:
    text_lower = (text or "").lower()
    return any(
        form in text_lower
        for keyword in subject_keywords
        for form in _singular_forms(keyword)
    )


def _looks_like_named_person_subject(video_subject: str) -> bool:
    """检测主题是否围绕一个具体真实姓名展开——见 llm.py 中的同名函数。

    这两份实现故意保持独立（不做跨模块依赖），但逻辑必须一致：命中时，
    通用素材库不可能有这个人的真实画面，要求结果元数据里出现他的名字只会
    把所有候选素材过滤成 0 个，所以直接跳过这层过滤，交给更精确的搜索词
    本身（含具体名词要求）来保证相关性。
    """
    return bool(re.search(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+\b", video_subject or ""))


def _filter_items_by_subject(
    items: List[MaterialInfo],
    metadata_texts: List[str],
    video_subject: str,
    provider_name: str,
    is_named_person: bool | None = None,
    subject_noun: str | None = None,
) -> List[MaterialInfo]:
    """
    只保留素材元数据（Pexels 页面 URL / Pixabay tags 等）中真正出现主体关键词的
    结果，丢弃仅靠搜索引擎模糊匹配到的不相关素材。如果没有可判断的主体关键词
    （比如极短或纯符号的主题），则不过滤，避免误伤正常流程。

    is_named_person / subject_noun 由调用方传入 llm.classify_subject() 的结果时
    直接使用（更准确，避免正则对 "Deep Ocean Creatures" 这类普通 Title Case 话题
    误判）；不传时才退回本地正则/关键词启发式，保持向后兼容。

    subject_noun 存在时只用它做匹配依据，不再用 _extract_subject_keywords 拆出
    的全部实义词——那份列表里任何一个词单独命中都会放行，例如
    "the science of how volcanoes erupt" 下，一个只提到 "science" 完全和火山
    无关的素材也会被当作合格结果。
    """
    named_person = (
        is_named_person
        if is_named_person is not None
        else _looks_like_named_person_subject(video_subject)
    )
    if named_person:
        return items

    subject_keywords = [subject_noun] if subject_noun else _extract_subject_keywords(
        video_subject
    )
    if not subject_keywords:
        return items

    # 如果该 provider 对这批结果完全没有提供可用的元数据文本（比如 Coverr 响应
    # 里没有 title 字段），就不过滤——宁可保留可能不完全相关的结果，也不要因为
    # 元数据缺失而把所有素材都判定为不相关。
    if not any((text or "").strip() for text in metadata_texts):
        logger.warning(
            f"{provider_name}: no relevance metadata available in results, "
            "skipping subject filter for this batch"
        )
        return items

    kept = []
    dropped = 0
    for item, text in zip(items, metadata_texts):
        if _text_matches_subject(text, subject_keywords):
            kept.append(item)
        else:
            dropped += 1

    if dropped:
        logger.info(
            f"{provider_name}: dropped {dropped}/{len(items)} results whose metadata "
            f"did not mention the video subject ({subject_keywords})"
        )
    return kept


def _get_tls_verify() -> bool:
    # 默认开启 TLS 证书校验，防止素材搜索和下载过程被中间人篡改。
    # 仅在企业代理、自签证书等明确需要的场景下，允许用户通过
    # `config.toml` 显式设置 `tls_verify = false` 临时关闭。
    tls_verify = config.app.get("tls_verify", True)
    if isinstance(tls_verify, str):
        tls_verify = tls_verify.strip().lower() not in ("0", "false", "no", "off")

    if not tls_verify:
        logger.warning(
            "TLS certificate verification is disabled by config.app.tls_verify=false. "
            "Only use this in trusted proxy environments."
        )

    return bool(tls_verify)


def get_api_key(cfg_key: str):
    api_keys = config.app.get(cfg_key)
    if not api_keys:
        raise ValueError(
            f"\n\n##### {cfg_key} is not set #####\n\nPlease set it in the config.toml file: {config.config_file}\n\n"
            f"{utils.to_json(config.app)}"
        )

    # if only one key is provided, return it
    if isinstance(api_keys, str):
        return api_keys

    global _api_key_counter
    with _api_key_lock:
        _api_key_counter += 1
        return api_keys[_api_key_counter % len(api_keys)]


def search_videos_pexels(
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect = VideoAspect.portrait,
    video_subject: str = "",
    is_named_person: bool | None = None,
    subject_noun: str | None = None,
) -> List[MaterialInfo]:
    aspect = VideoAspect(video_aspect)
    video_orientation = aspect.name
    video_width, video_height = aspect.to_resolution()
    api_key = get_api_key("pexels_api_keys")
    headers = {
        "Authorization": api_key,
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    }
    # Build URL
    params = {"query": search_term, "per_page": 20, "orientation": video_orientation}
    query_url = f"https://api.pexels.com/videos/search?{urlencode(params)}"
    logger.info(f"searching videos: {query_url}, with proxies: {config.proxy}")

    try:
        r = requests.get(
            query_url,
            headers=headers,
            proxies=config.proxy,
            verify=_get_tls_verify(),
            timeout=(30, 60),
        )
        response = r.json()
        video_items = []
        item_page_urls: List[str] = []
        if "videos" not in response:
            logger.error(f"search videos failed: {response}")
            return video_items
        videos = response["videos"]
        # loop through each video in the result
        for v in videos:
            duration = v["duration"]
            # check if video has desired minimum duration
            if duration < minimum_duration:
                continue
            video_files = v["video_files"]
            # loop through each url to determine the best quality
            for video in video_files:
                w = int(video["width"])
                h = int(video["height"])
                if w == video_width and h == video_height:
                    item = MaterialInfo()
                    item.provider = "pexels"
                    item.url = video["link"]
                    item.duration = duration
                    video_items.append(item)
                    # Pexels page URLs are auto-generated descriptive slugs, e.g.
                    # ".../octopus-gliding-over-ocean-floor-underwater-32199586/".
                    # This is the only relevance signal Pexels' video API exposes.
                    item_page_urls.append(v.get("url", ""))
                    break
        return _filter_items_by_subject(
            video_items, item_page_urls, video_subject, "pexels", is_named_person, subject_noun
        )
    except Exception as e:
        logger.error(f"search videos failed: {str(e)}")

    return []


def search_videos_pixabay(
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect = VideoAspect.portrait,
    video_subject: str = "",
    is_named_person: bool | None = None,
    subject_noun: str | None = None,
) -> List[MaterialInfo]:
    aspect = VideoAspect(video_aspect)

    video_width, video_height = aspect.to_resolution()

    api_key = get_api_key("pixabay_api_keys")
    # Build URL
    params = {
        "q": search_term,
        "video_type": "all",  # Accepted values: "all", "film", "animation"
        "per_page": 50,
        "key": api_key,
    }
    query_url = f"https://pixabay.com/api/videos/?{urlencode(params)}"
    logger.info(f"searching videos: {query_url}, with proxies: {config.proxy}")

    try:
        r = requests.get(
            query_url, proxies=config.proxy, verify=_get_tls_verify(), timeout=(30, 60)
        )
        response = r.json()
        video_items = []
        item_tags: List[str] = []
        if "hits" not in response:
            logger.error(f"search videos failed: {response}")
            return video_items
        videos = response["hits"]
        # loop through each video in the result
        for v in videos:
            duration = v["duration"]
            # check if video has desired minimum duration
            if duration < minimum_duration:
                continue
            video_files = v["videos"]
            # loop through each url to determine the best quality
            for video_type in video_files:
                video = video_files[video_type]
                w = int(video["width"])
                # h = int(video["height"])
                if w >= video_width:
                    item = MaterialInfo()
                    item.provider = "pixabay"
                    item.url = video["url"]
                    item.duration = duration
                    video_items.append(item)
                    # Pixabay exposes real contributor tags, e.g. "ocean, octopus, reef".
                    item_tags.append(v.get("tags", ""))
                    break
        return _filter_items_by_subject(
            video_items, item_tags, video_subject, "pixabay", is_named_person, subject_noun
        )
    except Exception as e:
        logger.error(f"search videos failed: {str(e)}")

    return []


def search_videos_coverr(
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect = VideoAspect.portrait,
    video_subject: str = "",
    is_named_person: bool | None = None,
    subject_noun: str | None = None,
) -> List[MaterialInfo]:
    """
    Coverr (https://coverr.co) - free HD/4K stock videos,
    subject to Coverr license terms (https://coverr.co/license).

    Coverr API notes (based on official docs at api.coverr.co/docs/):
      - 鉴权: Authorization: Bearer <api_key>
      - 搜索端点: GET /videos?query=...,响应结构 {"hits": [...], ...}
      - 加 ?urls=true 在搜索响应里直接返回 mp4 直链
      - URL 是 signed JWT(绑定 API key,无过期时间)
      - Coverr 库以 16:9 横屏为主,9:16 portrait 占比极低(约 1%)
        因此本函数不做 aspect_ratio 过滤,由下游 video.py 的
        resize + letterbox 逻辑统一处理
      - duration 字段同时存在 number 和 string 两种形态,本函数都接受

    本函数使用 urls.mp4_download 字段作为下载地址 —— 按 Coverr 官方文档
    (https://api.coverr.co/docs/videos/#download-a-video) 的说法,
    GET 这个 URL 本身就被 Coverr 当作一次合法的 download 事件计入统计,
    无需再调用 PATCH /videos/:id/stats/downloads。
    """
    api_key = get_api_key("coverr_api_keys")
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {
        "query": search_term,
        "page_size": 20,
        "urls": "true",
        "sort": "popular",
    }
    query_url = f"https://api.coverr.co/videos?{urlencode(params)}"
    logger.info(f"searching videos: {query_url}, with proxies: {config.proxy}")

    try:
        r = requests.get(
            query_url,
            headers=headers,
            proxies=config.proxy,
            verify=_get_tls_verify(),
            timeout=(30, 60),
        )
        response = r.json()
        video_items: List[MaterialInfo] = []
        item_titles: List[str] = []

        if not isinstance(response, dict) or "hits" not in response:
            logger.error(f"search videos failed: {response}")
            return video_items

        for v in response["hits"]:
            # duration 在不同响应里可能是 number(11.625) 或 string("10.500000")
            try:
                duration = int(float(v.get("duration") or 0))
            except (TypeError, ValueError):
                continue
            if duration < minimum_duration:
                continue

            video_id = v.get("id")
            mp4_download_url = (v.get("urls") or {}).get("mp4_download")
            if not video_id or not mp4_download_url:
                continue

            item = MaterialInfo()
            item.provider = "coverr"
            item.url = mp4_download_url
            item.duration = duration
            video_items.append(item)
            item_titles.append(v.get("title", ""))
        return _filter_items_by_subject(
            video_items, item_titles, video_subject, "coverr", is_named_person, subject_noun
        )
    except Exception as e:
        logger.error(f"search videos failed: {str(e)}")

    return []


# 实测拉一段 5s 的 1080p 直链需要 ~82s（YouTube 的 DASH CDN 对区间请求限速），
# 之前 15s 的预算意味着这条路径 100% 超时，每次都回退到渐进式流——也就是
# 永远只能拿到 360p 素材，再花几十秒去做注定失败的放大。宁可在这里多等一会
# 拿到真实的 1080p 像素，也比拿 360p 再插值出"假细节"划算。
_YT_1080P_FETCH_TIMEOUT = 120


def _pick_best_avc1_video_url(video_url: str, max_height: int = 1080) -> str | None:
    """解析出一个纯视频、H.264 编码、mp4 容器、不超过 max_height 的直链。

    只做元数据解析（走 YouTube 自己的 API，返回很快），不涉及从 CDN 拉取
    媒体数据本身，所以不会卡在下面 _download_youtube_clip_1080p 那种大文件
    区间拉取可能遇到的网络问题上。
    """
    import yt_dlp

    try:
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
            info = ydl.extract_info(video_url, download=False)
    except Exception as e:
        logger.debug(f"youtube: failed to resolve formats for '{video_url}': {e}")
        return None

    candidates = [
        f
        for f in (info or {}).get("formats") or []
        if f.get("ext") == "mp4"
        and f.get("acodec") == "none"
        and str(f.get("vcodec") or "").startswith("avc1")
        and (f.get("height") or 0) <= max_height
        and f.get("url")
    ]
    if not candidates:
        return None

    best = max(candidates, key=lambda f: f.get("height") or 0)
    return best["url"]


def _download_youtube_clip_1080p(
    video_url: str, start: float, end: float, output_path: str
) -> bool:
    """直接用 ffmpeg 截取一段 1080p 纯视频流，绕开 yt-dlp 的 external
    downloader 机制。

    yt-dlp 遇到 download_ranges + 需要合并/裁切的场景时，会自己拼一条
    ffmpeg 命令去做区间拉取；这条路径在部分网络环境下会在拉取 DASH
    (纯视频流) 时卡死且没有任何超时保护，拉不到东西也不报错，只是
    一直挂着。这里改为自己解析出直链、自己调用 ffmpeg，用
    subprocess.run 自带的 timeout 兜底：超时或失败都会返回 False，
    交给调用方回退到已验证可靠的渐进式流下载。
    """
    video_source_url = _pick_best_avc1_video_url(video_url)
    if not video_source_url:
        return False

    duration = end - start
    if duration <= 0:
        return False

    cmd = [
        utils.get_ffmpeg_binary(),
        "-y",
        "-ss", f"{start:.3f}",
        "-i", video_source_url,
        "-t", f"{duration:.3f}",
        "-an",
        # 直接流拷贝，不重新编码。原本这里转一遍 libx264，既慢又白白多掉一代
        # 画质——素材后面还要经过合成和成片两轮编码，没有理由在下载阶段就先
        # 压一次。实测同一段素材：转码要 82s，流拷贝拿 30s 的片段也只要 96s，
        # 时间几乎全花在连接和拉流上，和片段长度关系不大。
        "-c", "copy",
        "-movflags", "+faststart",
        output_path,
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, timeout=_YT_1080P_FETCH_TIMEOUT
        )
        if result.returncode != 0:
            logger.debug(
                f"youtube: 1080p direct fetch failed, will fall back to "
                f"progressive stream: {(result.stderr or b'')[-300:]}"
            )
            return False
    except subprocess.TimeoutExpired:
        logger.debug(
            "youtube: 1080p direct fetch timed out, falling back to "
            "progressive stream"
        )
        return False
    except Exception as e:
        logger.debug(f"youtube: 1080p direct fetch failed: {e}")
        return False

    return os.path.exists(output_path) and os.path.getsize(output_path) > 0


def _download_youtube_clip_progressive_fallback(
    video_url: str, start: float, end: float, output_path: str
) -> bool:
    """兜底路径：yt-dlp 自带的渐进式流下载，画质通常只有 720p 甚至更低，
    但在各种网络环境下都验证过快速可靠，用作 1080p 直链拉取失败时的保底。
    """
    import yt_dlp

    download_opts = {
        "quiet": True,
        "no_warnings": True,
        "format": "best[height<=1920][ext=mp4]/best[ext=mp4]/best",
        "outtmpl": output_path,
        "download_ranges": yt_dlp.utils.download_range_func(None, [(start, end)]),
        "force_keyframes_at_cuts": True,
        "ffmpeg_location": utils.get_ffmpeg_binary(),
    }
    try:
        with yt_dlp.YoutubeDL(download_opts) as ydl:
            ydl.download([video_url])
    except Exception as e:
        logger.error(f"youtube clip download failed for '{video_url}': {e}")
        return False

    return os.path.exists(output_path) and os.path.getsize(output_path) > 0


# 频道名里出现这些词，基本可以认为是一手来源（发行方/制作方的官方频道）。
# 刻意不放 "tv"/"channel" 这种词——它们在新闻台和搬运号里同样常见，区分不出来。
_OFFICIAL_SOURCE_MARKERS = (
    "official", "originals", "studios", "games", "entertainment", "pictures",
    "records",
)
# 媒体/娱乐新闻台。它们发的是自己重新剪过、压了台标的版本——ET 那条带
# "ET" 角标的预告片就是从这类频道来的。名字里往往也含 "entertainment"，
# 所以必须显式列出来，靠通用词区分不开（Marvel Entertainment 是官方，
# Entertainment Tonight 不是）。
_BROADCASTER_MARKERS = (
    "tonight", "cnn", "bbc", "fox", "nbc", "abc news", "cbs", "tmz", "variety",
    "access hollywood", "extra tv", "e! news", "hollywood reporter", "deadline",
    "ign", "gamespot", "screenrant", "collider", "gamesradar",
)
# 典型的搬运/解说/二创号——画面上大概率压着自己的台标、大字标题或者摄像头小窗。
_DERIVATIVE_SOURCE_MARKERS = (
    "reaction", "react", "breakdown", "explained", "theory", "leak", "leaks",
    "news", "daily", "updates", "fan", "edit", "compilation", "top 10", "top10",
    "everything", "you missed", "recap",
) + _BROADCASTER_MARKERS


def _official_source_rank(uploader: str, title: str) -> int:
    """给来源打个排序用的档位，越小越优先。

    0 = 频道名看着像官方/一手来源
    1 = 看不出来（中性）
    2 = 明显是搬运/解说/二创
    """
    channel = (uploader or "").lower()
    combined = f"{channel} {(title or '').lower()}"

    if any(marker in combined for marker in _DERIVATIVE_SOURCE_MARKERS):
        return 2
    if any(marker in channel for marker in _OFFICIAL_SOURCE_MARKERS):
        return 0
    return 1


def find_best_youtube_video(
    search_term: str, video_subject: str, minimum_duration: int = 0
) -> dict | None:
    """在 YouTube 上找到与 search_term/video_subject 最相关的一条视频。

    标题相关性过滤复用 _extract_subject_keywords/_text_matches_subject。
    截取片段（search_and_download_youtube_clip）和截图（
    extract_screenshot_frames）都基于这里返回的同一条已验证视频，避免
    两者分别独立搜索导致"关键词对得上但画面其实不相关"的问题。
    """
    import yt_dlp

    subject_keywords = _extract_subject_keywords(video_subject)

    search_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": "in_playlist",
        "default_search": "ytsearch8",
    }
    try:
        with yt_dlp.YoutubeDL(search_opts) as ydl:
            info = ydl.extract_info(search_term, download=False)
            entries = (info or {}).get("entries") or []
    except Exception as e:
        logger.error(f"youtube search failed for '{search_term}': {e}")
        return None

    matches = []
    for entry in entries:
        if not entry:
            continue
        title = entry.get("title") or ""
        duration = entry.get("duration") or 0
        video_id = entry.get("id")
        if not video_id or duration < minimum_duration:
            continue
        if subject_keywords and not _text_matches_subject(title, subject_keywords):
            continue
        matches.append({
            "video_id": video_id,
            "title": title,
            "duration": duration,
            "uploader": entry.get("uploader") or entry.get("channel") or "",
            "url": f"https://www.youtube.com/watch?v={video_id}",
        })

    if not matches:
        logger.info(
            f"youtube: no title-relevant results for '{search_term}' "
            f"(subject keywords: {subject_keywords})"
        )
        return None

    # 标题相关的结果里优先选官方/一手来源。搬运号和解说号的画面上通常压着
    # 自己的台标和大字标题，而且往往是二压过的糊画质；官方频道放出来的是
    # 原始素材。之前直接取第一条，等于把这个选择权交给了 YouTube 的排序。
    matches.sort(key=lambda m: _official_source_rank(m["uploader"], m["title"]))
    best = matches[0]
    if best["uploader"]:
        logger.info(
            f"youtube: picked '{best['title']}' from channel '{best['uploader']}'"
        )
    return best

    logger.info(
        f"youtube: no title-relevant results for '{search_term}' "
        f"(subject keywords: {subject_keywords})"
    )
    return None


def _download_youtube_segment(
    video_url: str, start: float, end: float, output_path: str
) -> bool:
    """截取 [start, end) 这段视频，优先走 1080p 直链，失败则回退到渐进式流。"""
    if _download_youtube_clip_1080p(video_url, start, end, output_path):
        return True
    return _download_youtube_clip_progressive_fallback(video_url, start, end, output_path)


def search_and_download_youtube_clip(
    search_term: str,
    minimum_duration: int,
    max_clip_duration: int,
    video_subject: str,
    output_dir: str,
) -> str:
    """
    在 YouTube 上搜索与 search_term 相关的视频，下载其中一小段作为素材。

    与 Pexels/Pixabay 不同：这里的目标就是找到真实拍到主体本人/事件的视频，
    所以标题相关性过滤是必须的，不能跳过（跳过会退化成随便下一个热门视频）。
    只下载一小段而不是整条视频，减小体积、降低使用风险，但不改变这类素材
    本质上是他人版权内容这一事实——调用方必须已经确认接受这一点。
    """
    video = find_best_youtube_video(search_term, video_subject, minimum_duration)
    if not video:
        return ""

    video_id, title, duration, video_url = (
        video["video_id"], video["title"], video["duration"], video["url"],
    )

    # 跳过大概率是片头的开场部分，从视频中段附近截取一小段。
    start = min(20, max(0, duration // 4))
    end = min(duration, start + max_clip_duration)
    if end - start < minimum_duration:
        start = 0
        end = min(duration, max_clip_duration)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"yt-{utils.md5(f'{video_id}-{start}')}.mp4")
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return output_path

    if _download_youtube_segment(video_url, start, end, output_path):
        logger.info(
            f"youtube: downloaded clip from '{title}' ({start}s-{end}s) for '{search_term}'"
        )
        return output_path
    return ""


def fetch_topic_segment(
    video_id: str,
    video_url: str,
    duration: float,
    output_dir: str,
    # 30s 足够切出 6 段 5s 的动态素材，而且实测拉 30s 和拉 15s 的耗时几乎
    # 一样（96s vs 86s，时间基本都花在建连接和拉流上）。之前默认 60s 反而
    # 经常超时，超时就回退到 360p 渐进流——拉得更多，结果却更差。
    segment_seconds: int = 30,
) -> tuple[str, float] | None:
    """把一条已验证主体相关的视频的中段拉到本地，返回 (本地路径, 片段时长)。

    截图和动态片段都从这一个本地文件里切，避免为了两种用途分别下载两次。
    已经下载过的直接复用。
    """
    if duration <= 0:
        return None

    os.makedirs(output_dir, exist_ok=True)

    margin = duration * 0.1
    seg_start = margin
    seg_end = min(duration - margin, seg_start + segment_seconds)
    if seg_end - seg_start < 5:
        seg_start, seg_end = 0, min(duration, segment_seconds)

    local_path = os.path.join(output_dir, f"ssbase-{video_id}.mp4")
    if not (os.path.exists(local_path) and os.path.getsize(local_path) > 0):
        if not _download_youtube_segment(video_url, seg_start, seg_end, local_path):
            return None

    return local_path, seg_end - seg_start


def extract_motion_clips(
    local_path: str,
    segment_duration: float,
    output_dir: str,
    video_id: str,
    count: int = 4,
    clip_duration: int = 5,
) -> List[str]:
    """从已经下载到本地的片段里，均匀切出若干条短的动态片段。

    真实的动态画面永远比"静态图 + Ken Burns 推镜"更有说服力——推镜只是在
    模拟运动，而这里就是原始素材本身。片段本来就已经为抽帧下载好了，切成
    动态片段几乎是白捡的，不需要任何额外下载。

    用流拷贝（-c copy）切割：不重新编码，既快又不会引入额外的画质损失。
    """
    if segment_duration <= 0 or count <= 0:
        return []

    ffmpeg_binary = utils.get_ffmpeg_binary()
    usable = max(0.0, segment_duration - clip_duration)
    clip_paths = []

    for i in range(count):
        start = (usable * i / count) if count > 1 else 0.0
        clip_path = os.path.join(output_dir, f"ssclip-{video_id}-{i}.mp4")
        if os.path.exists(clip_path) and os.path.getsize(clip_path) > 0:
            clip_paths.append(clip_path)
            continue
        cmd = [
            ffmpeg_binary, "-y",
            "-ss", f"{start:.3f}",
            "-i", local_path,
            "-t", str(clip_duration),
            "-an",
            "-c", "copy",
            clip_path,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=30)
            if (
                result.returncode == 0
                and os.path.exists(clip_path)
                and os.path.getsize(clip_path) > 0
            ):
                clip_paths.append(clip_path)
        except Exception as e:
            logger.debug(f"motion clip cut failed at {start:.1f}s for '{video_id}': {e}")

    if clip_paths:
        logger.info(
            f"youtube: cut {len(clip_paths)} motion clip(s) from the downloaded "
            f"segment of '{video_id}'"
        )
    return clip_paths


def extract_screenshot_frames(
    video_id: str,
    video_url: str,
    duration: float,
    output_dir: str,
    max_frames: int = 6,
    segment_seconds: int = 60,
) -> List[str]:
    """从一条已验证主体相关的视频里截取一段本地素材，再从本地均匀抽出若干
    静态截图。

    截图取自视频本身，而不是另外发起一次独立的图片搜索，所以画面内容天然
    和视频主体一致，不会出现关键词对得上、但图片其实是无关素材（比如纯
    天空背景）的问题。抽帧走本地文件而不是对远端直链反复 seek，更稳定也
    更快。
    """
    fetched = fetch_topic_segment(
        video_id, video_url, duration, output_dir, segment_seconds
    )
    if not fetched:
        return []
    local_path, seg_duration = fetched
    ffmpeg_binary = utils.get_ffmpeg_binary()
    frame_paths = []
    for i in range(1, max_frames + 1):
        ts = seg_duration * i / (max_frames + 1)
        frame_path = os.path.join(output_dir, f"ssframe-{video_id}-{i}.jpg")
        if os.path.exists(frame_path) and os.path.getsize(frame_path) > 0:
            frame_paths.append(frame_path)
            continue
        cmd = [
            ffmpeg_binary, "-y",
            "-ss", f"{ts:.3f}",
            "-i", local_path,
            "-frames:v", "1",
            "-q:v", "2",
            frame_path,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=15)
            if (
                result.returncode == 0
                and os.path.exists(frame_path)
                and os.path.getsize(frame_path) > 0
            ):
                frame_paths.append(frame_path)
        except Exception as e:
            logger.debug(
                f"screenshot extraction failed at {ts:.1f}s in local segment "
                f"for '{video_id}': {e}"
            )

    return frame_paths


def download_youtube_videos(
    task_id: str,
    search_terms: List[str],
    video_subject: str,
    audio_duration: float = 0.0,
    max_clip_duration: int = 5,
    material_directory: str = "",
) -> List[str]:
    """按搜索词逐个在 YouTube 上找相关素材，直到覆盖音频时长或用完关键词。"""
    output_dir = material_directory or utils.storage_dir("cache_videos")

    video_paths = []
    total_duration = 0.0
    for search_term in search_terms:
        clip_path = search_and_download_youtube_clip(
            search_term=search_term,
            minimum_duration=max_clip_duration,
            max_clip_duration=max_clip_duration,
            video_subject=video_subject,
            output_dir=output_dir,
        )
        if not clip_path:
            continue
        video_paths.append(clip_path)
        total_duration += max_clip_duration
        if total_duration > audio_duration:
            logger.info(
                f"total duration of downloaded youtube clips: {total_duration} seconds, "
                "skip downloading more"
            )
            break

    logger.success(f"downloaded {len(video_paths)} youtube clips")
    return video_paths


def save_video(video_url: str, save_dir: str = "") -> str:
    if not save_dir:
        save_dir = utils.storage_dir("cache_videos")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    url_without_query = video_url.split("?")[0]
    url_hash = utils.md5(url_without_query)
    video_id = f"vid-{url_hash}"
    video_path = f"{save_dir}/{video_id}.mp4"

    # if video already exists, return the path
    if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
        logger.info(f"video already exists: {video_path}")
        return video_path

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }

    # if video does not exist, download it
    with open(video_path, "wb") as f:
        f.write(
            requests.get(
                video_url,
                headers=headers,
                proxies=config.proxy,
                verify=_get_tls_verify(),
                timeout=(60, 240),
            ).content
        )

    if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
        clip = None
        try:
            clip = VideoFileClip(video_path)
            duration = clip.duration
            fps = clip.fps
            if duration > 0 and fps > 0:
                return video_path
        except Exception as e:
            logger.warning(f"invalid video file: {video_path} => {str(e)}")
            try:
                os.remove(video_path)
            except Exception as remove_error:
                logger.warning(
                    f"failed to remove invalid video file: {video_path}, error: {str(remove_error)}"
                )
        finally:
            if clip is not None:
                try:
                    clip.close()
                except Exception as close_error:
                    logger.warning(
                        f"failed to close video clip: {video_path}, error: {str(close_error)}"
                    )
    return ""


def download_videos(
    task_id: str,
    search_terms: List[str],
    source: str = "pexels",
    video_aspect: VideoAspect = VideoAspect.portrait,
    video_concat_mode: VideoConcatMode = VideoConcatMode.random,
    audio_duration: float = 0.0,
    max_clip_duration: int = 5,
    match_script_order: bool = False,
    video_subject: str = "",
    is_named_person: bool | None = None,
    subject_noun: str | None = None,
) -> List[str]:
    search_videos = search_videos_pexels
    if source == "pixabay":
        search_videos = search_videos_pixabay
    elif source == "coverr":
        search_videos = search_videos_coverr

    material_directory = config.app.get("material_directory", "").strip()
    if material_directory == "task":
        material_directory = utils.task_dir(task_id)
    elif material_directory and not os.path.isdir(material_directory):
        material_directory = ""

    if match_script_order:
        return _download_videos_by_script_order(
            task_id=task_id,
            search_terms=search_terms,
            search_videos=search_videos,
            video_aspect=video_aspect,
            audio_duration=audio_duration,
            max_clip_duration=max_clip_duration,
            material_directory=material_directory,
            video_subject=video_subject,
            is_named_person=is_named_person,
            subject_noun=subject_noun,
        )

    valid_video_items = []
    valid_video_urls = []
    found_duration = 0.0
    for search_term in search_terms:
        video_items = search_videos(
            search_term=search_term,
            minimum_duration=max_clip_duration,
            video_aspect=video_aspect,
            video_subject=video_subject,
            is_named_person=is_named_person,
            subject_noun=subject_noun,
        )
        logger.info(f"found {len(video_items)} videos for '{search_term}'")

        for item in video_items:
            if item.url not in valid_video_urls:
                valid_video_items.append(item)
                valid_video_urls.append(item.url)
                found_duration += item.duration

    logger.info(
        f"found total videos: {len(valid_video_items)}, required duration: {audio_duration} seconds, found duration: {found_duration} seconds"
    )
    video_paths = []

    concat_mode_value = getattr(video_concat_mode, "value", video_concat_mode)
    if concat_mode_value == VideoConcatMode.random.value:
        random.shuffle(valid_video_items)

    total_duration = 0.0
    for item in valid_video_items:
        try:
            logger.info(f"downloading video: {item.url}")
            saved_video_path = save_video(
                video_url=item.url, save_dir=material_directory
            )
            if saved_video_path:
                logger.info(f"video saved: {saved_video_path}")
                video_paths.append(saved_video_path)
                seconds = min(max_clip_duration, item.duration)
                total_duration += seconds
                if total_duration > audio_duration:
                    logger.info(
                        f"total duration of downloaded videos: {total_duration} seconds, skip downloading more"
                    )
                    break
        except Exception as e:
            logger.error(f"failed to download video: {utils.to_json(item)} => {str(e)}")
    logger.success(f"downloaded {len(video_paths)} videos")
    return video_paths


def _download_videos_by_script_order(
    task_id: str,
    search_terms: List[str],
    search_videos,
    video_aspect: VideoAspect,
    audio_duration: float,
    max_clip_duration: int,
    material_directory: str,
    video_subject: str = "",
    is_named_person: bool | None = None,
    subject_noun: str | None = None,
) -> List[str]:
    """
    按脚本文案顺序下载素材。

    默认下载逻辑会把所有关键词的候选素材合并成一个大列表；如果第一个
    关键词返回很多结果，最终下载时可能一直消耗这个关键词的素材，后续
    脚本主题就排不上时间线。这里按关键词分组后轮询下载：
    第 1 轮取每个关键词的第 1 个候选，第 2 轮取每个关键词的第 2 个候选。
    这样在不重写视频合成引擎的前提下，尽量保证素材顺序贴近文案顺序。
    """
    logger.info("downloading videos with script-order material matching")
    candidate_groups = []
    valid_video_urls = set()
    found_duration = 0.0

    for search_term in search_terms:
        video_items = search_videos(
            search_term=search_term,
            minimum_duration=max_clip_duration,
            video_aspect=video_aspect,
            video_subject=video_subject,
            is_named_person=is_named_person,
            subject_noun=subject_noun,
        )
        logger.info(f"found {len(video_items)} videos for '{search_term}'")

        term_items = []
        for item in video_items:
            if item.url in valid_video_urls:
                continue
            term_items.append(item)
            valid_video_urls.add(item.url)
            found_duration += item.duration

        if term_items:
            candidate_groups.append((search_term, term_items))

    logger.info(
        f"found total ordered video candidates: {sum(len(items) for _, items in candidate_groups)}, "
        f"required duration: {audio_duration} seconds, found duration: {found_duration} seconds"
    )

    video_paths = []
    total_duration = 0.0
    candidate_index = 0
    while candidate_groups and total_duration <= audio_duration:
        has_candidate = False
        for search_term, term_items in candidate_groups:
            if candidate_index >= len(term_items):
                continue

            has_candidate = True
            item = term_items[candidate_index]
            try:
                logger.info(
                    f"downloading ordered video for '{search_term}': {item.url}"
                )
                saved_video_path = save_video(
                    video_url=item.url, save_dir=material_directory
                )
                if saved_video_path:
                    logger.info(f"video saved: {saved_video_path}")
                    video_paths.append(saved_video_path)
                    total_duration += min(max_clip_duration, item.duration)
                    if total_duration > audio_duration:
                        logger.info(
                            f"total duration of downloaded videos: {total_duration} seconds, skip downloading more"
                        )
                        break
            except Exception as e:
                logger.error(
                    f"failed to download ordered video: {utils.to_json(item)} => {str(e)}"
                )

        if not has_candidate:
            break
        candidate_index += 1

    logger.success(f"downloaded {len(video_paths)} ordered videos")
    return video_paths


if __name__ == "__main__":
    download_videos(
        "test123", ["Money Exchange Medium"], audio_duration=100, source="pixabay"
    )
