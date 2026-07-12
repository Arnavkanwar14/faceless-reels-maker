import os
import random
import re
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
) -> List[MaterialInfo]:
    """
    只保留素材元数据（Pexels 页面 URL / Pixabay tags 等）中真正出现主体关键词的
    结果，丢弃仅靠搜索引擎模糊匹配到的不相关素材。如果没有可判断的主体关键词
    （比如极短或纯符号的主题），则不过滤，避免误伤正常流程。
    """
    if _looks_like_named_person_subject(video_subject):
        return items

    subject_keywords = _extract_subject_keywords(video_subject)
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
            video_items, item_page_urls, video_subject, "pexels"
        )
    except Exception as e:
        logger.error(f"search videos failed: {str(e)}")

    return []


def search_videos_pixabay(
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect = VideoAspect.portrait,
    video_subject: str = "",
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
        return _filter_items_by_subject(video_items, item_tags, video_subject, "pixabay")
    except Exception as e:
        logger.error(f"search videos failed: {str(e)}")

    return []


def search_videos_coverr(
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect = VideoAspect.portrait,
    video_subject: str = "",
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
        return _filter_items_by_subject(video_items, item_titles, video_subject, "coverr")
    except Exception as e:
        logger.error(f"search videos failed: {str(e)}")

    return []


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
        return ""

    candidates = []
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
        candidates.append((video_id, title, duration))

    if not candidates:
        logger.info(
            f"youtube: no title-relevant results for '{search_term}' "
            f"(subject keywords: {subject_keywords})"
        )
        return ""

    video_id, title, duration = candidates[0]
    video_url = f"https://www.youtube.com/watch?v={video_id}"

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
        return ""

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        logger.info(
            f"youtube: downloaded clip from '{title}' ({start}s-{end}s) for '{search_term}'"
        )
        return output_path
    return ""


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
