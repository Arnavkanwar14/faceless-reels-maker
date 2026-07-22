"""Free, royalty-free background music sourcing via Openverse.

Openverse (openverse.org) aggregates CC-licensed audio from Jamendo, Free
Music Archive, ccMixter, etc. Free to search and download, no API key
required. Results are filtered to `license_type=commercial,modification`,
which excludes NonCommercial and NoDerivatives licenses - BGM here gets
volume-adjusted/sidechain-ducked under the narration, which counts as a
modification of the original work, so ND-licensed tracks are not safe to use.

This is NOT a way to obtain actual copyrighted chart hits or specific
trending platform sounds - those are commercially licensed and require the
platform's own audio library (TikTok/Reels/Shorts) to use legally. This
sources royalty-free tracks in similar moods/genres (upbeat, hype, cinematic,
etc.) as a free, legal substitute.
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import List, Optional

import requests
from loguru import logger

from app.utils import utils

_OPENVERSE_API_URL = "https://api.openverse.org/v1/audio/"
_MIN_DURATION_MS = 30_000  # 30s - skip short stingers/sound effects
_MAX_DURATION_MS = 480_000  # 8 min - skip full albums/DJ mixes
_REQUEST_TIMEOUT = 20

# 病毒式短视频里最常见的几类配乐情绪/风格，用于批量建库，也是找不到
# 具体某首歌时的兜底搜索方向。真正的热门歌曲/特定平台原声做不到免费
# 合法下载——这里用同类情绪的免版税音乐作为替代。
VIRAL_MOOD_PRESETS = [
    "upbeat hype trap beat",
    "energetic pop dance",
    "cinematic epic trailer",
    "dramatic build up drop",
    "lofi chill beat",
    "feel good acoustic",
    "dark aggressive bass",
    "emotional uplifting piano",
    "corporate motivational",
    "suspense tension building",
]


def _safe_filename(query: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", query.strip().lower()).strip("-")[:60]
    query_hash = hashlib.md5(query.encode("utf-8")).hexdigest()[:8]
    return f"bgm-{slug}-{query_hash}.mp3"


def search_openverse_music(query: str, max_results: int = 10) -> List[dict]:
    """搜索 Openverse 的免版税音乐，只保留允许商用+允许改编的授权。"""
    try:
        response = requests.get(
            _OPENVERSE_API_URL,
            params={
                "q": query,
                "license_type": "commercial,modification",
                "page_size": max_results,
            },
            timeout=_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        results = response.json().get("results", [])
    except Exception as e:
        logger.warning(f"bgm: Openverse search failed for '{query}': {e}")
        return []

    candidates = []
    for item in results:
        url = item.get("url")
        duration = item.get("duration") or 0
        if not url or not (_MIN_DURATION_MS <= duration <= _MAX_DURATION_MS):
            continue
        candidates.append(item)
    return candidates


def _search_with_fallback(query: str, max_results: int = 10) -> List[dict]:
    """较长/较生僻的多词查询在 Openverse 上经常会 0 命中（比如 "upbeat hype
    trap beat" 整体搜不到，但 "upbeat" 或 "trap beat" 能搜到）。这里从完整
    查询开始，逐步去掉末尾的词再试，直到有结果或词用完。
    """
    words = query.split()
    for end in range(len(words), 0, -1):
        attempt = " ".join(words[:end])
        candidates = search_openverse_music(attempt, max_results=max_results)
        if candidates:
            if attempt != query:
                logger.debug(f"bgm: '{query}' had no matches, used '{attempt}' instead")
            return candidates
    return []


def download_bgm_by_title(query: str) -> Optional[str]:
    """按标题/情绪搜索并下载一首免版税背景音乐，缓存到 resource/songs
    目录。同一个查询词重复调用会直接复用已下载文件，不重复下载。

    找不到任何合法授权的匹配结果时返回 None——调用方应当回退到不使用
    背景音乐，而不是让整条生成流程失败。
    """
    query = (query or "").strip()
    if not query:
        return None

    song_dir = utils.song_dir()
    output_path = os.path.join(song_dir, _safe_filename(query))
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return output_path

    candidates = _search_with_fallback(query, max_results=10)
    if not candidates:
        logger.warning(f"bgm: no royalty-free match found for '{query}'")
        return None

    for candidate in candidates:
        url = candidate.get("url")
        try:
            response = requests.get(url, timeout=60, stream=True)
            response.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 256):
                    f.write(chunk)
            if os.path.exists(output_path) and os.path.getsize(output_path) > 10_000:
                logger.info(
                    f"bgm: downloaded '{candidate.get('title')}' for query "
                    f"'{query}' (license: {candidate.get('license')})"
                )
                return output_path
        except Exception as e:
            logger.debug(f"bgm: failed to download candidate for '{query}': {e}")
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass

    logger.warning(f"bgm: all candidates failed to download for '{query}'")
    return None


def populate_curated_library(target_new_tracks: int = 30) -> int:
    """批量为常见病毒式短视频情绪拉取一批免版税曲目，扩充内置背景音乐库。

    返回本次实际新增的曲目数量。已存在的文件会被跳过，可以安全地重复
    调用来逐步把库填满到目标数量。
    """
    song_dir = utils.song_dir()
    downloaded = 0
    for mood in VIRAL_MOOD_PRESETS:
        if downloaded >= target_new_tracks:
            break
        candidates = _search_with_fallback(mood, max_results=6)
        per_mood_count = 0
        for candidate in candidates:
            if downloaded >= target_new_tracks or per_mood_count >= 3:
                break
            title = candidate.get("title") or mood
            output_path = os.path.join(song_dir, _safe_filename(f"{mood}-{title}"))
            if os.path.exists(output_path):
                continue
            try:
                url = candidate.get("url")
                response = requests.get(url, timeout=60, stream=True)
                response.raise_for_status()
                with open(output_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 256):
                        f.write(chunk)
                if os.path.exists(output_path) and os.path.getsize(output_path) > 10_000:
                    downloaded += 1
                    per_mood_count += 1
                    logger.info(f"bgm library: added '{title}' ({mood})")
                else:
                    os.remove(output_path)
            except Exception as e:
                logger.debug(f"bgm library: failed for '{title}': {e}")
                if os.path.exists(output_path):
                    try:
                        os.remove(output_path)
                    except OSError:
                        pass
    return downloaded
