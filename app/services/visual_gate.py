"""Free vision-model relevance check for downloaded/generated video clips.

Text-metadata filtering (titles, tags, AI prompts) can still let an
unrelated clip through - a fuzzy stock-search match, an AI image that
missed its prompt, a YouTube result whose thumbnail lied about its content.
This does one more check on what's actually on screen: extract a frame,
ask a free vision model (Groq's Llama 4 Scout) whether it plausibly matches
the video's subject.

Fails open everywhere: no Groq API key configured, a network error, an
unparseable answer, or every single clip getting rejected (almost
certainly a broken check, not genuinely all-bad material) all result in
the original clip list passing through unfiltered. This is a quality
bonus, not a hard gate - it should never be the reason a video fails to
generate.
"""

from __future__ import annotations

import base64
import io
import os
import re
import subprocess
from typing import List

from loguru import logger
from openai import OpenAI
from PIL import Image

from app.config import config
from app.utils import utils

# Groq 已经下架了全部 llama vision 模型——llama-4-scout 现在直接返回 404，
# 而这个模块设计成"失败放行"，所以它一直在静默地全量放行，相关性检查等于
# 没有生效。qwen3.6-27b 是目前免费额度里唯一还支持图片输入的模型。
_DEFAULT_VISION_MODEL = "qwen/qwen3.6-27b"
_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
_REQUEST_TIMEOUT = 30

# Groq 免费额度是每天 200k token，而每检查一张图要 ~2.5k，也就是一天只够
# 看 ~80 张。素材一多就会撞上限，撞上之后这个模块只能返回"没检查"，水印
# 就拦不住了。Gemini 走 OpenAI 兼容端点，作为第二道来源顶上，让检查在日常
# 使用中基本不会因为额度而失效。
_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
_DEFAULT_GEMINI_VISION_MODEL = "gemini-3.5-flash-lite"

VERDICT_OK = "ok"
VERDICT_UNRELATED = "unrelated"
# 和主题不矛盾、但画面本身没有任何能指向主题的信息（随便一辆车、一条普通
# 街道、一个认不出是什么的特写）。这类图不算"错"，但放进成片就是废镜头，
# 观众看不出它和主题有什么关系，所以同样要拒掉。
VERDICT_GENERIC = "generic"
VERDICT_WATERMARK = "watermark"
# 检查没能真正跑起来（没配 key、超额、网络错误）。必须和 OK 区分开：把
# "没检查"当成"检查通过"，正是带水印素材混进成片的原因。
VERDICT_UNKNOWN = "unknown"


def _extract_frame_base64(video_path: str, timestamp: float = 1.0) -> str | None:
    """从素材中间抽一帧，缩到 640px 以内后转成 base64 data URI 喂给视觉模型。

    视觉模型按图片尺寸计费：直接发 1080p 原帧每次要 ~3500 token，Groq 免费
    额度一天 200k，几十条素材就能把当天的额度全部耗光（耗光之后这个模块会
    静默失败放行，等于检查形同虚设）。缩到 640px 后单帧只要几百 token，
    而判断主体相关性和角落里的水印都还完全够用。
    """
    ffmpeg_binary = utils.get_ffmpeg_binary()
    frame_path = f"{video_path}.gate-frame.jpg"
    cmd = [
        ffmpeg_binary,
        "-y",
        "-ss", f"{timestamp:.2f}",
        "-i", video_path,
        "-frames:v", "1",
        # 只在长边超过 640 时缩小，本来就小的帧保持原样（-2 保证边长为偶数）。
        "-vf", "scale='min(640,iw)':-2",
        "-q:v", "4",
        frame_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=15)
        if result.returncode != 0 or not os.path.exists(frame_path):
            return None
        with open(frame_path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{data}"
    except Exception as e:
        logger.debug(f"visual_gate: failed to extract frame from {video_path}: {e}")
        return None
    finally:
        if os.path.exists(frame_path):
            try:
                os.remove(frame_path)
            except OSError:
                pass


def _build_prompt(video_subject: str) -> str:
    return (
        "This image was selected as visual material for "
        f"a video about '{video_subject}'.\n\n"
        "Reply with exactly one of these four tokens:\n"
        "WATERMARK - if the image carries ANY third-party "
        "branding baked into it: a watermark, a site or "
        "channel logo, a TV network bug, a username/@handle, "
        "a 'SUBSCRIBE' prompt, or a large overlaid headline/"
        "title treatment (i.e. it is a composed thumbnail or "
        "article header rather than a plain photo or "
        "screenshot). Check all four corners and all edges.\n"
        "UNRELATED - if it carries no such branding, but shows "
        "something clearly belonging to a different subject.\n"
        "COLLAGE - if the image is not one single photograph but a "
        "grid, collage, montage, split-screen or side-by-side "
        "comparison built from several separate images, or has "
        "panels divided by hard straight borders.\n"
        "GENERIC - if it is plausibly on-topic but contains "
        "nothing that actually identifies this specific subject: "
        "an ordinary vehicle, building, street, sky, crowd or "
        "landscape that could come from almost anywhere, or a "
        "close-up too tight to tell what it is. Ask yourself "
        "whether a viewer seeing ONLY this image would recognise "
        f"it as being about '{video_subject}'. If they could not, "
        "answer GENERIC.\n"
        "OK - only if the image is a single unbroken photo or "
        "screenshot showing something specific and recognisable "
        "about the subject.\n\n"
        "Answer with the single token only."
    )


def _ask_vision_provider(
    data_uri: str, video_subject: str, api_key: str, base_url: str, model: str, **extra
) -> str | None:
    """向一个 OpenAI 兼容的视觉服务提问，返回原始回答；调用失败返回 None。"""
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _build_prompt(video_subject)},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ],
        max_tokens=10,
        timeout=_REQUEST_TIMEOUT,
        **extra,
    )
    return (response.choices[0].message.content or "").strip().upper()


def _classify_data_uri(data_uri: str, video_subject: str, label: str) -> str:
    """对一张已经编码好的图片做"可用 / 不相关 / 带水印"三分类。

    先用 Groq（免费额度更宽松），额度用尽/报错时自动切到 Gemini。两家都
    用不了才返回 UNKNOWN，由调用方决定放行还是丢弃。
    """
    if not data_uri:
        return VERDICT_UNKNOWN

    groq_key = config.app.get("groq_api_key", "")
    gemini_key = config.app.get("gemini_vision_api_key", "")

    providers = []
    if groq_key:
        providers.append((
            "groq", groq_key, _GROQ_BASE_URL,
            config.app.get("groq_vision_model_name", "") or _DEFAULT_VISION_MODEL,
            # 推理模型默认会先写一长段 <think>，max_tokens 很小的情况下会导致
            # 答案还没输出就被截断。
            {"reasoning_effort": "none"},
        ))
    if gemini_key:
        providers.append((
            "gemini", gemini_key, _GEMINI_BASE_URL,
            config.app.get("gemini_vision_model_name", "")
            or _DEFAULT_GEMINI_VISION_MODEL,
            {},
        ))

    for name, key, base_url, model, extra in providers:
        try:
            answer = _ask_vision_provider(
                data_uri, video_subject, key, base_url, model, **extra
            )
        except Exception as e:
            logger.debug(f"visual_gate: {name} check failed for {label}: {e}")
            continue

        if not answer:
            continue
        if "WATERMARK" in answer:
            logger.info(
                f"visual_gate: rejected for watermark/branding ({name}): {label}"
            )
            return VERDICT_WATERMARK
        if "UNRELATED" in answer:
            logger.info(
                f"visual_gate: rejected as unrelated to '{video_subject}' "
                f"({name}): {label}"
            )
            return VERDICT_UNRELATED
        if "COLLAGE" in answer:
            # 和 GENERIC 归为同一类处理（都是"不能当成一个独立镜头用"），
            # 但日志里保留真实原因，方便排查素材质量问题。
            logger.info(
                f"visual_gate: rejected as a collage/grid of multiple images "
                f"({name}): {label}"
            )
            return VERDICT_GENERIC
        if "GENERIC" in answer:
            logger.info(
                f"visual_gate: rejected as generic filler - nothing identifies "
                f"'{video_subject}' ({name}): {label}"
            )
            return VERDICT_GENERIC
        return VERDICT_OK

    # 注意返回的是 UNKNOWN 而不是 OK：调用方需要知道这张图"没被验证过"，
    # 才能按自己的风险偏好决定是放行还是丢弃。
    logger.debug(f"visual_gate: no vision provider could check {label}")
    return VERDICT_UNKNOWN


def _encode_image_file(image_path: str, max_side: int = 640) -> str | None:
    """把图片缩到 max_side 以内再编码，控制单次调用的 token 开销。"""
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            if max(img.size) > max_side:
                scale = max_side / max(img.size)
                img = img.resize(
                    (max(1, round(img.width * scale)), max(1, round(img.height * scale))),
                    Image.LANCZOS,
                )
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85)
        data = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{data}"
    except Exception as e:
        logger.debug(f"visual_gate: failed to encode {image_path}: {e}")
        return None


def classify_image(image_path: str, video_subject: str) -> str:
    """对一张静态图片做三分类，用于素材下载阶段就把带水印的图剔掉。

    放在下载后、生成 Ken Burns 片段前检查：与其等成片阶段再回头筛，不如在
    还只是一张图的时候就丢掉，省下后面所有处理开销。
    """
    return _classify_data_uri(
        _encode_image_file(image_path), video_subject, os.path.basename(image_path)
    )


_BATCH_SIZE = 8
_GROQ_MAX_IMAGES_PER_REQUEST = 3


def _parse_batch_answer(answer: str, count: int) -> List[str] | None:
    """从模型回答里解出每张图的判定。解析不出来返回 None，由调用方回退。"""
    verdict_by_token = {
        "WATERMARK": VERDICT_WATERMARK,
        # 拼图和海报都归到"不能当独立镜头用"这一类：它们本身不违规，但都不是
        # 我们要的真实画面。放在 GENERIC 桶里，兜底逻辑就不会把它们捞回来。
        "COLLAGE": VERDICT_GENERIC,
        "POSTER": VERDICT_GENERIC,
        "GENERIC": VERDICT_GENERIC,
        "UNRELATED": VERDICT_UNRELATED,
        "OK": VERDICT_OK,
    }

    # 期望形如 "1:OK 2:WATERMARK 3:GENERIC"，但模型可能加上换行、逗号或者
    # 编号后的空格，所以用正则按 "序号 + 判定词" 成对提取，容错性更好。
    found = re.findall(r"(\d+)\s*[:.\)-]\s*([A-Z]+)", answer.upper())
    if not found:
        return None

    verdicts: List[str] = [VERDICT_UNKNOWN] * count
    for index_text, token in found:
        try:
            index = int(index_text) - 1
        except ValueError:
            continue
        if 0 <= index < count and token in verdict_by_token:
            verdicts[index] = verdict_by_token[token]
    return verdicts


def classify_images_batch(
    image_paths: List[str], video_subject: str
) -> dict[str, str]:
    """一次调用判定多张图片，返回 {路径: 判定}。

    逐张调用意味着几十张图就是几十次请求，Groq 免费额度一天只够 ~80 次，
    素材稍微多一点当天就再也检查不了了。把多张图放进同一次请求里，配额
    消耗和延迟都按批次算而不是按张算。

    批次解析失败时回退到逐张判定，保证结果始终可用。
    """
    verdicts: dict[str, str] = {}
    if not image_paths:
        return verdicts

    providers_available = config.app.get("groq_api_key", "") or config.app.get(
        "gemini_vision_api_key", ""
    )
    if not providers_available:
        return {path: VERDICT_UNKNOWN for path in image_paths}

    for start in range(0, len(image_paths), _BATCH_SIZE):
        batch = image_paths[start : start + _BATCH_SIZE]
        data_uris = [_encode_image_file(p) for p in batch]
        if any(uri is None for uri in data_uris):
            for path in batch:
                verdicts[path] = classify_image(path, video_subject)
            continue

        content: List[dict] = [
            {"type": "text", "text": _build_batch_prompt(video_subject, len(batch))}
        ]
        for uri in data_uris:
            content.append({"type": "image_url", "image_url": {"url": uri}})

        answer = _ask_batch(content, len(batch))
        parsed = _parse_batch_answer(answer or "", len(batch))
        if parsed is None:
            logger.debug(
                "visual_gate: batch answer unparseable, falling back to per-image checks"
            )
            for path in batch:
                verdicts[path] = classify_image(path, video_subject)
            continue

        for path, verdict in zip(batch, parsed):
            verdicts[path] = verdict
            if verdict == VERDICT_WATERMARK:
                logger.info(
                    f"visual_gate: rejected for watermark/branding: "
                    f"{os.path.basename(path)}"
                )
            elif verdict == VERDICT_GENERIC:
                logger.info(
                    f"visual_gate: rejected as generic/collage: {os.path.basename(path)}"
                )
            elif verdict == VERDICT_UNRELATED:
                logger.info(
                    f"visual_gate: rejected as unrelated to '{video_subject}': "
                    f"{os.path.basename(path)}"
                )

    return verdicts


def _build_batch_prompt(video_subject: str, count: int) -> str:
    return (
        f"You are given {count} candidate images, in order, for a video about "
        f"'{video_subject}'.\n\n"
        "Judge EACH image independently and assign exactly one verdict:\n"
        "WATERMARK - carries third-party branding baked in: a watermark, site "
        "or channel logo, TV network bug, username/@handle, 'SUBSCRIBE' prompt, "
        "or a large overlaid headline/title treatment. Check corners and edges.\n"
        "COLLAGE - not one single photo but a grid, montage, split-screen or "
        "side-by-side built from several images.\n"
        "POSTER - promotional art rather than real footage: a movie/game "
        "poster, key art, box art, cover, or any composed promotional image "
        "with a title logo or character line-up. This includes official "
        "posters and fan-made ones. We want actual footage or screenshots, "
        "not marketing art.\n"
        "UNRELATED - shows something clearly belonging to a different subject.\n"
        "GENERIC - plausibly on-topic but shows nothing that identifies this "
        "specific subject (an ordinary vehicle, building, street, sky or "
        "landscape that could be from anywhere, or a close-up too tight to "
        "tell what it is).\n"
        "OK - a single unbroken frame of real footage, a gameplay/screen "
        "capture, or a candid photo showing something specific and "
        f"recognisable about '{video_subject}'.\n\n"
        f"Reply with exactly {count} lines, nothing else, in this form:\n"
        "1:VERDICT\n2:VERDICT\n...\n"
        f"Use the image order given. Output {count} lines."
    )


def _ask_batch(content: List[dict], count: int) -> str | None:
    """按 groq -> gemini 的顺序尝试同一个批次请求。"""
    groq_key = config.app.get("groq_api_key", "")
    gemini_key = config.app.get("gemini_vision_api_key", "")

    providers = []
    # Groq 单次请求最多只接受 3 张图，超过会直接 400。批次比这大时干脆不试，
    # 省掉一次注定失败的往返。
    if groq_key and count <= _GROQ_MAX_IMAGES_PER_REQUEST:
        providers.append((
            "groq", groq_key, _GROQ_BASE_URL,
            config.app.get("groq_vision_model_name", "") or _DEFAULT_VISION_MODEL,
            {"reasoning_effort": "none"},
        ))
    if gemini_key:
        providers.append((
            "gemini", gemini_key, _GEMINI_BASE_URL,
            config.app.get("gemini_vision_model_name", "")
            or _DEFAULT_GEMINI_VISION_MODEL,
            {},
        ))

    for name, key, base_url, model, extra in providers:
        try:
            client = OpenAI(api_key=key, base_url=base_url)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                max_tokens=20 * count + 40,
                timeout=_REQUEST_TIMEOUT * 2,
                **extra,
            )
            answer = (response.choices[0].message.content or "").strip()
            if answer:
                return answer
        except Exception as e:
            logger.debug(f"visual_gate: batch check failed on {name}: {e}")
    return None


def filter_clean_images(
    image_paths: List[str], video_subject: str, strict: bool = False
) -> List[str]:
    """剔除带第三方水印/台标/大标题的图片，返回干净可用的子集。

    带水印的图永远不会被兜底逻辑放回来——这类素材用了就等于把别人的频道
    标识印进成片。相关性误判则保留兜底：全部被判不相关时，把没有水印的
    图整体放行。

    strict 控制"没能完成检查"（没配 key / 超额 / 网络错误）时怎么办：
    - False：放行。适用于本身就比较可信的来源（比如从正片里抽的帧）。
    - True：丢弃。适用于网络图片搜索这种高风险来源——宁可这一轮少几张图，
      也不要在无法核实的情况下把可能带水印的图放进成片。
    """
    if not (
        config.app.get("groq_api_key", "") or config.app.get("gemini_vision_api_key", "")
    ):
        if strict:
            logger.warning(
                "visual_gate: no vision api key configured, cannot verify web "
                "images are watermark-free - skipping them rather than risking it"
            )
            return []
        return image_paths

    verdicts = {path: classify_image(path, video_subject) for path in image_paths}
    clean = [p for p, v in verdicts.items() if v == VERDICT_OK]
    watermarked = [p for p, v in verdicts.items() if v == VERDICT_WATERMARK]
    generic = [p for p, v in verdicts.items() if v == VERDICT_GENERIC]
    unverified = [p for p, v in verdicts.items() if v == VERDICT_UNKNOWN]

    if watermarked:
        logger.info(
            f"visual_gate: dropped {len(watermarked)} watermarked/branded image(s)"
        )
    if generic:
        logger.info(
            f"visual_gate: dropped {len(generic)} generic image(s) that show "
            f"nothing identifying '{video_subject}'"
        )
    if unverified:
        logger.warning(
            f"visual_gate: could not verify {len(unverified)} image(s) "
            f"(api unavailable/quota) - "
            f"{'skipping them' if strict else 'allowing them through'}"
        )

    if strict:
        return clean

    if not clean:
        # 相关性判断可能整体失灵时的兜底顺序：先用"没被判定为水印、也没被判定为
        # 泛泛而谈"的图；实在一张都不剩，泛图也比没有强，但带水印的图永远不放行。
        fallback = [
            p
            for p, v in verdicts.items()
            if v not in (VERDICT_WATERMARK, VERDICT_GENERIC)
        ]
        return fallback or generic
    return clean + unverified


def classify_frame(video_path: str, video_subject: str) -> str:
    """判断一段素材属于三种情况中的哪一种：可用 / 主题不相关 / 带第三方水印。

    区分后两者是有意为之：主题相关性判断可能误伤（提示词、模型抽风），所以
    调用方在"全被拒"时可以放行兜底；而水印是硬性排除项——搬运别人打了标的
    画面等于把对方频道标识印进成片，任何情况下都不该被兜底逻辑放回来。

    Groq 不可用或调用失败时返回"可用"——这一层是锦上添花的质量提升，
    不该因为一个可选的免费服务不可用就拖垮整条素材下载链路。
    """
    return _classify_data_uri(
        _extract_frame_base64(video_path),
        video_subject,
        os.path.basename(video_path),
    )


def filter_relevant_clips(video_paths: List[str], video_subject: str) -> List[str]:
    """对一批素材做视觉过滤，返回通过检查的子集。

    没配 Groq key 时直接原样返回（功能未启用）。

    两类拒绝的兜底策略不同：带水印的素材永远不会被放回来，哪怕最后一条
    素材都不剩；而"主题不相关"在全部素材都被拒时会整体放行——那种情况下
    检测本身出问题的概率，远高于每一条素材都真的文不对题，不值得为一个
    可选的质量检查赔上整次生成。
    """
    if not (
        config.app.get("groq_api_key", "") or config.app.get("gemini_vision_api_key", "")
    ):
        return video_paths

    verdicts = {path: classify_frame(path, video_subject) for path in video_paths}

    watermarked = [p for p, v in verdicts.items() if v == VERDICT_WATERMARK]
    not_watermarked = [p for p, v in verdicts.items() if v != VERDICT_WATERMARK]
    usable = [p for p, v in verdicts.items() if v == VERDICT_OK]

    if watermarked:
        logger.info(
            f"visual_gate: permanently dropped {len(watermarked)} clip(s) "
            "carrying third-party watermarks/branding"
        )

    if not usable:
        # 只把"不相关"的放回来兜底，带水印的仍然排除在外。
        logger.warning(
            "visual_gate: every clip failed the relevance check, which is more "
            "likely a gate malfunction than genuinely all-bad material - "
            "restoring the non-watermarked clips"
        )
        return not_watermarked

    if len(usable) < len(video_paths):
        logger.info(
            f"visual_gate: kept {len(usable)}/{len(video_paths)} clips "
            "after relevance + watermark check"
        )

    return usable
