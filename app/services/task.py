import math
import os.path
import re
from os import path

from loguru import logger

from app.config import config
from app.models import const
from app.models.schema import VideoConcatMode, VideoParams
from app.services import (
    ai_visuals,
    llm,
    material,
    quality_gate,
    real_images,
    subtitle,
    twelvelabs,
    video,
    visual_gate,
    voice,
    upload_post,
)
from app.services import state as sm
from app.utils import file_security, utils


def generate_script(task_id, params):
    logger.info("\n\n## generating video script")
    video_script = params.video_script.strip()
    if not video_script:
        video_script = llm.generate_script(
            video_subject=params.video_subject,
            language=params.video_language,
            paragraph_number=params.paragraph_number,
            video_script_prompt=params.video_script_prompt,
            custom_system_prompt=params.custom_system_prompt,
        )
    else:
        logger.debug(f"video script: \n{video_script}")

    if not video_script:
        sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
        logger.error("failed to generate video script.")
        return None

    return video_script


def generate_terms(task_id, params, video_script, video_source=None, subject_classification=None):
    logger.info("\n\n## generating video terms")
    video_terms = params.video_terms
    effective_source = (
        video_source if video_source is not None else params.video_source
    )
    if not video_terms:
        # 开启素材按文案顺序匹配后，关键词本身也必须按脚本叙事顺序生成；
        # 否则后续即使顺序下载和顺序拼接，也只能复用一组全局主题词，
        # 无法改善“后面内容的画面提前出现”的问题。
        #
        # 剪辑点已经落在句子边界上，素材也按顺序铺，所以关键词精确到"每句
        # 话一个"时，画面才真的跟着旁白在讲的内容走，而不只是笼统地贴合
        # 整个主题。
        if params.match_materials_to_script:
            video_terms = llm.generate_sentence_visual_queries(
                video_subject=params.video_subject,
                video_script=video_script,
                video_source=effective_source,
            )

        if not video_terms:
            video_terms = llm.generate_terms(
                video_subject=params.video_subject,
                video_script=video_script,
                amount=8 if params.match_materials_to_script else 5,
                match_script_order=params.match_materials_to_script,
                video_source=effective_source,
                subject_classification=subject_classification,
            )
    else:
        if isinstance(video_terms, str):
            video_terms = [term.strip() for term in re.split(r"[,，]", video_terms)]
        elif isinstance(video_terms, list):
            video_terms = [term.strip() for term in video_terms]
        else:
            raise ValueError("video_terms must be a string or a list of strings.")

        logger.debug(f"video terms: {utils.to_json(video_terms)}")

    if not video_terms:
        sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
        logger.error("failed to generate video terms.")
        return None

    # 可选的 TwelveLabs Marengo 语义重排：未启用时返回原顺序，无任何副作用。
    # 顺序匹配模式下关键词顺序本身就是脚本叙事顺序，必须保持原样，故跳过。
    if not params.match_materials_to_script:
        video_terms = twelvelabs.rerank_terms_by_subject(
            video_subject=params.video_subject,
            search_terms=video_terms,
        )

    return video_terms


def save_script_data(task_id, video_script, video_terms, params):
    script_file = path.join(utils.task_dir(task_id), "script.json")
    script_data = {
        "script": video_script,
        "search_terms": video_terms,
        "params": params,
    }

    with open(script_file, "w", encoding="utf-8") as f:
        f.write(utils.to_json(script_data))


def resolve_custom_audio_file(task_id: str, custom_audio_file: str | None) -> str:
    requested_file = (custom_audio_file or "").strip()
    if not requested_file:
        return ""

    task_dir = utils.task_dir(task_id)
    try:
        return file_security.resolve_path_within_directory(
            task_dir,
            requested_file,
        )
    except ValueError as exc:
        task_dir_error = exc

    server_audio_file = path.realpath(
        requested_file
        if path.isabs(requested_file)
        else path.join(utils.root_dir(), requested_file)
    )
    if not path.isabs(requested_file):
        project_root = path.realpath(utils.root_dir())
        try:
            if path.commonpath([project_root, server_audio_file]) != project_root:
                raise ValueError(
                    "relative custom audio paths must stay within the project directory"
                )
        except ValueError as exc:
            raise ValueError(
                "custom audio file must be task-local or an existing server-side file"
            ) from exc

    if not path.isfile(server_audio_file):
        raise ValueError(
            "custom audio file does not exist or is not a file"
        ) from task_dir_error

    return server_audio_file


def generate_audio(task_id, params, video_script):
    '''
    Generate audio for the video script.
    If a custom audio file is provided, it will be used directly.
    There will be no subtitle maker object returned in this case.
    Otherwise, TTS will be used to generate the audio.
    Returns:
        - audio_file: path to the generated or provided audio file
        - audio_duration: duration of the audio in seconds
        - sub_maker: subtitle maker object if TTS is used, None otherwise
    '''
    logger.info("\n\n## generating audio")
    # /audio 和 /subtitle 请求模型不包含 custom_audio_file，
    # 这里统一做兼容读取，避免直调接口时抛属性错误。
    requested_custom_audio_file = getattr(params, "custom_audio_file", None)
    try:
        custom_audio_file = resolve_custom_audio_file(
            task_id, requested_custom_audio_file
        )
    except ValueError as exc:
        logger.error(
            "custom audio file is invalid, "
            f"task_id: {task_id}, path: {requested_custom_audio_file}, error: {str(exc)}"
        )
        sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
        return None, None, None

    if not custom_audio_file:
        logger.info("no custom audio file provided, using TTS to generate audio.")
        audio_file = path.join(utils.task_dir(task_id), "audio.mp3")
        sub_maker = voice.tts(
            text=video_script,
            voice_name=voice.parse_voice_name(params.voice_name),
            voice_rate=params.voice_rate,
            voice_file=audio_file,
        )
        if sub_maker is None:
            sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
            logger.error(
                """failed to generate audio:
1. check if the language of the voice matches the language of the video script.
2. check if the network is available. If you are in China, it is recommended to use a VPN and enable the global traffic mode.
            """.strip()
            )
            return None, None, None
        audio_duration = math.ceil(voice.get_audio_duration(sub_maker))
        if audio_duration == 0:
            sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
            logger.error("failed to get audio duration.")
            return None, None, None
        return audio_file, audio_duration, sub_maker
    else:
        logger.info(f"using custom audio file: {custom_audio_file}")
        audio_duration = voice.get_audio_duration(custom_audio_file)
        if audio_duration == 0:
            sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
            logger.error("failed to get audio duration from custom audio file.")
            return None, None, None
        return custom_audio_file, audio_duration, None

def generate_subtitle(task_id, params, video_script, sub_maker, audio_file):
    '''
    Generate subtitle for the video script.
    If subtitle generation is disabled or no subtitle maker is provided, it will return an empty string.
    Otherwise, it will generate the subtitle using the specified provider.
    Returns:
        - subtitle_path: path to the generated subtitle file
    '''
    logger.info("\n\n## generating subtitle")
    if not params.subtitle_enabled:
        return ""

    subtitle_path = path.join(utils.task_dir(task_id), "subtitle.srt")
    subtitle_provider = config.app.get("subtitle_provider", "edge").strip().lower()
    logger.info(f"\n\n## generating subtitle, provider: {subtitle_provider}")

    if sub_maker is None and subtitle_provider != "whisper":
        # 自定义音频不会经过 TTS，因此没有 Edge/Azure 等 TTS 返回的
        # sub_maker 时间轴。只有 Whisper 可以直接从音频文件转写字幕；
        # 其他字幕提供方继续保持原有行为，避免生成错误的空时间轴。
        logger.warning(
            "subtitle maker is missing, skip subtitle generation for provider: "
            f"{subtitle_provider}"
        )
        return ""

    subtitle_fallback = False
    if subtitle_provider == "edge":
        voice.create_subtitle(
            text=video_script, sub_maker=sub_maker, subtitle_file=subtitle_path
        )
        if not os.path.exists(subtitle_path):
            subtitle_fallback = True
            logger.warning("subtitle file not found, fallback to whisper")

    if subtitle_provider == "whisper" or subtitle_fallback:
        subtitle.create(audio_file=audio_file, subtitle_file=subtitle_path)
        logger.info("\n\n## correcting subtitle")
        subtitle.correct(subtitle_file=subtitle_path, video_script=video_script)

    subtitle_lines = subtitle.file_to_subtitles(subtitle_path)
    if not subtitle_lines:
        logger.warning(f"subtitle file is invalid: {subtitle_path}")
        return ""

    # 逐词弹出字幕（卡拉OK效果）需要真实的逐词时间轴，只有 TTS 阶段的
    # sub_maker 能提供；Whisper 转写路径没有这份数据，此时逐词字幕功能
    # 会自动不生效，仍然回退到整句字幕，不影响正常生成。
    if sub_maker is not None:
        words_file = path.join(utils.task_dir(task_id), "words.json")
        voice.save_word_timings(sub_maker, words_file)

    return subtitle_path


def _ai_visuals_fallback(task_id, params, video_terms, is_named_person):
    """真实素材来源（YouTube/真实图片）找不到可用结果时的最后兜底。

    默认关闭：用户明确不想要 AI 生成的画面出现在成片里，哪怕只是找不到
    真实素材时的最后一道防线也不行。只有用户在 UI 里显式勾选"允许 AI
    生成兜底"（params.disable_ai_visuals = False）时才会真的生成 AI 画面；
    否则直接判定任务失败，绝不静默产出 AI 画面冒充"生成成功"。
    """
    if params.disable_ai_visuals:
        sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
        logger.error(
            "no real footage or images found for this subject, and AI-generated "
            "visuals are disabled. Try a different/more well-known video subject, "
            "or enable 'Allow AI-generated images as last resort' in settings."
        )
        return None

    downloaded_videos = ai_visuals.generate_ai_visual_clips(
        task_id=task_id,
        search_terms=video_terms,
        video_subject=params.video_subject,
        video_aspect=params.video_aspect,
        max_clip_duration=params.video_clip_duration,
        is_named_person=is_named_person,
    )
    if not downloaded_videos:
        sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
        logger.error(
            "failed to find relevant real materials or generate AI visuals "
            "for this subject. Try a different/more specific video subject."
        )
        return None
    return downloaded_videos


def get_video_materials(
    task_id,
    params,
    video_terms,
    audio_duration,
    video_source=None,
    is_named_person=None,
    subject_noun=None,
):
    # video_source 覆盖参数用于虚构主题的自动切换：task.start() 检测到主题
    # 虚构时，只想临时改变"这次去哪里找素材"，不想连带覆盖 params.video_source
    # 本身——那个字段还要用于任务恢复/展示用户原始选择的素材来源。
    effective_source = video_source if video_source is not None else params.video_source

    if effective_source == "local":
        logger.info("\n\n## preprocess local materials")
        materials = video.preprocess_video(
            materials=params.video_materials, clip_duration=params.video_clip_duration
        )
        if not materials:
            sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
            logger.error(
                "no valid materials found, please check the materials and try again."
            )
            return None
        return [material_info.url for material_info in materials]
    elif effective_source == "ai_visuals":
        logger.info("\n\n## generating AI visual clips (no real-world footage available)")
        downloaded_videos = ai_visuals.generate_ai_visual_clips(
            task_id=task_id,
            search_terms=video_terms,
            video_subject=params.video_subject,
            video_aspect=params.video_aspect,
            max_clip_duration=params.video_clip_duration,
            is_named_person=is_named_person,
        )
        if not downloaded_videos:
            sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
            logger.error("failed to generate AI visual clips.")
            return None
        return downloaded_videos
    elif effective_source == "real_images":
        logger.info("\n\n## sourcing real images (YouTube thumbnails / web search)")
        downloaded_videos = real_images.download_real_image_clips(
            task_id=task_id,
            search_terms=video_terms,
            video_subject=params.video_subject,
            video_aspect=params.video_aspect,
            audio_duration=audio_duration * params.video_count,
            max_clip_duration=params.video_clip_duration,
        )
        if not downloaded_videos:
            logger.warning(
                "no usable real images found for this subject, "
                "falling back to AI-generated visuals"
            )
            return _ai_visuals_fallback(task_id, params, video_terms, is_named_person)
        return downloaded_videos
    elif effective_source == "youtube":
        # 下载他人在 YouTube 上发布的真实内容片段，用于素材库里根本不存在的
        # 具体真实人物/事件画面。这类素材本质上是他人版权内容，只截取短片段
        # 不改变这一点——用户已经在选择这个来源时明确接受相应风险。
        logger.info("\n\n## downloading video clips from YouTube")
        target_duration = audio_duration * params.video_count
        downloaded_videos = material.download_youtube_videos(
            task_id=task_id,
            search_terms=video_terms,
            video_subject=params.video_subject,
            audio_duration=target_duration,
            max_clip_duration=params.video_clip_duration,
        )

        covered_duration = len(downloaded_videos) * params.video_clip_duration
        if covered_duration < target_duration:
            # YouTube 搜索经常因为关键词覆盖不够而凑不满音频时长；与其把已有
            # 片段反复循环播放（观感很廉价），不如先用真实图片补足，AI 生成
            # 留到两者都不够时再用。
            logger.info(
                f"youtube clips only cover {covered_duration:.1f}s of "
                f"{target_duration:.1f}s needed, topping up with real images"
            )
            topup_videos = real_images.download_real_image_clips(
                task_id=task_id,
                search_terms=video_terms,
                video_subject=params.video_subject,
                video_aspect=params.video_aspect,
                audio_duration=target_duration - covered_duration,
                max_clip_duration=params.video_clip_duration,
            )
            downloaded_videos = downloaded_videos + topup_videos

        if not downloaded_videos:
            logger.warning(
                "no usable YouTube clips or real images found for this subject, "
                "falling back to AI-generated visuals"
            )
            return _ai_visuals_fallback(task_id, params, video_terms, is_named_person)
        return downloaded_videos
    else:
        logger.info(f"\n\n## downloading videos from {effective_source}")
        # 顺序匹配模式只在用户显式开启时生效。这里强制素材下载按关键词顺序
        # 轮询，避免某个早期关键词下载太多素材，把后续脚本主题挤出最终时间线。
        downloaded_videos = material.download_videos(
            task_id=task_id,
            search_terms=video_terms,
            source=effective_source,
            video_aspect=params.video_aspect,
            video_concat_mode=(
                VideoConcatMode.sequential
                if params.match_materials_to_script
                else params.video_concat_mode
            ),
            audio_duration=audio_duration * params.video_count,
            max_clip_duration=params.video_clip_duration,
            match_script_order=params.match_materials_to_script,
            video_subject=params.video_subject,
            is_named_person=is_named_person,
            subject_noun=subject_noun,
        )
        if not downloaded_videos:
            sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
            logger.error(
                "failed to download videos, maybe the network is not available. if you are in China, please use a VPN."
            )
            return None
        return downloaded_videos


def _verify_video_file(video_path: str) -> bool:
    """
    渲染流程之前只检查 ffmpeg/moviepy 写出过程是否报错，从不检查产物本身
    事后是否可读。exit code 0 加文件存在，不代表容器已经完整落盘——我们已经
    不止一次在渲染进行中途亲眼看到 "moov atom not found"：文件已经创建、
    体积在增长，但索引还没写完。这里强制重新打开一次产物文件，读取时长，
    读不出来就说明文件还不能被播放器正常打开，不能当作成功产物返回给用户。

    复用 VideoFileClip（而不是自己再去猜 ffprobe 二进制路径）是为了和
    video.py 里其它素材校验逻辑保持同一套探测方式。
    """
    if not video_path or not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
        logger.error(f"video file missing or empty: {video_path}")
        return False

    clip = None
    try:
        clip = video._open_video_clip_quietly(video_path)
        duration = clip.duration
    except Exception as exc:
        logger.error(f"video file failed integrity check: {video_path}, error: {exc}")
        return False
    finally:
        video.close_clip(clip)

    if not duration or duration <= 0:
        logger.error(f"video file has no readable duration: {video_path}")
        return False

    return True


_SRT_TIME_RANGE_RE = re.compile(
    r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)"
)


def _parse_srt_time_range(time_range: str) -> tuple[float, float] | None:
    match = _SRT_TIME_RANGE_RE.search(time_range or "")
    if not match:
        return None
    h1, m1, s1, ms1, h2, m2, s2, ms2 = (int(g) for g in match.groups())
    start = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000
    end = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000
    return start, end


def _get_sentence_durations(subtitle_path: str) -> list[float]:
    """从字幕时间轴推导每句话的时长，用于让画面剪辑点落在句子边界上，
    而不是固定间隔——这样画面切换的节奏会跟着旁白的叙事节奏走，而不是
    和内容语义无关的定时器。

    字幕文件缺失或解析失败时返回空列表，调用方据此回退到原有的固定
    时长剪辑，不影响现有行为。
    """
    if not subtitle_path or not os.path.exists(subtitle_path):
        return []

    durations = []
    for _, time_range, text in subtitle.file_to_subtitles(subtitle_path):
        if not text.strip():
            continue
        parsed = _parse_srt_time_range(time_range)
        if not parsed:
            continue
        start, end = parsed
        duration = end - start
        if duration > 0:
            durations.append(duration)
    return durations


_MIN_UNIQUE_MATERIAL_COVERAGE = 0.6


def _warn_if_materials_too_thin(
    downloaded_videos: list, audio_duration: float, params
) -> None:
    """素材不够铺满旁白时给出明确警告。

    素材不足时，合成阶段会把已有片段循环播放来凑时长——观众看到的就是同样
    几个画面反复出现，这也是"整条 reel 一直在重复"的由来。循环本身是必要的
    兜底（总比黑屏好），但它会把"素材没找够"这件事悄悄掩盖过去，日志里只剩
    一行 looping clips。这里显式把覆盖率算出来讲清楚，免得每次都要靠看成片
    才发现素材不够。
    """
    if not downloaded_videos or audio_duration <= 0:
        return

    clip_duration = params.video_clip_duration or 5
    unique_coverage = len(set(downloaded_videos)) * clip_duration
    ratio = unique_coverage / audio_duration
    if ratio >= _MIN_UNIQUE_MATERIAL_COVERAGE:
        return

    logger.warning(
        f"only {len(set(downloaded_videos))} unique clip(s) covering "
        f"{unique_coverage:.0f}s of {audio_duration:.0f}s narration "
        f"({ratio:.0%}) - the same shots will repeat several times. "
        "Consider a broader subject, more search terms, or allowing more "
        "sources."
    )


def generate_final_videos(
    task_id, params, downloaded_videos, audio_file, subtitle_path
):
    final_video_paths = []
    combined_video_paths = []
    # 多视频生成默认会打散素材以增加差异；但“按文案顺序匹配素材”追求的是
    # 时间线稳定性和可解释性，所以开启后所有输出都使用顺序拼接。
    if params.match_materials_to_script:
        video_concat_mode = VideoConcatMode.sequential
    elif params.video_count == 1:
        video_concat_mode = params.video_concat_mode
    else:
        video_concat_mode = VideoConcatMode.random
    video_transition_mode = params.video_transition_mode
    # 有字幕时间轴就优先按句子边界剪辑，让画面切换跟着叙事节奏走；
    # 没有字幕（禁用字幕或字幕生成失败）时为空列表，combine_videos 会
    # 自动回退到原有的固定时长剪辑。
    sentence_durations = _get_sentence_durations(subtitle_path)

    _progress = 50
    for i in range(params.video_count):
        index = i + 1
        combined_video_path = path.join(
            utils.task_dir(task_id), f"combined-{index}.mp4"
        )
        logger.info(f"\n\n## combining video: {index} => {combined_video_path}")
        video.combine_videos(
            combined_video_path=combined_video_path,
            video_paths=downloaded_videos,
            audio_file=audio_file,
            video_aspect=params.video_aspect,
            video_concat_mode=video_concat_mode,
            video_transition_mode=video_transition_mode,
            max_clip_duration=params.video_clip_duration,
            threads=params.n_threads,
            sentence_durations=sentence_durations,
        )

        _progress += 50 / params.video_count / 2
        sm.state.update_task(task_id, progress=_progress)

        final_video_path = path.join(utils.task_dir(task_id), f"final-{index}.mp4")

        logger.info(f"\n\n## generating video: {index} => {final_video_path}")
        video.generate_video(
            video_path=combined_video_path,
            audio_path=audio_file,
            subtitle_path=subtitle_path,
            output_file=final_video_path,
            params=params,
        )

        if not _verify_video_file(final_video_path):
            # 不要把打不开的文件当成功产物交出去——宁可这一路视频直接失败，
            # 也不要让用户看到进度 100% 却打不开文件。
            logger.error(
                f"video {index} failed post-render integrity check, "
                "not returning it as a successful output"
            )
            continue

        _progress += 50 / params.video_count / 2
        sm.state.update_task(task_id, progress=_progress)

        final_video_paths.append(final_video_path)
        combined_video_paths.append(combined_video_path)

    return final_video_paths, combined_video_paths


def start(task_id, params: VideoParams, stop_at: str = "video"):
    logger.info(f"start task: {task_id}, stop_at: {stop_at}")
    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=5)

    # 1. Generate script
    video_script = generate_script(task_id, params)
    if not video_script or "Error: " in video_script:
        sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
        return

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=10)

    # 一次 LLM 调用同时判断：主题是否虚构、是否围绕一个真实姓名展开、核心
    # 主体词是什么。素材库（Pexels/Pixabay/YouTube）都不可能有虚构对象的
    # 实拍画面，与其继续搜索返回空结果或无关素材，不如提前切换到 AI 图像
    # 生成路径。这里只算出一个"本次实际使用的来源"，不改写 params.video_source
    # 本身——那个字段要留给任务恢复/展示用户当初的原始选择使用。
    subject_classification = None
    effective_video_source = params.video_source
    if params.video_source != "local":
        subject_classification = llm.classify_subject(params.video_subject)
        if subject_classification["is_fictional"]:
            # AI 生成默认关闭：虚构主题（游戏/影视角色等）改走 real_images——
            # YouTube 缩略图和网络图片搜索通常能找到官方预告片截图、剧照等
            # 真实画面，比让免费图像模型凭空"想象"这些角色好得多。只有用户
            # 显式允许 AI 兜底时，才继续用旧的直接切到 ai_visuals 的行为。
            fallback_source = "ai_visuals" if not params.disable_ai_visuals else "real_images"
            logger.info(
                f"subject '{params.video_subject}' looks fictional/has no real-world "
                f"stock footage available; using '{fallback_source}' for this run "
                f"(original selection: '{params.video_source}')"
            )
            effective_video_source = fallback_source

    if stop_at == "script":
        sm.state.update_task(
            task_id, state=const.TASK_STATE_COMPLETE, progress=100, script=video_script
        )
        return {"script": video_script}

    # 2. Generate terms
    video_terms = ""
    if params.video_source != "local":
        video_terms = generate_terms(
            task_id,
            params,
            video_script,
            video_source=effective_video_source,
            subject_classification=subject_classification,
        )
        if not video_terms:
            sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
            return

    save_script_data(task_id, video_script, video_terms, params)

    if stop_at == "terms":
        sm.state.update_task(
            task_id, state=const.TASK_STATE_COMPLETE, progress=100, terms=video_terms
        )
        return {"script": video_script, "terms": video_terms}

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=20)

    # 3. Generate audio
    audio_file, audio_duration, sub_maker = generate_audio(
        task_id, params, video_script
    )
    if not audio_file:
        sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
        return

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=30)

    if stop_at == "audio":
        sm.state.update_task(
            task_id,
            state=const.TASK_STATE_COMPLETE,
            progress=100,
            audio_file=audio_file,
        )
        return {"audio_file": audio_file, "audio_duration": audio_duration}

    # 4. Generate subtitle
    subtitle_path = generate_subtitle(
        task_id, params, video_script, sub_maker, audio_file
    )

    if stop_at == "subtitle":
        sm.state.update_task(
            task_id,
            state=const.TASK_STATE_COMPLETE,
            progress=100,
            subtitle_path=subtitle_path,
        )
        return {"subtitle_path": subtitle_path}

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=40)

    # 5. Get video materials
    downloaded_videos = get_video_materials(
        task_id,
        params,
        video_terms,
        audio_duration,
        video_source=effective_video_source,
        is_named_person=(subject_classification or {}).get("is_named_person"),
        subject_noun=(subject_classification or {}).get("subject_noun"),
    )
    if not downloaded_videos:
        sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
        return

    if effective_video_source != "local":
        # 本地素材是用户手动挑选的，没有"搜索词/主题相关性"这个概念，
        # 跳过视觉相关性检查；其它来源都可能因为文本匹配不准确而混入
        # 画面上完全不相关的素材，用免费视觉模型再做一层把关。
        downloaded_videos = visual_gate.filter_relevant_clips(
            downloaded_videos, params.video_subject
        )
        # 技术质量把关（近黑屏/模糊/重复）不依赖搜索词相关性判断，
        # 本地素材同样可能踩中这些问题，因此对所有来源统一生效。
    downloaded_videos = quality_gate.filter_low_quality_clips(downloaded_videos)
    # 开场镜头决定观众要不要划走，把视觉上最抓眼的一段素材挪到最前面。
    downloaded_videos = quality_gate.rank_by_visual_interest(downloaded_videos)

    _warn_if_materials_too_thin(downloaded_videos, audio_duration, params)

    if stop_at == "materials":
        sm.state.update_task(
            task_id,
            state=const.TASK_STATE_COMPLETE,
            progress=100,
            materials=downloaded_videos,
        )
        return {"materials": downloaded_videos}

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=50)

    # 仅完整视频生成流程才需要处理视频拼接模式；
    # 这样可以避免 /subtitle 和 /audio 这类请求访问不存在的字段。
    if type(params.video_concat_mode) is str:
        params.video_concat_mode = VideoConcatMode(params.video_concat_mode)

    # 6. Generate final videos
    final_video_paths, combined_video_paths = generate_final_videos(
        task_id, params, downloaded_videos, audio_file, subtitle_path
    )

    if not final_video_paths:
        sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
        return

    logger.success(
        f"task {task_id} finished, generated {len(final_video_paths)} videos."
    )

    # 7. Cross-post to social platforms (if enabled)
    cross_post_results = []
    if upload_post.upload_post_service.is_configured() and upload_post.upload_post_service.auto_upload:
        platforms = upload_post.upload_post_service.platforms
        logger.info(f"\n\n## cross-posting videos to {', '.join(platforms)}")

        youtube_extra = None
        if any(p.startswith("youtube") for p in platforms):
            metadata = llm.generate_social_metadata(
                video_subject=params.video_subject,
                video_script=video_script,
                language=params.video_language or "",
                platform="youtube_shorts",
            )
            youtube_extra = {
                "youtube_title": metadata.get("title", params.video_subject),
                "youtube_description": metadata.get("caption", ""),
                "tags": metadata.get("hashtags", []),
                "privacyStatus": upload_post.upload_post_service.youtube_privacy_status,
                "containsSyntheticMedia": True,
            }

        for video_path in final_video_paths:
            result = upload_post.cross_post_video(
                video_path=video_path,
                title=params.video_subject or "Check out this video! #shorts #viral",
                youtube_extra=youtube_extra,
            )
            cross_post_results.append(result)
            if result.get('success'):
                logger.info(f"✅ Cross-posted: {video_path}")
            else:
                logger.warning(f"⚠️ Failed to cross-post: {video_path} - {result.get('error', 'Unknown error')}")

    kwargs = {
        "videos": final_video_paths,
        "combined_videos": combined_video_paths,
        "script": video_script,
        "terms": video_terms,
        "audio_file": audio_file,
        "audio_duration": audio_duration,
        "subtitle_path": subtitle_path,
        "materials": downloaded_videos,
        "cross_post_results": cross_post_results if cross_post_results else None,
    }
    sm.state.update_task(
        task_id, state=const.TASK_STATE_COMPLETE, progress=100, **kwargs
    )
    return kwargs


if __name__ == "__main__":
    task_id = "task_id"
    params = VideoParams(
        video_subject="金钱的作用",
        voice_name="zh-CN-XiaoyiNeural-Female",
        voice_rate=1.0,
    )
    start(task_id, params, stop_at="video")
