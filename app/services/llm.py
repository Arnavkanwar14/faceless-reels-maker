import json
import logging
import re
from time import perf_counter
from typing import List

from loguru import logger
from openai import AzureOpenAI, OpenAI
from openai.types.chat import ChatCompletion

from app.config import config
from app.models.llm_provider import DEFAULT_LLM_PROVIDER_ID, get_llm_provider

_max_retries = 5
# Small stopword list used to extract the significant/subject words from a video
# subject string (e.g. "5 surprising facts about octopuses" -> {"octopuses"}).
_SUBJECT_STOPWORDS = frozenset(
    {
        "a", "an", "the", "of", "in", "on", "at", "to", "for", "and", "or", "with",
        "about", "facts", "fact", "surprising", "amazing", "top", "best", "things",
        "thing", "how", "why", "what", "is", "are", "video", "short", "reel",
    }
)


def _extract_subject_keywords(video_subject: str) -> List[str]:
    """提取视频主题中的实义词，用于校验/强制搜索词包含真正的主体名词。

    优先使用主题原文中的专有名词（大写词），它们通常才是真正的主体（人名/
    品牌名），比"controversy"、"reaction"这类泛化话题词更能代表相关性——
    否则任意两个人吵架的素材都能靠"controversy"混过滤器。
    """
    proper_nouns = re.findall(r"\b[A-Z][a-zA-Z]+\b", video_subject or "")
    proper_nouns = [
        w.lower() for w in proper_nouns
        if w.lower() not in _SUBJECT_STOPWORDS and len(w) > 2
    ]
    if proper_nouns:
        return proper_nouns

    words = re.findall(r"[A-Za-z]+", video_subject.lower())
    keywords = [
        w for w in words if w not in _SUBJECT_STOPWORDS and not w.isdigit() and len(w) > 2
    ]
    return keywords or words


def _singular_forms(word: str) -> List[str]:
    """返回一个词的常见单复数变体，用于宽松匹配（如 octopuses -> octopus）。"""
    forms = [word]
    if word.endswith("es") and len(word) > 4:
        forms.append(word[:-2])
    if word.endswith("s") and len(word) > 3:
        forms.append(word[:-1])
    return forms


def _term_contains_subject(term: str, subject_keywords: List[str]) -> bool:
    """粗略匹配（含单复数）判断搜索词是否包含主体关键词，而不仅仅是抽象概念词。"""
    term_lower = term.lower()
    for keyword in subject_keywords:
        if any(form in term_lower for form in _singular_forms(keyword)):
            return True
    return False


def _looks_like_named_person_subject(video_subject: str) -> bool:
    """检测主题是否围绕一个具体真实姓名（如 "Samay Raina"）展开。

    连续两个及以上大写开头的词通常是人名模式。这种主题下，通用素材库根本
    不可能有这个人的真实画面——强制在每个搜索词里塞入他的名字只会让每个
    查询都返回 0 个结果。此时应放行主题化的场景词（棋盘、直播设备、社媒
    通知等），而不是死守字面姓名匹配。
    """
    return bool(re.search(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+\b", video_subject or ""))


def _enforce_subject_in_terms(
    search_terms: List[str],
    video_subject: str,
    video_source: str = "pexels",
    is_named_person: bool | None = None,
    subject_noun: str | None = None,
) -> List[str]:
    """
    防御性兜底：无论 LLM 是否遵守 prompt 约束，都强制每个搜索词包含视频主体的
    实义词，避免抽象概念词（如 "intelligence"、"behavior"）在 Pexels/Pixabay
    的字面关键词搜索中匹配到完全无关的素材（机器人、棋类等）。

    主题是具体真实姓名时，仅在非 YouTube 来源下跳过这层强制——见
    classify_subject。YouTube 恰恰相反，需要真实姓名才能找到这个人的真实画面。

    is_named_person / subject_noun 由调用方传入 classify_subject() 的结果时
    直接使用（更准确）；不传时才退回正则/关键词启发式，保持向后兼容。
    """
    if not search_terms:
        return search_terms

    named_person = (
        is_named_person
        if is_named_person is not None
        else _looks_like_named_person_subject(video_subject)
    )
    if video_source != "youtube" and named_person:
        return search_terms

    # subject_noun 存在时只用它做"是否已经相关"的判定，不再退回
    # _extract_subject_keywords 拆出的全部实义词——那份列表里 "science"、
    # "erupt" 这类词单独出现也会被判定为"已经相关"，导致完全不沾边的素材
    # 混进来（例如 "the science of how volcanoes erupt" 下，一个只提到
    # "science" 的无关素材会被当作合格结果）。
    primary_subject = subject_noun or _fallback_subject_noun(video_subject)
    required_keywords = [primary_subject] if subject_noun else _extract_subject_keywords(
        video_subject
    )
    if not required_keywords:
        return search_terms

    fixed_terms = []
    for term in search_terms:
        if not isinstance(term, str) or not term.strip():
            continue
        if _term_contains_subject(term, required_keywords):
            fixed_terms.append(term)
        else:
            fixed_terms.append(f"{primary_subject} {term}".strip())
    return fixed_terms


MIN_SCRIPT_PARAGRAPH_NUMBER = 1
MAX_SCRIPT_PARAGRAPH_NUMBER = 10
MAX_SCRIPT_PROMPT_LENGTH = 2000
MAX_SCRIPT_SYSTEM_PROMPT_LENGTH = 8000
_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
_UNCLOSED_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*$", re.IGNORECASE | re.DOTALL)
_URL_USERINFO_RE = re.compile(
    r"((?:https?|wss?)://)([^/\s?#@]*:[^/\s?#@]*@)", re.IGNORECASE
)
_SENSITIVE_QUERY_RE = re.compile(
    r"([?&](?:api[_-]?key|access[_-]?token|token|key|secret|password)=)([^&#\s]+)",
    re.IGNORECASE,
)

DEFAULT_SCRIPT_SYSTEM_PROMPT = """
# Role: Video Script Generator

## Goals:
Generate a script for a video, depending on the subject of the video.

## Constrains:
1. the script is to be returned as a string with the specified number of paragraphs.
2. do not under any circumstance reference this prompt in your response.
3. get straight to the point, don't start with unnecessary things like, "welcome to this video".
4. you must not include any type of markdown or formatting in the script, never use a title.
5. only return the raw content of the script.
6. do not include "voiceover", "narrator" or similar indicators of what should be spoken at the beginning of each paragraph or line.
7. you must not mention the prompt, or anything about the script itself. also, never talk about the amount of paragraphs or lines. just write the script.
8. respond in the same language as the video subject.
""".strip()


def _normalize_text_response(content, llm_provider: str) -> str:
    # 不同 LLM SDK 在异常或被拦截场景下，可能返回 None、空字符串，
    # 甚至返回非字符串对象。这里统一做兜底校验，避免后续直接调用
    # `.replace()` 时抛出 `NoneType` 之类的属性错误。
    if content is None:
        raise ValueError(f"[{llm_provider}] returned empty text content")

    if not isinstance(content, str):
        raise TypeError(
            f"[{llm_provider}] returned non-text content: {type(content).__name__}"
        )

    # MiniMax M3、DeepSeek R1 这类 reasoning 模型可能会把内部推理包在
    # `<think>...</think>` 中返回。视频脚本和关键词只需要最终可朗读文本，
    # 如果不在服务层统一清理，WebUI、字幕和配音都会把思考过程当正文处理。
    content = _THINK_BLOCK_RE.sub("", content)
    content = _UNCLOSED_THINK_BLOCK_RE.sub("", content).strip()
    if not content:
        raise ValueError(f"[{llm_provider}] returned empty text content")

    return content.replace("\n", "")


def _sanitize_error_message(error: object) -> str:
    """
    清理返回给 WebUI/API 的错误信息，避免自定义 base_url 中的凭据泄露。

    一些 OpenAI-compatible SDK 会把请求 URL 原样拼进异常信息。如果用户为了
    代理网关配置了 `https://user:pass@example.com/v1`，直接返回 `str(e)`
    就会把密码暴露给页面、API 调用方或后续日志。这里仅处理错误文案，不改变
    实际请求地址，避免影响正常调用链路。
    """
    message = str(error)
    message = _URL_USERINFO_RE.sub(r"\1***:***@", message)
    message = _SENSITIVE_QUERY_RE.sub(r"\1***", message)
    return message


def _extract_chat_completion_text(response, llm_provider: str) -> str:
    # OpenAI 兼容接口在异常场景下，可能返回没有 choices、
    # 或者 choices/message/content 为空的响应对象。
    # 这里统一做结构校验，避免出现 `NoneType is not subscriptable`
    # 这类底层属性访问错误。
    choices = getattr(response, "choices", None)
    if not choices:
        raise ValueError(f"[{llm_provider}] returned empty choices")

    first_choice = choices[0]
    message = getattr(first_choice, "message", None)
    if message is None:
        raise ValueError(f"[{llm_provider}] returned empty message")

    content = getattr(message, "content", None)
    return _normalize_text_response(content, llm_provider)


def _get_response_field(value, key: str):
    """兼容 dict 和 SDK 响应对象的字段读取。"""
    if isinstance(value, dict):
        return value.get(key)

    try:
        return value[key]
    except (KeyError, TypeError, AttributeError):
        return getattr(value, key, None)


def _extract_qwen_generation_text(response) -> str:
    """
    从 DashScope Generation 响应中提取文本。

    Qwen 使用 `messages` 调用时返回的是 chat 结构：
    `output.choices[0].message.content`；旧 completion 形态才会返回
    `output.text`。这里两个路径都兼容，避免 `output.text` 为 None 时
    继续 `.replace()` 触发不可诊断的 AttributeError。
    """
    output = _get_response_field(response, "output")
    choices = _get_response_field(output, "choices") if output else None
    if choices is not None:
        if not choices:
            logger.warning("Qwen returned an empty choices list")
            raise ValueError("[qwen] returned empty choices")

        first_choice = choices[0]
        message = _get_response_field(first_choice, "message")
        content = _get_response_field(message, "content") if message else None
        if content is not None:
            return _normalize_text_response(content, "qwen")

    text = _get_response_field(output, "text") if output else None
    return _normalize_text_response(text, "qwen")


def _generate_response(prompt: str) -> str:
    try:
        llm_provider = str(
            config.app.get("llm_provider", DEFAULT_LLM_PROVIDER_ID)
        ).lower()
        provider = get_llm_provider(llm_provider)
        if provider is None:
            raise ValueError(f"{llm_provider}: unsupported llm provider")

        logger.info(f"llm provider: {llm_provider}")
        api_key = config.app.get(provider.config_key("api_key"), "")
        configured_model = config.app.get(provider.config_key("model_name"), "")
        model_name = provider.resolve_model_name(configured_model)
        if configured_model and model_name != configured_model:
            logger.warning(
                f"{llm_provider} model '{configured_model}' is deprecated, "
                f"fallback to '{model_name}'"
            )
        configured_base_url = config.app.get(provider.config_key("base_url"), "")
        base_url = provider.resolve_base_url(configured_base_url)
        if configured_base_url and configured_base_url.strip().rstrip("/") in {
            url.rstrip("/") for url in provider.deprecated_base_urls
        }:
            logger.warning(
                f"{llm_provider} base URL '{configured_base_url}' is deprecated, "
                f"fallback to '{base_url}'"
            )
        adapter = provider.adapter
        api_version = ""

        # Ollama 的默认地址依赖当前是否运行在容器中，无法作为静态 Registry
        # 值保存；Registry 仍负责模型和必填规则，运行环境差异在这里解析。
        if llm_provider == "ollama":
            api_key = "ollama"
            if not base_url:
                base_url = config.get_default_ollama_base_url()

        if adapter == "azure":
            api_version = config.app.get(
                provider.config_key("api_version"), "2024-02-15-preview"
            )

        extra_values = {
            field.config_suffix: (
                config.app.get(provider.config_key(field.config_suffix), "")
                or field.default_value
            )
            for field in provider.extra_fields
        }

        if provider.requires_api_key and not api_key:
            raise ValueError(
                f"{llm_provider}: api_key is not set, please set it in the config.toml file."
            )
        if provider.requires_model_name and not model_name:
            raise ValueError(
                f"{llm_provider}: model_name is not set, please set it in the config.toml file."
            )
        if provider.requires_base_url and not base_url:
            raise ValueError(
                f"{llm_provider}: base_url is not set, please set it in the config.toml file."
            )

        for field in provider.extra_fields:
            if field.required and not extra_values[field.config_suffix]:
                raise ValueError(
                    f"{llm_provider}: {field.config_suffix} is not set, "
                    "please set it in the config.toml file."
                )

        if adapter == "qwen":
            import dashscope
            from dashscope.api_entities.dashscope_response import GenerationResponse

            dashscope.api_key = api_key
            response = dashscope.Generation.call(
                model=model_name, messages=[{"role": "user", "content": prompt}]
            )
            if response:
                if isinstance(response, GenerationResponse):
                    status_code = response.status_code
                    if status_code != 200:
                        raise Exception(
                            f'[{llm_provider}] returned an error response: "{response}"'
                        )

                    return _extract_qwen_generation_text(response)
                else:
                    raise Exception(
                        f'[{llm_provider}] returned an invalid response: "{response}"'
                    )
            else:
                raise Exception(f"[{llm_provider}] returned an empty response")

        if adapter == "gemini":
            from google import genai
            from google.genai import types

            http_options = types.HttpOptions(base_url=base_url) if base_url else None
            generation_config = types.GenerateContentConfig(
                temperature=0.5,
                top_p=1,
                top_k=1,
                max_output_tokens=2048,
                safety_settings=[
                    types.SafetySetting(
                        category="HARM_CATEGORY_HARASSMENT",
                        threshold="BLOCK_ONLY_HIGH",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_HATE_SPEECH",
                        threshold="BLOCK_ONLY_HIGH",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        threshold="BLOCK_ONLY_HIGH",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_DANGEROUS_CONTENT",
                        threshold="BLOCK_ONLY_HIGH",
                    ),
                ],
            )

            try:
                # 新版 google-genai 通过统一 Client 暴露模型服务。上下文管理器
                # 会在请求结束后关闭底层 HTTP 连接，避免频繁生成时积累连接资源。
                with genai.Client(
                    api_key=api_key,
                    http_options=http_options,
                ) as client:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=generation_config,
                    )
                generated_text = response.text
            except (AttributeError, IndexError, ValueError) as e:
                logger.warning(f"gemini returned invalid response content: {str(e)}")
                raise ValueError(f"[{llm_provider}] returned invalid response content")

            return _normalize_text_response(generated_text, llm_provider)

        if adapter == "cloudflare_ai_gateway":
            account_id = extra_values["account_id"]
            gateway_id = extra_values["gateway_id"]
            # Cloudflare 当前推荐的 AI Gateway REST API 兼容 OpenAI SDK。
            # Account ID 用于构造统一端点，Gateway ID 通过请求头选择；这里
            # 不再调用 Workers AI 的 /ai/run/{model} 专用接口。
            client = OpenAI(
                api_key=api_key,
                base_url=(
                    f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"
                ),
                default_headers={"cf-aig-gateway-id": gateway_id},
            )
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            return _extract_chat_completion_text(response, llm_provider)

        if adapter == "litellm":
            import litellm

            if not model_name:
                raise ValueError(
                    f"{llm_provider}: model_name is not set, please set it in the config.toml file."
                )

            response = litellm.completion(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                drop_params=True,
            )

            if not response:
                raise ValueError(f"[{llm_provider}] returned empty response")
            if not getattr(response, "choices", None):
                raise ValueError(f"[{llm_provider}] returned empty response")

            return _extract_chat_completion_text(response, llm_provider)

        if adapter == "azure":
            # Azure OpenAI SDK 使用 `azure_endpoint` 和 `api_version` 生成专用请求地址，
            # 不能继续复用下面普通 OpenAI-compatible 的 `base_url` 初始化逻辑。
            # 这里在 Azure 分支内完成请求并立即返回，避免客户端被后续 fallback
            # 覆盖，导致用户配置的 Azure 凭证通过校验但实际请求没有被使用。
            logger.info(f"requesting azure chat completion, model: {model_name}")
            client = AzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=base_url,
            )
            response = client.chat.completions.create(
                model=model_name, messages=[{"role": "user", "content": prompt}]
            )
            if response:
                if isinstance(response, ChatCompletion):
                    return _extract_chat_completion_text(response, llm_provider)
                else:
                    raise Exception(
                        f'[{llm_provider}] returned an invalid response: "{response}", please check your network '
                        f"connection and try again."
                    )
            else:
                raise Exception(
                    f"[{llm_provider}] returned an empty response, please check your network connection and try again."
                )

        if adapter == "modelscope":
            content = ""
            client = OpenAI(
                api_key=api_key,
                base_url=base_url,
            )
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                extra_body={"enable_thinking": False},
                stream=True,
            )
            if response:
                for chunk in response:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        content += delta.content

                if not content.strip():
                    raise ValueError("Empty content in stream response")

                return _normalize_text_response(content, llm_provider)
            else:
                raise Exception(f"[{llm_provider}] returned an empty response")

        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        response = client.chat.completions.create(
            model=model_name, messages=[{"role": "user", "content": prompt}]
        )
        if response:
            if isinstance(response, ChatCompletion):
                return _extract_chat_completion_text(response, llm_provider)
            else:
                raise Exception(
                    f'[{llm_provider}] returned an invalid response: "{response}", please check your network '
                    f"connection and try again."
                )
        else:
            raise Exception(
                f"[{llm_provider}] returned an empty response, please check your network connection and try again."
            )

    except Exception as e:
        return f"Error: {_sanitize_error_message(e)}"


def test_connection() -> tuple[bool, str, float]:
    """
    使用当前 Provider 配置发起一次最小请求，验证实际生成链路是否可用。

    连接测试直接复用 `_generate_response()`，因此会覆盖 API Key、Base URL、
    模型名称和 Provider 专用字段，但不会进入脚本生成的重试逻辑，也不会发送
    用户的视频主题或文案。返回值依次为成功状态、错误信息和请求耗时。
    """
    started_at = perf_counter()
    response = _generate_response(prompt="Reply with exactly: OK")
    elapsed = perf_counter() - started_at

    if not response:
        error_message = "LLM returned an empty response"
        logger.warning(f"llm connection test failed: {error_message}")
        return False, error_message, elapsed

    if response.startswith("Error:"):
        error_message = response.removeprefix("Error:").strip()
        logger.warning(f"llm connection test failed: {error_message}")
        return False, error_message, elapsed

    logger.info(f"llm connection test succeeded, elapsed: {elapsed:.2f}s")
    return True, "", elapsed


def _limit_script_text(text: str | None, max_length: int, field_name: str) -> str:
    value = (text or "").strip()
    if len(value) <= max_length:
        return value

    # API 层已经用 Pydantic 做长度校验；这里继续兜底，是为了保护
    # WebUI 或内部服务直接调用 generate_script 时不会把超长提示词发送给模型，
    # 避免 token 成本异常和请求失败。
    logger.warning(
        f"{field_name} is too long and will be truncated to {max_length} characters."
    )
    return value[:max_length]


def _normalize_script_paragraph_number(paragraph_number: int | None) -> int:
    try:
        value = int(paragraph_number or MIN_SCRIPT_PARAGRAPH_NUMBER)
    except (TypeError, ValueError):
        value = MIN_SCRIPT_PARAGRAPH_NUMBER

    if value < MIN_SCRIPT_PARAGRAPH_NUMBER or value > MAX_SCRIPT_PARAGRAPH_NUMBER:
        # WebUI 和 API 都会限制范围；这里兜底处理内部调用，避免异常参数直接扩大
        # LLM 生成成本或生成空结果。
        logger.warning(
            f"script paragraph_number is out of range and will be clamped: {value}"
        )
        return max(MIN_SCRIPT_PARAGRAPH_NUMBER, min(value, MAX_SCRIPT_PARAGRAPH_NUMBER))

    return value


def build_script_prompt(
    video_subject: str,
    language: str = "",
    paragraph_number: int = 1,
    video_script_prompt: str = "",
    custom_system_prompt: str = "",
) -> str:
    paragraph_number = _normalize_script_paragraph_number(paragraph_number)
    video_script_prompt = _limit_script_text(
        video_script_prompt, MAX_SCRIPT_PROMPT_LENGTH, "video_script_prompt"
    )
    custom_system_prompt = _limit_script_text(
        custom_system_prompt, MAX_SCRIPT_SYSTEM_PROMPT_LENGTH, "custom_system_prompt"
    )

    # 将“脚本生成规则”和“运行时上下文”分开拼接。这样高级用户即使覆盖默认
    # system prompt，也不会漏掉视频主题、语言、段落数这些每次生成都必须带上的参数。
    prompt = custom_system_prompt or DEFAULT_SCRIPT_SYSTEM_PROMPT
    prompt += f"""

# Initialization:
- video subject: {video_subject}
- number of paragraphs: {paragraph_number}
""".rstrip()
    if language:
        prompt += f"\n- language: {language}"
    if video_script_prompt:
        prompt += f"""

# Additional User Requirements:
{video_script_prompt}
""".rstrip()

    return prompt


def generate_script(
    video_subject: str,
    language: str = "",
    paragraph_number: int = 1,
    video_script_prompt: str = "",
    custom_system_prompt: str = "",
) -> str:
    paragraph_number = _normalize_script_paragraph_number(paragraph_number)
    video_script_prompt = _limit_script_text(
        video_script_prompt, MAX_SCRIPT_PROMPT_LENGTH, "video_script_prompt"
    )
    custom_system_prompt = _limit_script_text(
        custom_system_prompt, MAX_SCRIPT_SYSTEM_PROMPT_LENGTH, "custom_system_prompt"
    )
    prompt = build_script_prompt(
        video_subject=video_subject,
        language=language,
        paragraph_number=paragraph_number,
        video_script_prompt=video_script_prompt,
        custom_system_prompt=custom_system_prompt,
    )
    final_script = ""
    logger.info(
        "generating video script: "
        f"subject={video_subject}, paragraph_number={paragraph_number}, "
        f"has_custom_prompt={bool(video_script_prompt.strip())}, "
        f"has_custom_system_prompt={bool(custom_system_prompt.strip())}"
    )

    def format_response(response):
        # Clean the script
        # Remove asterisks, hashes
        response = response.replace("*", "")
        response = response.replace("#", "")

        # Remove markdown syntax
        response = re.sub(r"\[.*\]", "", response)
        response = re.sub(r"\(.*\)", "", response)

        # Split the script into paragraphs
        paragraphs = response.split("\n\n")

        # Select the specified number of paragraphs
        # selected_paragraphs = paragraphs[:paragraph_number]

        # Join the selected paragraphs into a single string
        return "\n\n".join(paragraphs)

    for i in range(_max_retries):
        try:
            response = _generate_response(prompt=prompt)
            if response:
                final_script = format_response(response)
            else:
                logging.error("gpt returned an empty response")

            # Some upstream providers may return quota errors as plain text.
            if final_script and "当日额度已消耗完" in final_script:
                raise ValueError(final_script)

            if final_script:
                break
        except Exception as e:
            logger.error(f"failed to generate script: {e}")

        if i < _max_retries:
            logger.warning(f"failed to generate video script, trying again... {i + 1}")
    if "Error: " in final_script:
        logger.error(f"failed to generate video script: {final_script}")
    else:
        logger.success(f"completed: \n{final_script}")
    return final_script.strip()


def _strip_code_fence(text: str) -> str:
    """Strip a surrounding markdown code fence from an LLM response.

    Non-OpenAI providers (Claude, Gemini, …) frequently wrap JSON output in a
    ```json … ``` fence even when asked to return raw JSON. Removing it lets the
    first json.loads() succeed instead of falling through to the regex recovery
    path (and spuriously logging a warning). Mirrors the DOTALL handling already
    used in _parse_social_metadata().
    """
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def generate_terms(
    video_subject: str,
    video_script: str,
    amount: int = 5,
    match_script_order: bool = False,
    video_source: str = "pexels",
    subject_classification: dict | None = None,
) -> List[str]:
    if video_source == "youtube":
        # YouTube 搜索的目标恰恰相反：这里就是要找到真人真事的实拍画面，所以
        # 必须直接用真实姓名搜索，而不是像 Pexels/Pixabay 那样避开它。
        named_person_rule = (
            "6. if the subject is about a specific real, named individual, EVERY "
            "term must include their actual name combined with a relevant angle "
            '(e.g. "Samay Raina controversy", "Samay Raina chess", "Samay Raina '
            'apology") - these terms are searched directly on YouTube, which does '
            "carry real footage of real people, unlike generic stock libraries."
        )
    else:
        named_person_rule = (
            "6. if the subject is about a specific real, named individual (a public figure,\n"
            "   streamer, YouTuber, celebrity, etc.), do NOT search for their literal name -\n"
            "   general stock-footage libraries (Pexels/Pixabay) do not carry footage of\n"
            "   specific real people and a name-based search will return zero or wrong\n"
            "   results. Instead, generate terms for generic footage that matches the\n"
            '   THEME/SETTING of the story: their content niche (e.g. "chess board closeup",\n'
            '   "esports gaming setup" for a gaming/chess streamer), the medium (e.g.\n'
            '   "youtube interface screen", "person streaming laptop", "webcam recording\n'
            '   setup"), and the emotional tone of the story (e.g. "social media backlash",\n'
            '   "phone notifications", "crowd reaction booing", "breaking news text" for a\n'
            "   controversy). These terms must still each contain a concrete noun from this\n"
            "   list, not abstract words alone."
        )

    if match_script_order:
        goal = (
            f"Generate {amount} chronological stock-video search terms that follow "
            "the order of topics in the video script."
        )
        ordering_rule = (
            "6. keep the terms in the same order as the script narration; "
            "earlier terms must describe earlier visual moments."
        )
        # 有序关键词模式下，示例数量要和 amount 保持一致，避免模型被固定
        # 的 4 个示例误导，导致长文案只返回少量关键词，影响素材覆盖度。
        example_terms = [
            "opening visual topic",
            *[f"script visual topic {index}" for index in range(2, max(amount, 1))],
            "final visual topic",
        ]
        output_example = json.dumps(example_terms[:amount], ensure_ascii=False)
    else:
        goal = (
            f"Generate {amount} search terms for stock videos, depending on the "
            "subject of a video."
        )
        ordering_rule = ""
        output_example = (
            '["search term 1", "search term 2", "search term 3",'
            '"search term 4", "search term 5"]'
        )

    prompt = f"""
# Role: Video Search Terms Generator

## Goals:
{goal}

## Constrains:
1. the search terms are to be returned as a json-array of strings.
2. each search term should consist of 1-3 words, and EVERY term must literally contain
   the main subject noun of the video (e.g. if the subject is "octopuses", every term
   must contain the word "octopus" or "octopuses" - not just relate to it conceptually).
3. you must only return the json-array of strings. you must not return anything else. you must not return the script.
4. the search terms must describe a CONCRETE, PHYSICALLY FILMABLE scene or object -
   these terms are fed directly into a literal stock-footage keyword search (Pexels/Pixabay),
   which has no semantic understanding. Abstract or conceptual words (e.g. "intelligence",
   "behavior", "communication", "strategy") frequently match unrelated stock footage
   (robots, chess, business meetings) because the search engine matches on that word alone.
   Bad: "Octopus Intelligence", "Octopus Behavior", "Octopus Communication".
   Good: "Octopus Underwater", "Octopus Tentacles Closeup", "Octopus Camouflage Rock",
   "Octopus Swimming Ocean".
5. reply with english search terms only.
{named_person_rule}
{ordering_rule}

## Output Example:
{output_example}

## Context:
### Video Subject
{video_subject}

### Video Script
{video_script}

Please note that you must use English for generating video search terms; Chinese is not accepted.
""".strip()

    logger.info(f"subject: {video_subject}, match_script_order: {match_script_order}")

    search_terms = []
    response = ""
    for i in range(_max_retries):
        try:
            response = _generate_response(prompt)
            if "Error: " in response:
                logger.error(f"failed to generate video script: {response}")
                return response
            search_terms = json.loads(_strip_code_fence(response))
            if not isinstance(search_terms, list) or not all(
                isinstance(term, str) for term in search_terms
            ):
                logger.error("response is not a list of strings.")
                continue

        except Exception as e:
            logger.warning(f"failed to generate video terms: {str(e)}")
            if response:
                match = re.search(r"\[.*]", response, re.DOTALL)
                if match:
                    try:
                        search_terms = json.loads(match.group())
                    except Exception as e:
                        # 这里保留重试流程，但必须记录 LLM 返回的非标准 JSON，
                        # 否则后续排查搜索词为空时无法定位
                        # 是模型格式问题还是解析逻辑问题。
                        logger.warning(f"failed to generate video terms: {str(e)}")

        if search_terms and len(search_terms) > 0:
            break
        if i < _max_retries:
            logger.warning(f"failed to generate video terms, trying again... {i + 1}")

    original_terms = list(search_terms)
    search_terms = _enforce_subject_in_terms(
        search_terms,
        video_subject,
        video_source,
        is_named_person=(subject_classification or {}).get("is_named_person"),
        subject_noun=(subject_classification or {}).get("subject_noun"),
    )
    if search_terms != original_terms:
        logger.warning(
            "some generated search terms did not contain the video subject; "
            f"rewrote them for relevance: {original_terms} -> {search_terms}"
        )

    logger.success(f"completed: \n{search_terms}")
    return search_terms


# =============================================================================
# Subject classification
#
# 一次 LLM 调用回答三件事：主题是否虚构（无真实世界画面可能）、是否围绕一个
# 具体真实姓名展开、以及主体核心名词/名字是什么。
#
# 这里把原本分散的两个判断合并成一次调用，而不是各自单独调用一次 LLM：
# 1. 虚构判断（原 is_fictional_subject）：Pexels/Pixabay/YouTube 都不可能有
#    虚构角色的真实画面，需要改用 AI 图像生成。
# 2. 真实姓名判断（原 _looks_like_named_person_subject 的正则版本）：正则
#    "连续两个大写词" 对 "Deep Ocean Creatures"、"Ancient Roman Empire" 这类
#    普通 Title Case 话题一样会误判，错误地跳过相关性过滤/关键词强制。
# 3. subject_noun：直接把"这条视频真正的主体是什么"问出来，而不是用
#    "关键词列表最后一个词"这种启发式去猜——后者在 "octopus behavior in the
#    wild" 这类主题上会选中 "wild" 而不是 "octopus"。
#
# LLM 调用失败时三个字段都退化到原来的正则/启发式实现，保证离线或 LLM 报错
# 时功能仍然可用，只是精度回退。
# =============================================================================


def classify_subject(video_subject: str) -> dict:
    """返回 {"is_fictional": bool, "is_named_person": bool, "subject_noun": str}。"""
    fallback = {
        "is_fictional": False,
        "is_named_person": _looks_like_named_person_subject(video_subject),
        "subject_noun": _fallback_subject_noun(video_subject),
    }

    if not video_subject or not video_subject.strip():
        return fallback

    prompt = f"""
Analyze this video subject and answer three questions:

1. is_fictional: does it reference a FICTIONAL character, creature, franchise \
IP, or brand mascot that has NO real-world photographic or video footage \
available anywhere (e.g. a video game character, an anime/cartoon character, \
a Pokemon, a fictional creature, a brand mascot)?
2. is_named_person: is it specifically about one real, named individual (a \
public figure, streamer, YouTuber, celebrity)? Answer false for generic \
topics even if phrased with capitalized words (e.g. "Deep Ocean Creatures", \
"Ancient Roman Empire" are NOT about a named person).
3. subject_noun: what is the single core subject noun or name of this video? \
(e.g. "octopus" for "octopus behavior in the wild", "Samay Raina" for \
"controversy on YouTuber Samay Raina", "Pikachu" for "Pikachu explains \
electricity")

Respond with ONLY a json object, nothing else:
{{"is_fictional": true|false, "is_named_person": true|false, "subject_noun": "..."}}

Video subject: {video_subject}
""".strip()

    try:
        response = _generate_response(prompt)
        if "Error: " in response:
            logger.warning(f"subject classification failed, using fallback: {response}")
            return fallback
        parsed = json.loads(_strip_code_fence(response))
        if not isinstance(parsed, dict):
            raise ValueError("response is not a json object")
        return {
            "is_fictional": bool(parsed.get("is_fictional", fallback["is_fictional"])),
            "is_named_person": bool(parsed.get("is_named_person", fallback["is_named_person"])),
            "subject_noun": str(parsed.get("subject_noun") or fallback["subject_noun"]).strip()
            or fallback["subject_noun"],
        }
    except Exception as e:
        logger.warning(f"subject classification failed, using fallback: {e}")
        return fallback


def _fallback_subject_noun(video_subject: str) -> str:
    """classify_subject 调用失败时的兜底主体词——优先专有名词，否则拼接全部
    关键词而不是猜一个，避免重蹈 "选中 wild 而不是 octopus" 的覆辙。"""
    keywords = _extract_subject_keywords(video_subject)
    return " ".join(keywords) if keywords else (video_subject or "").strip()


# =============================================================================
# Social publishing metadata
#
# 根据视频主题和脚本生成发布到短视频平台时常用的 title、caption 和 hashtags。
# 这块能力只复用现有 LLM provider，不接入任何外部发布服务，也不影响视频生成主链路。
# =============================================================================

# 不同平台的文案长度和 hashtag 数量偏好不同。这里使用保守上限，避免模型返回
# 过长内容后调用方还需要二次裁剪。
SOCIAL_PLATFORMS = {
    "tiktok": {"title_max": 100, "caption_max": 2200, "hashtag_count": 5},
    "youtube_shorts": {"title_max": 100, "caption_max": 5000, "hashtag_count": 3},
    "instagram_reels": {"title_max": 125, "caption_max": 2200, "hashtag_count": 8},
    "facebook_reels": {"title_max": 125, "caption_max": 2200, "hashtag_count": 5},
}
DEFAULT_SOCIAL_PLATFORM = "tiktok"
DEFAULT_SOCIAL_LANGUAGE = "auto"
MAX_SOCIAL_SUBJECT_LENGTH = 500
MAX_SOCIAL_SCRIPT_LENGTH = 8000
MAX_SOCIAL_LANGUAGE_LENGTH = 64

SOCIAL_PLATFORM_LABELS = {
    "tiktok": "TikTok",
    "youtube_shorts": "YouTube Shorts",
    "instagram_reels": "Instagram Reels",
    "facebook_reels": "Facebook Reels",
}

# LLM 不可用时的通用兜底标签。这里故意不绑定某个国家或语种，保证 API
# 对中文、英文、越南语等不同场景都能返回可用结构。
DEFAULT_SOCIAL_HASHTAGS = [
    "#shorts",
    "#viral",
    "#trending",
    "#fyp",
    "#video",
    "#reels",
    "#creator",
    "#content",
]


def _resolve_social_platform(platform: str | None) -> str:
    value = (platform or "").strip().lower()
    return value if value in SOCIAL_PLATFORMS else DEFAULT_SOCIAL_PLATFORM


def _normalize_social_language(language: str | None) -> str:
    value = (language or DEFAULT_SOCIAL_LANGUAGE).strip()
    if len(value) > MAX_SOCIAL_LANGUAGE_LENGTH:
        logger.warning(
            "social metadata language is too long and will be truncated to "
            f"{MAX_SOCIAL_LANGUAGE_LENGTH} characters."
        )
        value = value[:MAX_SOCIAL_LANGUAGE_LENGTH]
    return value or DEFAULT_SOCIAL_LANGUAGE


def _limit_social_text(text: str | None, max_length: int, field_name: str) -> str:
    value = (text or "").strip()
    if len(value) <= max_length:
        return value

    # API 层会限制长度；这里继续兜底，是为了保护内部调用或未来 WebUI
    # 直接调用时不会把超长内容发送给模型，避免 token 成本异常。
    logger.warning(
        f"{field_name} is too long and will be truncated to {max_length} characters."
    )
    return value[:max_length]


def _social_language_instruction(language: str | None) -> str:
    language = _normalize_social_language(language)
    if language.lower() == DEFAULT_SOCIAL_LANGUAGE:
        return (
            "Use the same language as the video subject and script. If the subject "
            "and script use different languages, prefer the script language."
        )

    return f'Write "title" and "caption" in this language: {language}.'


def _clamp_text(text, max_length: int) -> str:
    value = ("" if text is None else str(text)).strip()
    if max_length and len(value) > max_length:
        return value[:max_length].rstrip()
    return value


def _normalize_hashtags(raw, count: int) -> List[str]:
    """
    将 LLM 返回的 hashtag 统一整理成 `#tag` 格式。

    LLM 可能返回字符串、数组、带空格的词组、重复标签或包含标点的内容。
    这里集中清洗，可以让接口响应结构稳定，也避免平台发布时出现空标签、
    重复标签或不符合常见格式的 hashtag。
    """
    if isinstance(raw, str):
        candidates = re.split(r"[\s,]+", raw)
    elif isinstance(raw, (list, tuple)):
        # 数组里的每一项视为一个完整标签，因此 "du lich" 会变成
        # "#dulich"，而不是拆成两个标签。
        candidates = [str(entry) for entry in raw]
    else:
        candidates = []

    seen = set()
    result: List[str] = []
    for item in candidates:
        tag = re.sub(r"[^\w]", "", item, flags=re.UNICODE)
        if not tag:
            continue
        key = tag.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(f"#{tag}")
        if count and len(result) >= count:
            break
    return result


def build_social_metadata_prompt(
    video_subject: str,
    video_script: str = "",
    language: str = DEFAULT_SOCIAL_LANGUAGE,
    platform: str = DEFAULT_SOCIAL_PLATFORM,
) -> str:
    video_subject = _limit_social_text(
        video_subject, MAX_SOCIAL_SUBJECT_LENGTH, "video_subject"
    )
    video_script = _limit_social_text(
        video_script, MAX_SOCIAL_SCRIPT_LENGTH, "video_script"
    )
    platform = _resolve_social_platform(platform)
    spec = SOCIAL_PLATFORMS[platform]
    label = SOCIAL_PLATFORM_LABELS.get(platform, platform)
    language_instruction = _social_language_instruction(language)

    prompt = f"""
# Role: Short-Video Social Media Copywriter

## Goal
Write engaging publishing metadata for a short video that will be posted on {label}.

## Constraints
1. Respond ONLY with a single valid minified JSON object. No markdown, no code fences, no commentary.
2. The JSON must contain exactly these keys: "title", "caption", "hashtags".
3. "title": a catchy hook, at most {spec["title_max"]} characters.
4. "caption": an engaging description that ends with a call to action, at most {spec["caption_max"]} characters. Do not put hashtags inside the caption.
5. "hashtags": a JSON array of exactly {spec["hashtag_count"]} strings. Each must start with "#", contain no spaces, and be relevant to the topic and to {label}.
6. {language_instruction}

## Output Example
{{"title":"...","caption":"...","hashtags":["#example","#video"]}}

## Context
### Video Subject
{video_subject}

### Video Script
{video_script}
""".strip()
    return prompt


def _parse_social_metadata(response: str, platform: str) -> dict:
    spec = SOCIAL_PLATFORMS[_resolve_social_platform(platform)]

    data = None
    try:
        data = json.loads(_strip_code_fence(response))
    except Exception:
        # 部分模型会在 JSON 外层包一段说明文字或 markdown fence。
        # API 调用方只需要稳定结构，所以这里尝试提取第一个 JSON object。
        match = re.search(r"\{.*\}", response or "", re.DOTALL)
        if match:
            data = json.loads(match.group())

    if not isinstance(data, dict):
        raise ValueError("social metadata response is not a JSON object")

    title = _clamp_text(data.get("title", ""), spec["title_max"])
    caption = _clamp_text(data.get("caption", ""), spec["caption_max"])
    hashtags = _normalize_hashtags(data.get("hashtags", []), spec["hashtag_count"])

    if not title and not caption:
        raise ValueError("social metadata response is missing both title and caption")

    return {"title": title, "caption": caption, "hashtags": hashtags}


def _fallback_social_metadata(
    video_subject: str, video_script: str, platform: str
) -> dict:
    spec = SOCIAL_PLATFORMS[_resolve_social_platform(platform)]
    subject = (video_subject or "").strip()
    script = (video_script or "").strip()

    title = subject
    if not title and script:
        # 没有主题时，用脚本第一句兜底生成 title，避免接口返回空标题。
        title = re.split(r"(?<=[.!?。！？])\s+", script)[0]

    return {
        "title": _clamp_text(title, spec["title_max"]),
        "caption": _clamp_text(script or subject, spec["caption_max"]),
        "hashtags": _normalize_hashtags(DEFAULT_SOCIAL_HASHTAGS, spec["hashtag_count"]),
    }


def generate_social_metadata(
    video_subject: str,
    video_script: str = "",
    language: str = DEFAULT_SOCIAL_LANGUAGE,
    platform: str = DEFAULT_SOCIAL_PLATFORM,
) -> dict:
    """
    生成短视频发布文案元数据。

    返回结构固定为 `{"title": str, "caption": str, "hashtags": List[str]}`。
    如果 LLM 不可用或返回格式异常，会降级为通用启发式结果，保证 API
    调用方始终拿到可展示、可发布前编辑的数据结构。
    """
    platform = _resolve_social_platform(platform)
    language = _normalize_social_language(language)
    video_subject = _limit_social_text(
        video_subject, MAX_SOCIAL_SUBJECT_LENGTH, "video_subject"
    )
    video_script = _limit_social_text(
        video_script, MAX_SOCIAL_SCRIPT_LENGTH, "video_script"
    )
    prompt = build_social_metadata_prompt(
        video_subject=video_subject,
        video_script=video_script,
        language=language,
        platform=platform,
    )
    logger.info(f"generating social metadata: platform={platform}, language={language}")

    response = ""
    for i in range(_max_retries):
        try:
            response = _generate_response(prompt)
            if isinstance(response, str) and "Error: " in response:
                logger.error(f"failed to generate social metadata: {response}")
                break
            metadata = _parse_social_metadata(response, platform)
            logger.success(f"completed: \n{metadata}")
            return metadata
        except Exception as e:
            logger.warning(f"failed to parse social metadata: {str(e)}")

        if i < _max_retries - 1:
            logger.warning(
                f"failed to generate social metadata, trying again... {i + 1}"
            )

    logger.warning("falling back to heuristic social metadata")
    return _fallback_social_metadata(video_subject, video_script, platform)


if __name__ == "__main__":
    video_subject = "生命的意义是什么"
    script = generate_script(
        video_subject=video_subject, language="zh-CN", paragraph_number=1
    )
    print("######################")
    print(script)
    search_terms = generate_terms(
        video_subject=video_subject, video_script=script, amount=5
    )
    print("######################")
    print(search_terms)
